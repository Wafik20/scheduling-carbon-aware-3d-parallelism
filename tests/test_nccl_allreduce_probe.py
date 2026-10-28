import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("allreduce_probe", scripts / "nccl_allreduce_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "environ", dict(module.os.environ))
    monkeypatch.setenv("SLURM_PROCID", "0")
    return module


@pytest.mark.parametrize("code,expected", [(7, 7), (-6, 134)])
def test_supervisor_preserves_failure(probe, monkeypatch, tmp_path, code, expected):
    def run(command, stdout, stderr):
        assert "--worker" in command
        stdout.write("before collective\n")
        stderr.write("Python failure evidence\n")
        (tmp_path / "rank0.nccl.log").write_text(
            "NCCL failure evidence\nNET/IB : Using [0]mlx5_0:1/IB; OOB eno1:10.1.2.3<0>\n"
            "Channel 00/0 : 0 -> 1 via NET/IB/0/GDRDMA\n")
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe, "verify_log", lambda _: pytest.fail("must not verify failed child"))
    assert probe.main(["--result-dir", str(tmp_path)]) == expected
    status = json.loads((tmp_path / "rank0.status.json").read_text())
    assert status["exit_code"] == code
    assert not status["verified"]
    assert status["observed_gdrdma"]
    assert status["socket_selection_lines"] == [
        "NET/IB : Using [0]mlx5_0:1/IB; OOB eno1:10.1.2.3<0>"]
    assert (tmp_path / "rank0.stdout.log").read_text() == "before collective\n"
    assert "Python failure evidence" in (tmp_path / "rank0.stderr.log").read_text()
    assert "NCCL failure evidence" in (tmp_path / "rank0.nccl.log").read_text()
    with pytest.raises(FileExistsError):
        probe.main(["--result-dir", str(tmp_path)])


def test_success_verifies_dedicated_nccl_log(probe, monkeypatch, tmp_path):
    def run(command, stdout, stderr):
        stdout.write("not the NCCL log")
        stderr.write("Python stderr only\n")
        (tmp_path / "rank0.nccl.log").write_text(
            "NCCL INFO Using network IB\nNCCL INFO NET/IB : Using [0]mlx5_0:1/IB\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(probe.subprocess, "run", run)
    assert probe.main(["--result-dir", str(tmp_path)]) == 0
    status = json.loads((tmp_path / "rank0.status.json").read_text())
    assert status["verified"] and status["exit_code"] == 0


def test_verification_failure_retains_child_exit(probe, monkeypatch, tmp_path):
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert probe.main(["--result-dir", str(tmp_path)]) == 1
    status = json.loads((tmp_path / "rank0.status.json").read_text())
    assert status["exit_code"] == 0
    assert "verification_error" in status


def test_host_staging_is_opt_in(probe, monkeypatch, tmp_path):
    monkeypatch.delenv("NCCL_NET_GDR_LEVEL", raising=False)
    log = tmp_path / "rank0.nccl.log"
    probe.configure_worker(False, log)
    assert "NCCL_NET_GDR_LEVEL" not in probe.os.environ
    assert probe.os.environ["NCCL_DEBUG_FILE"] == str(log)
    monkeypatch.setenv("NCCL_NET_GDR_LEVEL", "SYS")
    probe.configure_worker(False, log)
    assert probe.os.environ["NCCL_NET_GDR_LEVEL"] == "SYS"
    probe.configure_worker(True, log)
    assert probe.os.environ["NCCL_NET_GDR_LEVEL"] == "LOC"
    from nccl_builtin_ib_smoke import SETTINGS
    assert "NCCL_NET_GDR_LEVEL" not in SETTINGS


def test_host_staging_checks_actual_channels(probe, tmp_path):
    log = tmp_path / "stderr"
    log.write_text("GDRDMA capability supported\n")
    probe.verify_host_staging(log)
    log.write_text("Channel 00/0 : 0 -> 1 [send] via NET/IB/0/GDRDMA\n")
    with pytest.raises(RuntimeError, match="GDRDMA"):
        probe.verify_host_staging(log)


@pytest.mark.parametrize("correct", [True, False])
def test_payloads(probe, correct, capsys):
    calls, fills, syncs = [], [], []
    class Payload:
        def fill_(self, value):
            fills.append(value)
        def __eq__(self, value):
            assert value == 3
            return SimpleNamespace(item=lambda: correct)
    def empty(size, dtype, device):
        assert dtype == "bf16" and device == "cuda:0"
        calls.append(size)
        return Payload()
    reductions = []
    torch = SimpleNamespace(empty=empty, bfloat16="bf16", all=lambda x: x,
                            cuda=SimpleNamespace(synchronize=lambda: syncs.append(True)))
    def reduce(payload, op, group):
        assert "all_reduce elements=" in capsys.readouterr().out
        reductions.append((op, group))
    dist = SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(SUM="sum"))
    if correct:
        probe.run_payloads(torch, dist, "subgroup", 1)
        assert calls == [1, 1024, 1572864, 4194304]
        assert fills == [2] * 20 and len(syncs) == 20
        assert reductions == [("sum", "subgroup")] * 20
    else:
        with pytest.raises(RuntimeError, match="Incorrect SUM"):
            probe.run_payloads(torch, dist, "subgroup", 0)
        assert fills == [1] and len(syncs) == 1


def test_real_child_signal_preserves_logs(probe, monkeypatch, tmp_path):
    run = probe.subprocess.run
    def abort_child(command, stdout, stderr):
        # Replace only the worker command; exercise real process/file/signal behavior.
        program = ("import os, signal, sys; "
                   "print('before signal', flush=True); "
                   "print('earlier Python stderr', file=sys.stderr, flush=True); "
                   f"log=open({str(tmp_path / 'rank0.nccl.log')!r}, 'w'); "
                   "log.write('NCCL initialization failure detail\\n'); log.flush(); "
                   "print('abort evidence', file=sys.stderr, flush=True); "
                   "os.kill(os.getpid(), signal.SIGTERM)")
        return run([probe.sys.executable, "-c", program], stdout=stdout, stderr=stderr)
    monkeypatch.setattr(probe.subprocess, "run", abort_child)
    assert probe.main(["--result-dir", str(tmp_path)]) == 143
    status = json.loads((tmp_path / "rank0.status.json").read_text())
    assert status["exit_code"] == -15 and not status["observed_gdrdma"]
    assert "before signal" in (tmp_path / "rank0.stdout.log").read_text()
    assert (tmp_path / "rank0.stderr.log").read_text() == "earlier Python stderr\nabort evidence\n"
    assert (tmp_path / "rank0.nccl.log").read_text() == "NCCL initialization failure detail\n"


@pytest.mark.parametrize("override", [{}, {"SLURM_NTASKS": "4"},
                                      {"SLURM_JOB_NUM_NODES": "1"},
                                      {"POWER_GPUS_PER_NODE": "2"},
                                      {"SLURM_LOCALID": "1"}])
def test_layout(probe, monkeypatch, override):
    environment = dict(SLURM_NTASKS="2", SLURM_JOB_NUM_NODES="2",
                       POWER_GPUS_PER_NODE="1", SLURM_LOCALID="0",
                       MASTER_ADDR="host0", MASTER_PORT="20001")
    environment.update(override)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    if override:
        with pytest.raises(ValueError, match="exactly two nodes"):
            probe.validate_layout()
    else:
        assert probe.validate_layout() == 0
        assert probe.os.environ["WORLD_SIZE"] == "2"

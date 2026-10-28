"""CPU-only integration checks for the Slurm profiling OOB preflight."""
import os
from pathlib import Path
import sys

import pytest

from scripts import nccl_oob_preflight as oob

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("world", [2, 4, 8])
def test_profile_oob_uses_cross_node_peer_and_invocation_namespace(monkeypatch, tmp_path, world):
    calls = []
    monkeypatch.setattr(oob, "configure_oob",
                        lambda directory, rank, peer_rank: calls.append((directory, rank, peer_rank)))
    for rank in range(world):
        node, local = divmod(rank, world // 2)
        for key, value in dict(SLURM_PROCID=rank, SLURM_NTASKS=world,
                               SLURM_NODEID=node, SLURM_LOCALID=local,
                               SLURM_JOB_NUM_NODES=2, SLURM_JOB_ID=123,
                               SLURM_STEP_ID=4, PROFILE_MAPPING_RUN="invocation").items():
            monkeypatch.setenv(key, str(value))
        oob.configure_profile_oob(tmp_path)
    directory = tmp_path / "oob_preflight" / "123.4-invocation"
    assert calls == [(directory, rank, (rank + world // 2) % world)
                     for rank in range(world)]


def test_profile_oob_rejects_nonblock_placement_before_exchange(monkeypatch, tmp_path):
    for key, value in dict(SLURM_PROCID=1, SLURM_NTASKS=4, SLURM_NODEID=1,
                           SLURM_LOCALID=0, SLURM_JOB_NUM_NODES=2).items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(oob, "configure_oob", lambda *a, **k: pytest.fail("unexpected exchange"))
    with pytest.raises(ValueError, match="block-placed"):
        oob.configure_profile_oob(tmp_path)


@pytest.mark.parametrize("mode", ["success", "failure", "dry-run"])
def test_profile_launch_requires_oob_after_ib_configuration(tmp_path, monkeypatch, mode):
    import train_profile
    from scripts import memlock_smoke, nccl_builtin_ib_smoke as ib

    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key, value in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="2",
                           MASTER_ADDR="node0", MASTER_PORT="29500").items():
        monkeypatch.setenv(key, value)
    args = ["train_profile.py", "--model", "gpt2", "--profile-menu",
            str(ROOT / "configs/profile_menu_2gpu_tp-dp-pp.json"),
            "--work-dir", str(tmp_path), "--output", str(tmp_path / "out.json"),
            "--slurm", "--in-container", "--num-gpus", "2"]
    if mode == "dry-run":
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    events = []
    transport = dict(NCCL_NET="IB", NCCL_IB_DISABLE="0", NCCL_NET_PLUGIN="none")
    def configure(log):
        events.append("configure")
        os.environ.update(transport)
    def preflight(work):
        assert events == ["memlock", "configure"]
        assert Path(work) == tmp_path
        assert {key: os.environ[key] for key in transport} == transport
        events.append("oob")
        if mode == "failure":
            raise TimeoutError("peer TCP connection timed out")
    def launch(command):
        assert events == ["memlock", "configure", "oob"]
        assert {key: os.environ[key] for key in transport} == transport
        assert command[0] == sys.executable
        assert any(arg.endswith("profile_strategies.py") for arg in command)
        events.append("launch")
        return 0
    monkeypatch.setattr(memlock_smoke, "check_memlock", lambda: events.append("memlock"))
    monkeypatch.setattr(ib, "configure", configure)
    monkeypatch.setattr(oob, "configure_profile_oob", preflight)
    monkeypatch.setattr(ib, "verify_log", lambda log: events.append("verify"))
    monkeypatch.setattr(train_profile.subprocess, "call", launch)
    if mode == "dry-run":
        train_profile.main()
        assert events == []
    elif mode == "failure":
        with pytest.raises(TimeoutError, match="peer TCP"):
            train_profile.main()
        assert events == ["memlock", "configure", "oob"]
    else:
        with pytest.raises(SystemExit) as exc:
            train_profile.main()
        assert exc.value.code == 0
        assert events == ["memlock", "configure", "oob", "launch", "verify"]

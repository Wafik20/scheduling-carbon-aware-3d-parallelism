import importlib.util
from pathlib import Path
import sys
import pytest

@pytest.fixture
def diagnostic(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("ib_diag", scripts / "nccl_builtin_ib_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("rank", [0, 1])
def test_overrides_container_plugin(diagnostic, monkeypatch, tmp_path, capsys, rank):
    monkeypatch.setenv("NCCL_NET_PLUGIN", "spcx")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    log = tmp_path / f"rank{rank}.log"
    diagnostic.configure(log)
    for key, value in diagnostic.SETTINGS.items():
        assert diagnostic.os.environ[key] == value
    assert f"rank={rank}" in capsys.readouterr().out

@pytest.mark.parametrize("text,valid", [
    ("NCCL INFO NET/IB : Using [0]mlx5_3:1/IB [1]mlx5_5:1/IB\nNCCL INFO Using network IB\n", True),
    ("NCCL INFO Using network IB\n", False),
    ("NCCL INFO Using network Socket\n", False),
    ("NCCL INFO Loaded net plugin SPCX (v12)\nUsing network IB\n", False),
    ("NCCL INFO init complete\n", False),
])
def test_actual_log_required(diagnostic, tmp_path, text, valid, capsys):
    log = tmp_path / "nccl.log"
    log.write_text(text)
    if valid:
        diagnostic.verify_log(log)
        output = capsys.readouterr().out
        assert "selected adapters:" in output and "mlx5_3:1" in output
    else:
        with pytest.raises(RuntimeError):
            diagnostic.verify_log(log)


@pytest.mark.parametrize("override", ["", "=mlx5_3:1"])
def test_hca_discovery_or_explicit_override(diagnostic, monkeypatch, tmp_path, override):
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("NCCL_IB_HCA", "=stale:1")
    monkeypatch.setenv("APPTAINERENV_NCCL_IB_HCA", "=stale:1")
    monkeypatch.setenv("SMOKE_NCCL_IB_HCA", override)
    diagnostic.configure(tmp_path / "nccl.log")
    assert "APPTAINERENV_NCCL_IB_HCA" not in diagnostic.os.environ
    assert diagnostic.os.environ.get("NCCL_IB_HCA") == (override or None)

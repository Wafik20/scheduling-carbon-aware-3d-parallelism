"""Eight-GPU matrix and CPU-only validation; never submits GPU jobs."""
import json
import os
from pathlib import Path

import pytest

from experiments.placement_sweep import sweep, validate
from test_placement_validation import synthetic_calibration

FORK = Path(os.environ.get("MEGATRON_FORK", "/vast/cs2731_2026f/wat29/Megatron-LM"))


@pytest.fixture
def generated(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    return sweep.generate(world_size=8)


def test_matrix_exclusions_and_legacy_defaults(generated):
    manifest = sweep.load_manifest(generated)
    assert len(manifest["cases"]) == 364
    assert len(manifest["tasks"]) == 166
    excluded = manifest["excluded_configs"]
    assert len(excluded) == 14
    assert {(e["model"], e["tp"], e["pp"]) for e in excluded} == {
        (m, 8, 1) for m in ("gpt2", "gpt2_large", "llama_2b")
    } | {(m, 1, 8) for m in ("gpt2", "gpt2_large", "llama_2b", "llama_3b")}
    assert all(e["reasons"] for e in excluded)
    assert all(c["tp"] * c["pp"] * c["dp"] == 8 for c in manifest["cases"])
    old = sweep.load_manifest(sweep.generate())
    assert len(old["cases"]) == 162
    assert len(old["tasks"]) == 108
    assert "excluded_configs" not in old
    assert old["gpus_per_node"] == 2


def test_commands_and_dry_run(generated, monkeypatch, capsys):
    manifest = sweep.load_manifest(generated)
    command = sweep.profile_command(manifest, manifest["cases"][0], generated.parent / "attempt")
    assert command[:2] == ["bash", "batch_files/profile_multinode.sbatch"]
    assert manifest["nodes"] == 2 and manifest["gpus_per_node"] == 4
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: pytest.fail("submitted a job"))
    sweep.submit(generated, concurrency=2)
    output = capsys.readouterr().out
    for argument in ("--nodes=2", "--ntasks=8", "--ntasks-per-node=4", "--gres=gpu:4", "%2", "Dry run"):
        assert argument in output


@pytest.mark.skipif(not (FORK / "profile_strategies.py").exists(), reason="external fork unavailable")
def test_actual_fork_all_cases_and_exclusions(generated, monkeypatch):
    manifest = sweep.load_manifest(generated)
    # Absolute menus allow the actual repository launcher to remain the CLI reference.
    for case in manifest["cases"]:
        case["menu"] = str(sweep.ROOT / case["menu"])
    generated.write_text(json.dumps(manifest))
    monkeypatch.setattr(sweep, "ROOT", validate.ROOT)
    report = validate.validate_manifest(generated, FORK)
    assert report["case_count"] == 364
    assert len(report["excluded_configs"]) == 14
    manifest["excluded_configs"][0].update(tp=1, pp=1, dp=8)
    generated.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="supported by fork"):
        validate.validate_manifest(generated, FORK)


def test_eight_gpu_calibration_topology(generated, tmp_path):
    manifest = sweep.load_manifest(generated)
    case = {**manifest, **next(c for c in manifest["cases"] if c["tp"] == c["pp"] == c["dp"] == 2)}
    artifact = synthetic_calibration(case)
    artifact["n_gpus"] = 8
    mapping = []
    for rank in range(8):
        mapping.append(dict(rank=rank, hostname=f"host-{rank // 4}", local_rank=rank % 4,
                            gpu_uuid=f"fixture-{rank}", **{axis + "_group": next(g for g in case["groups"][axis] if rank in g)
                                                          for axis in ("tp", "pp", "dp")}))
    artifact["strategies"][0]["rank_mapping"] = mapping
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))
    validate.validate_calibration(path, case)
    mapping[7]["gpu_uuid"] = mapping[0]["gpu_uuid"]
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="distinct"):
        validate.validate_calibration(path, case)
    mapping[7]["gpu_uuid"] = "fixture-7"
    mapping[7]["hostname"] = "host-0"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="block-distributed"):
        validate.validate_calibration(path, case)


def test_oom_then_sequential_resume_preserves_logs(generated, monkeypatch):
    manifest = sweep.load_manifest(generated)
    index, task = next((i, task) for i, task in enumerate(manifest["tasks"])
                       if len(task["case_ids"]) == 6)
    calls = []
    fail = True

    def launch(command, **kwargs):
        case = next(c for c in manifest["cases"] if c["menu"] == command[command.index("--profile-menu") + 1])
        calls.append(case["id"])
        work = sweep.ROOT / command[command.index("--work-dir") + 1]
        checkpoint = work / "profile_ckpt"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_train_state.pt").write_bytes(b"checkpoint")
        (work / "rank0.log").write_text("CUDA out of memory" if fail else "completed")
        if not fail:
            artifact = synthetic_calibration({**manifest, **case})
            artifact["n_gpus"] = 8
            artifact["strategies"][0]["rank_mapping"] = [
                dict(rank=r, hostname=f"host-{r // 4}", local_rank=r % 4, gpu_uuid=f"fixture-{r}",
                     **{axis + "_group": next(g for g in case["groups"][axis] if r in g)
                        for axis in ("tp", "pp", "dp")}) for r in range(8)]
            sweep.write_json(sweep.ROOT / command[command.index("--output") + 1], artifact)
        return type("Result", (), {"returncode": 1 if fail else 0})()

    monkeypatch.setattr(sweep.subprocess, "run", launch)
    assert sweep.run_task(generated, index) == 1
    fail = False
    assert sweep.run_task(generated, index) == 0
    assert calls == task["case_ids"] * 2
    assert sweep.run_task(generated, index) == 0
    assert calls == task["case_ids"] * 2
    for case_id in task["case_ids"]:
        directory = generated.parent / "cases" / case_id
        assert (directory / "attempts/0001/work/rank0.log").read_text() == "CUDA out of memory"
        assert (directory / "attempts/0002/work/rank0.log").read_text() == "completed"
        assert json.loads((directory / "attempts/0001/status.json").read_text())["status"] == "failed"
        assert json.loads((directory / "status.json").read_text())["status"] == "completed"
    assert not list(generated.parent.rglob("*.pt"))

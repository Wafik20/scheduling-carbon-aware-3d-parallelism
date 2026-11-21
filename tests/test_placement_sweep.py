import json
from pathlib import Path
import pytest
from experiments.placement_sweep import sweep


@pytest.fixture
def generated(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    return sweep.generate("runs/test")


def test_default_matrix_and_commands(generated):
    manifest = sweep.load_manifest(generated)
    assert len(manifest["cases"]) == 162
    assert len(manifest["tasks"]) == 108
    assert len({c["id"] for c in manifest["cases"]}) == 162
    for case in manifest["cases"]:
        menu = json.loads((sweep.ROOT / case["menu"]).read_text())
        assert menu == [{k: case[k] for k in ("tp", "pp", "micro_batch_size", "order")}]
        assert case["tp"] * case["pp"] * case["dp"] == 4
        assert len(case["equivalent_orders"]) in (3, 6)
    for task in manifest["tasks"]:
        rows = [c for c in manifest["cases"] if c["id"] in task["case_ids"]]
        assert len({(c["model"], c["tp"], c["pp"], c["dp"], c["micro_batch_size"]) for c in rows}) == 1
    case = manifest["cases"][0]
    command = sweep.profile_command(manifest, case, generated.parent / "attempt")
    assert command[:2] == ["bash", "batch_files/profile_multinode_4gpu.sbatch"]
    assert command[command.index("--seq-length") + 1] == "1024"
    assert not Path(command[command.index("--output") + 1]).is_absolute()


def test_generation_preserves_manifest(generated):
    assert sweep.generate("runs/test") == generated
    with pytest.raises(ValueError, match="differs"):
        sweep.generate("runs/test", micro_batches=(1,))
    with pytest.raises(ValueError, match="divisible"):
        sweep.generate("runs/invalid", micro_batches=(3,))


def test_submission_is_dry_run_by_default(generated, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    sweep.submit(generated, concurrency=3)
    assert calls == []
    output = capsys.readouterr().out
    assert "--partition=a100" in output and "--constraint=amd" in output
    assert "%3" in output and "Dry run" in output
    sweep.submit(generated, concurrency=3, execute=True)
    assert len(calls) == 1


def test_failed_attempt_retried_and_preserved(generated, monkeypatch):
    class Result:
        returncode = 7
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: Result())
    assert sweep.run_task(generated, 0) == 1
    assert sweep.run_task(generated, 0) == 1
    manifest = sweep.load_manifest(generated)
    directory = sweep.case_dir(generated, manifest["cases"][0])
    assert (directory / "attempts/0001/status.json").exists()
    assert (directory / "attempts/0002/status.json").exists()
    assert json.loads((directory / "status.json").read_text())["status"] == "failed"
    assert sweep.status(generated)["failed"] == 1
    assert sweep.aggregate(generated) == []


def test_resume_revalidates_artifact(generated, monkeypatch):
    from experiments.placement_sweep import validate
    manifest = sweep.load_manifest(generated)
    case = manifest["cases"][0]
    directory = sweep.case_dir(generated, case)
    sweep.write_json(directory / "status.json", {"status": "completed", "calibration": "missing.json"})
    assert sweep.valid_result(generated, manifest, case) is None
    monkeypatch.setattr(validate, "validate_calibration", lambda *args: {})
    assert sweep.valid_result(generated, manifest, case)["status"] == "completed"
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: pytest.fail("completed case reran"))
    assert sweep.run_task(generated, 0) == 0


def test_partial_task_resume_with_real_calibration(generated, monkeypatch):
    from test_placement_validation import synthetic_calibration
    manifest = sweep.load_manifest(generated)
    index, task = next((i, t) for i, t in enumerate(manifest["tasks"]) if len(t["case_ids"]) == 2)
    cases = {c["id"]: c for c in manifest["cases"]}
    calls = []
    fail_second = True

    def launch(command, **kwargs):
        nonlocal fail_second
        case = next(c for c in manifest["cases"] if c["menu"] == command[command.index("--profile-menu") + 1])
        calls.append(case["id"])
        checkpoint = sweep.ROOT / command[command.index("--work-dir") + 1] / "profile_ckpt"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_train_state.pt").write_bytes(b"checkpoint")
        class Result:
            returncode = 0
        if fail_second and case["id"] == task["case_ids"][1]:
            Result.returncode = 1
        else:
            output = sweep.ROOT / command[command.index("--output") + 1]
            sweep.write_json(output, synthetic_calibration({**manifest, **case}))
        return Result()

    monkeypatch.setattr(sweep.subprocess, "run", launch)
    assert sweep.run_task(generated, index) == 1
    fail_second = False
    assert sweep.run_task(generated, index) == 0
    assert calls == [*task["case_ids"], task["case_ids"][1]]
    assert not list(generated.parent.rglob("*.pt"))
    rows = sweep.aggregate(generated)
    assert len(rows) == 2
    assert all(row["energy_per_step_j"] == row["power_w"] * row["step_time_s"] for row in rows)
    directory = sweep.case_dir(generated, cases[task["case_ids"][1]])
    state = json.loads((directory / "status.json").read_text())
    state["status"] = "running"
    sweep.write_json(directory / "status.json", state)
    assert sweep.run_task(generated, index) == 0
    assert (directory / "attempts/0003/status.json").exists()
    assert (directory / "attempts/0002/calibration.json").exists()


def test_checkpoint_cleanup_preserves_artifacts_and_requires_idle(generated):
    import fcntl
    manifest = sweep.load_manifest(generated)
    case = manifest["cases"][0]
    directory = sweep.case_dir(generated, case)
    attempt = directory / "attempts/0001"
    checkpoint = attempt / "work/profile_ckpt/iter_0000000"
    checkpoint.mkdir(parents=True)
    payload = checkpoint / "__0_0.distcp"
    payload.write_bytes(b"checkpoint")
    preserved = [checkpoint / "metadata.json", checkpoint / "run_config.yaml",
                 attempt / "calibration.json", attempt / "power.csv", attempt / "launcher.log"]
    for path in preserved:
        path.write_text("preserve")
    sweep.write_json(attempt / "status.json", {"status": "running"})
    sweep.cleanup_checkpoints(generated)
    assert payload.exists()
    lock_path = generated.parent / "locks" / (case["task_id"] + ".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sweep.cleanup_checkpoints(generated, inactive=True)
        assert payload.exists()
    sweep.cleanup_checkpoints(generated, inactive=True)
    assert not payload.exists()
    assert all(path.read_text() == "preserve" for path in preserved)
    assert json.loads((attempt / "status.json").read_text())["status"] == "running"
    assert sweep.cleanup_checkpoints(generated, inactive=True) == 0


def test_failed_launch_cleans_checkpoint_payloads(generated, monkeypatch):
    def launch(command, **kwargs):
        work = sweep.ROOT / command[command.index("--work-dir") + 1]
        checkpoint = work / "profile_ckpt"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest_train_state.pt").write_bytes(b"checkpoint")
        (work / "nccl-rank0.log").write_text("keep")
        assert "--profile-no-checkpoints" in command
        return type("Result", (), {"returncode": 1})()
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    assert sweep.run_task(generated, 0) == 1
    assert not list(generated.parent.rglob("*.pt"))
    assert list(generated.parent.rglob("nccl-rank0.log"))

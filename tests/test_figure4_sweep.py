import importlib.util
from pathlib import Path
import csv
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_figure4_sweep.py"
spec = importlib.util.spec_from_file_location("figure4_prepare", PATH)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_grid_and_flags(tmp_path):
    root = PATH.parents[1]
    manifest = sweep.build_manifest(root / "runs/workshop_fixed_start_may2024_ready",
                                    root / "data/recovered_workshop_may2024", tmp_path)
    assert len(manifest["runs"]) == 48
    assert manifest["max_offset_h"] == 537.5
    assert [w["deadline_h"] for w in manifest["workloads"]] == [4,8,12,15,19,23,26,30,17,34,50,67,83,100,117,133]
    for run in manifest["runs"]:
        cmd = run["command"]
        assert cmd[cmd.index("--n-episodes") + 1] == "64"
        assert cmd[cmd.index("--num-workers") + 1] == "16"
        assert "--include-job-shift" not in cmd
        assert "--include-rl" not in cmd
    assert [w["token_quantum"] for w in manifest["workloads"][:8]] == [5000000]*3 + [10000000]*4 + [25000000]


def test_dryrun_never_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    sweep.run({"runs": [{"output": str(tmp_path / "new"), "command": ["unused"]}]})
    assert not (tmp_path / "new").exists()


def test_resume_refuses_partial_and_checks_episodes(tmp_path):
    run = dict(output=str(tmp_path), command=["unused"], profile_sha256="p", trace_sha256="t")
    with pytest.raises(ValueError, match="Partial"):
        sweep.result_complete(run)
    sweep.write_json(tmp_path / "status.json", dict(run, status="completed", returncode=0))
    (tmp_path / "summary.csv").write_text("policy,mean\nstatic(a),1\n")
    with (tmp_path / "results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "episode"])
        writer.writeheader()
        for policy in ["static(a)", "Greedy", "Fixed Start Time Dynamic Parallelism"]:
            for episode in range(64):
                writer.writerow(dict(policy=policy, episode=episode))
    assert sweep.result_complete(run)
    run["profile_sha256"] = "changed"
    with pytest.raises(ValueError, match="Mismatched"):
        sweep.result_complete(run)

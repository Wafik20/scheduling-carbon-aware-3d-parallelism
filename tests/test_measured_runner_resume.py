"""Resume integrity and weighted resource admission for measured runs."""
import json
from concurrent.futures import Future

import pytest

from test_measured_model_wrap import runner, prepare_inputs


def completed_workload(runner, prepare_inputs):
    args, _ = prepare_inputs
    w, = runner.prepare(args)
    out = args.out / "gpt2"
    for name in runner.ARTIFACTS:
        (out / name).write_text(json.dumps({"model": "gpt2"}))
    runner.write_completion(w)
    return w, out


def test_resume_requires_matching_inputs_and_all_artifacts(runner, prepare_inputs):
    w, out = completed_workload(runner, prepare_inputs)
    assert runner.cached_summary(w) == {"model": "gpt2"}
    # Regenerating profiles with identical bytes must not invalidate the cache.
    args, _ = prepare_inputs
    regenerated, = runner.prepare(args)
    assert runner.cached_summary(regenerated) == {"model": "gpt2"}
    assert runner.cached_summary(dict(w, tokens=w["tokens"] + 1)) is None
    (out / "schedules.csv").write_text("truncated")
    assert runner.cached_summary(w) is None


@pytest.mark.parametrize("which", ["profile", "trace", "algorithm", "missing", "legacy"])
def test_resume_rejects_changed_sources_and_old_outputs(runner, prepare_inputs, monkeypatch, which):
    w, out = completed_workload(runner, prepare_inputs)
    if which in ("profile", "trace"):
        from pathlib import Path
        path = Path(w[which])
        path.write_text(path.read_text() + "\n")
    elif which == "algorithm":
        monkeypatch.setattr(runner, "ALGORITHM_VERSION", "different-version")
    elif which == "missing":
        (out / "comparison.png").unlink()
    else:
        (out / "completion.json").unlink()
    assert runner.cached_summary(w) is None


def test_worker_limit_honors_affinity_and_hard_cap(runner, monkeypatch):
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: set(range(3)))
    assert runner.worker_limit(8) == 3
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _: set(range(32)))
    assert runner.worker_limit(24) == 16
    assert runner.worker_limit(2) == 2


def test_weighted_admission_packs_small_jobs_without_exceeding_budget(runner, monkeypatch):
    import concurrent.futures

    batches = []
    active = []
    class Pool:
        def __init__(self, max_workers):
            assert max_workers == 3
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, w):
            future = Future()
            future.set_result(w["model"])
            active.append((future, w["estimated_worker_bytes"]))
            assert len(active) <= 3
            assert sum(size for _, size in active) <= 22
            return future
    def wait(futures, return_when):
        batches.append([size for _, size in active])
        done = {future for future, _ in active}
        active.clear()
        return done, set()
    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(runner, "WORKER_BUDGET_BYTES", 22)
    monkeypatch.setattr(concurrent.futures, "wait", wait)
    workloads = [dict(model=str(i), estimated_worker_bytes=size)
                 for i, size in enumerate([15, 10, 7])]
    results = []
    runner.run_pending(workloads, 3, results.append)
    assert batches == [[15, 7], [10]]
    assert set(results) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="exceeds"):
        runner.run_pending([dict(model="huge", estimated_worker_bytes=23)], 3, results.append)

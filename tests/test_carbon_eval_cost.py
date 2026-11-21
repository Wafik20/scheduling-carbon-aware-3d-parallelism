import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("cost_estimate", Path(__file__).parents[1] / "scripts/estimate_carbon_eval_cost.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_dense_memory_and_outer_start_counts():
    w = dict(model="tiny", tokens=100, token_quantum=10, deadline_h=4, strategies=3)
    fixed = mod.estimate(w, episodes=2, regions=3)
    outer = mod.estimate(w, episodes=2, regions=3, include_job_shift=True)
    assert fixed["states"] == 5 * 11 * 3
    assert fixed["arrays_gib"] * mod.GIB == 14 * 5 * 11 * 3
    assert fixed["scans_per_episode"] == 4 * 11 * 3
    assert outer["scans_per_episode"] == (4 + 4 + 3 + 2 + 1) * 11 * 3
    assert outer["total_scans"] == outer["scans_per_episode"] * 6
    assert outer["transition_scenarios_per_episode"]["1.0"] == outer["scans_per_episode"] * 3


def test_single_worker_can_still_exceed_budget():
    w = dict(model="huge", tokens=1e12, token_quantum=1, deadline_h=100, strategies=20)
    result = mod.estimate(w)
    assert result["workers"] == 1
    assert result["over_budget"]

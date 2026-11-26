"""Exact small-instance checks for the zero-overhead oracle.

Whole decision intervals produce integer token counts, so quantization does not
obscure the brute-force reference. Final intervals may be partial.
"""
from itertools import product
import math

import numpy as np
import pytest

from carbon_aware_3d.carbon_aware_env import (
    CarbonSim, SimConfig, rollout_oracle_dp, rollout_oracle_zero_overhead,
    rollout_static,
)


def strategy(rate, power, **kwargs):
    return dict(tp=1, pp=1, dp=8, micro_batch_size=1, placement_id="a",
                throughput_tok_s=rate / 3600.0, power_w=power,
                warmup_wall_s=0.0, warmup_iters=0, save_s=0.0,
                **kwargs)


def config(target=5):
    slow = strategy(1, 1000)
    fast = strategy(2, 2500)
    fast.update(micro_batch_size=2, placement_id="b")
    return SimConfig(strategies=[slow, fast], pretraining_tokens=target,
                     tokens_per_iter=1)


def brute_force(cfg, trace_t, trace_ci, *, max_hours, trace_offset_s=0,
                decision_interval_s=3600, wrap_trace=False):
    """Enumerate exact simulator schedules, independently of oracle internals."""
    steps = math.ceil(max_hours * 3600 / decision_interval_s)
    feasible = []
    for actions in product(range(len(cfg.strategies)), repeat=steps):
        sim = CarbonSim(cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
        sim.reset(trace_offset_s, initial_strategy_idx=actions[0])
        for action in actions:
            sim.step(action, decision_interval_s)
            if sim.done():
                break
        if sim.done() and sim.cursor_s <= max_hours * 3600 + 1e-6:
            feasible.append(sim)
    return min(feasible, key=lambda sim: sim.total_carbon_g) if feasible else None


@pytest.mark.parametrize("target,deadline,offset", [
    (5, 3, 0), (5, 2.5, 0), (3, 2.25, 1800), (4, 3.5, 900),
])
def test_matches_exact_brute_force_with_partial_final_step(target, deadline, offset):
    cfg = config(target)
    times = np.arange(8, dtype=float) * 3600
    ci = np.array([100, 900, 100, 500, 50, 600, 100, 300], dtype=float)
    kwargs = dict(max_hours=deadline, trace_offset_s=offset)
    expected = brute_force(cfg, times, ci, **kwargs)
    actual = rollout_oracle_zero_overhead(
        cfg, times, ci, token_quantum=1, **kwargs)
    assert expected is not None
    assert actual.done()
    assert actual.tokens_completed == pytest.approx(target)
    assert actual.cursor_s <= deadline * 3600 + 1e-6
    assert actual.total_carbon_g == pytest.approx(expected.total_carbon_g)
    if deadline == 2.5:
        assert actual.cursor_s == pytest.approx(2.5 * 3600)
        assert actual.history[-1].t_end_s - actual.history[-1].t_start_s == 1800


@pytest.mark.parametrize("distinguishing_field", ["placement_id", "micro_batch_size"])
def test_keeps_distinct_placement_and_microbatch_actions(distinguishing_field):
    cfg = config(5)
    other_field = ("micro_batch_size" if distinguishing_field == "placement_id"
                   else "placement_id")
    cfg.strategies[1][other_field] = cfg.strategies[0][other_field]
    times = np.arange(6, dtype=float) * 3600
    ci = np.array([100, 900, 100, 900, 100, 900], dtype=float)
    kwargs = dict(trace_offset_s=0, max_hours=3, token_quantum=1)
    actual = rollout_oracle_zero_overhead(cfg, times, ci, **kwargs)
    legacy = rollout_oracle_dp(cfg, times, ci, respect_deadline=True, **kwargs)
    expected = brute_force(cfg, times, ci, max_hours=3)
    assert [row.strategy_idx for row in actual.history] == [1, 0, 1]
    assert actual.reshard_count == 2
    assert actual.total_carbon_g == pytest.approx(expected.total_carbon_g)
    assert actual.total_carbon_g == pytest.approx(legacy.total_carbon_g)


@pytest.mark.parametrize("offset", [0, 5400])
def test_wrap_matches_tiled_trace_and_brute_force(offset):
    cfg = config(7)
    short_t = np.array([0., 3600.])
    short_ci = np.array([100., 900.])
    long_t = np.arange(12, dtype=float) * 3600
    long_ci = np.tile(short_ci, 6)
    kwargs = dict(trace_offset_s=offset, max_hours=5, token_quantum=1)
    wrapped = rollout_oracle_zero_overhead(
        cfg, short_t, short_ci, wrap_trace=True, **kwargs)
    tiled = rollout_oracle_zero_overhead(cfg, long_t, long_ci, **kwargs)
    expected = brute_force(cfg, short_t, short_ci, max_hours=5,
                           trace_offset_s=offset, wrap_trace=True)
    assert wrapped.done() and tiled.done()
    assert wrapped.total_carbon_g == pytest.approx(expected.total_carbon_g)
    assert wrapped.total_carbon_g == pytest.approx(tiled.total_carbon_g)
    assert wrapped.history == tiled.history


@pytest.mark.parametrize("field", [
    "container_startup_s", "fresh_build_s", "switch_time_s",
    "first_switch_cache_miss_s",
])
def test_rejects_nonzero_config_overhead(field):
    cfg = config()
    setattr(cfg, field, 1)
    with pytest.raises(ValueError):
        rollout_oracle_zero_overhead(cfg, np.array([0.]), np.array([100.]),
                                    trace_offset_s=0, max_hours=1)


@pytest.mark.parametrize("field", [
    "warmup_wall_s", "warmup_iters", "save_s", "reshard_wall_time_s",
])
def test_rejects_nonzero_strategy_overhead(field):
    cfg = config()
    cfg.strategies[1][field] = 1
    with pytest.raises(ValueError):
        rollout_oracle_zero_overhead(cfg, np.array([0.]), np.array([100.]),
                                    trace_offset_s=0, max_hours=1)


def test_coarse_quantum_retains_best_feasible_static():
    cfg = config(5)
    times = np.arange(8, dtype=float) * 3600
    ci = np.full(8, 100.)
    actual = rollout_oracle_zero_overhead(
        cfg, times, ci, trace_offset_s=0, max_hours=5,
        token_quantum=100)
    expected = rollout_static(cfg, times, ci, trace_offset_s=0, strategy_idx=0,
                              max_hours=5)
    assert actual.done()
    assert actual.cursor_s <= 5 * 3600
    assert actual.total_carbon_g == pytest.approx(expected.total_carbon_g)


@pytest.mark.parametrize("target", [5, 100])
def test_infeasible_deadline_does_not_look_feasible(target):
    cfg = config(target)
    times = np.arange(8, dtype=float) * 3600
    ci = np.full(8, 100.)
    assert brute_force(cfg, times, ci, max_hours=1) is None
    actual = rollout_oracle_zero_overhead(
        cfg, times, ci, trace_offset_s=0, max_hours=1,
        token_quantum=1)
    assert not actual.done() or actual.cursor_s > 3600 + 1e-6
    assert actual.tokens_completed > 0


def test_short_decisions_match_exact_brute_force():
    cfg = config(5)
    cfg.strategies[0]["throughput_tok_s"] *= 2
    cfg.strategies[1]["throughput_tok_s"] *= 2
    times = np.arange(4, dtype=float) * 3600
    ci = np.array([100., 900., 100., 900.])
    kwargs = dict(max_hours=1.5, trace_offset_s=1800,
                  decision_interval_s=1800)
    expected = brute_force(cfg, times, ci, **kwargs)
    actual = rollout_oracle_zero_overhead(
        cfg, times, ci, token_quantum=1, **kwargs)
    assert actual.done()
    assert actual.cursor_s <= 5400
    assert actual.total_carbon_g == pytest.approx(expected.total_carbon_g)


def test_nonwrapping_trace_allows_early_completion_beyond_horizon():
    cfg = config(1)
    actual = rollout_oracle_zero_overhead(
        cfg, np.array([0.]), np.array([100.]), trace_offset_s=0,
        max_hours=5, token_quantum=1)
    assert actual.done()
    assert actual.cursor_s <= 3600
    assert actual.total_carbon_g == pytest.approx(100.)


def test_short_nonwrapping_trace_returns_incomplete_progress():
    cfg = config(100)
    actual = rollout_oracle_zero_overhead(
        cfg, np.array([0.]), np.array([100.]), trace_offset_s=0,
        max_hours=5, token_quantum=1)
    assert not actual.done()
    assert 0 < actual.tokens_completed < cfg.pretraining_tokens
    assert actual.cursor_s <= 3600


def test_high_rate_target_just_beyond_deadline_capacity_is_incomplete():
    cfg = SimConfig(
        strategies=[strategy(3_600_000_000_000, 1000)],
        pretraining_tokens=3_600_000_000_001, tokens_per_iter=1,
    )
    actual = rollout_oracle_zero_overhead(
        cfg, np.array([0., 3600.]), np.array([100., 100.]),
        trace_offset_s=0, max_hours=1, token_quantum=1e12,
    )
    assert not actual.done()
    assert actual.tokens_completed < cfg.pretraining_tokens
    assert actual.cursor_s <= 3600
    assert actual.history

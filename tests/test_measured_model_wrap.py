"""Regression coverage for repeated measured-model carbon traces."""

import numpy as np
import pytest

from carbon_aware_3d.carbon_aware_env import CarbonSim, SimConfig, rollout_oracle_dp


def strategy(power_w=1000.0, throughput_tok_s=1.0):
    return dict(
        tp=1, pp=1, power_w=power_w, throughput_tok_s=throughput_tok_s,
        warmup_wall_s=0.0, warmup_iters=0, save_s=0.0,
    )


@pytest.mark.parametrize(
    "offset_s, duration_s, expected_carbon_g",
    [
        (0.0, 5 * 3600.0, 900.0),
        (0.0, 6 * 3600.0, 1200.0),
        (5400.0, 5 * 3600.0, 1000.0),
    ],
)
def test_carbon_sim_integrates_multiple_trace_cycles(
    offset_s, duration_s, expected_carbon_g,
):
    cfg = SimConfig(
        strategies=[strategy()], pretraining_tokens=int(duration_s),
        tokens_per_iter=1,
    )
    sim = CarbonSim(
        cfg, np.array([0.0, 3600.0]), np.array([100.0, 300.0]),
        wrap_trace=True,
    )
    sim.reset(offset_s)
    result = sim.step(0, decision_interval_s=duration_s)

    assert result["terminated"]
    assert sim.cursor_s == duration_s
    assert sim.tokens_completed == duration_s
    assert sim.total_energy_j == pytest.approx(1000.0 * duration_s)
    assert sim.total_carbon_g == pytest.approx(expected_carbon_g)
    for boundary_s in (7200.0, 14400.0, 21600.0):
        assert sim.ci_at(boundary_s - 1e-6) == 300.0
        assert sim.ci_at(boundary_s) == 100.0
        assert sim.ci_at(boundary_s + 1e-6) == 100.0


@pytest.mark.parametrize("offset_s", [0.0, 5400.0])
def test_oracle_wrap_matches_explicitly_tiled_trace(offset_s):
    cfg = SimConfig(
        strategies=[strategy(1000.0, 1.0), strategy(2500.0, 2.0)],
        pretraining_tokens=8 * 3600, tokens_per_iter=1,
    )
    kwargs = dict(
        trace_offset_s=offset_s, decision_interval_s=3600.0,
        max_hours=6.0, token_quantum=3600.0, respect_deadline=True,
    )
    wrapped = rollout_oracle_dp(
        cfg, np.array([0.0, 3600.0]), np.array([100.0, 300.0]),
        wrap_trace=True, **kwargs,
    )
    tiled = rollout_oracle_dp(
        cfg, np.arange(12, dtype=float) * 3600.0,
        np.tile([100.0, 300.0], 6), wrap_trace=False, **kwargs,
    )

    assert wrapped.done() and tiled.done()
    assert 4 * 3600.0 < wrapped.cursor_s <= 6 * 3600.0
    assert wrapped.tokens_completed == tiled.tokens_completed == cfg.pretraining_tokens
    assert wrapped.cursor_s == pytest.approx(tiled.cursor_s)
    assert wrapped.total_carbon_g == pytest.approx(tiled.total_carbon_g)
    assert wrapped.total_energy_j == pytest.approx(tiled.total_energy_j)
    assert wrapped.reshard_count == tiled.reshard_count
    wrapped_indices = [row.strategy_idx for row in wrapped.history]
    tiled_indices = [row.strategy_idx for row in tiled.history]
    assert wrapped_indices == tiled_indices
    assert set(wrapped_indices) == {0, 1}
    assert wrapped.history == tiled.history


@pytest.fixture
def runner():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_measured_models.py"
    spec = importlib.util.spec_from_file_location("measured_model_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measured_menu_keeps_all_unique_actions(runner):
    sweep = runner.ROOT / "runs/placement_sweep_8gpu"
    if not (sweep / "results.csv").exists():
        pytest.skip("Measured sweep is not included in this checkout")
    frame = runner.validate_menu(sweep)
    assert len(frame) == 348
    assert len(frame[frame.model == "gpt2"]) == 38
    assert not frame.duplicated(["model", *runner.TUPLE]).any()


def test_duplicate_menu_tuple_is_rejected(runner, tmp_path):
    import pandas as pd

    models = sorted(runner.MODELS)
    rows = [dict(model=models[i % 9], tp=1, pp=1, dp=8,
                 micro_batch_size=1, placement_id=str(i)) for i in range(348)]
    rows[-1] = rows[0].copy()
    pd.DataFrame(rows).to_csv(tmp_path / "results.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.validate_menu(tmp_path)


@pytest.fixture
def prepare_inputs(runner, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pandas as pd

    rows = []
    for i in range(2):
        row = dict(strategy(1000.0 + i * 500, 1.0 + i), model="gpt2",
                   id=f"case{i}", attempt=1, dp=8, micro_batch_size=1,
                   placement_id=f"placement{i}", global_batch_size=8,
                   seq_length=16)
        rows.append(row)
        attempt = tmp_path / "cases" / row["id"] / "attempts/0001"
        attempt.mkdir(parents=True)
        (attempt / "launcher.log").write_text("number of parameters on rank 0: 1000\n")
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(runner, "MODELS", {"gpt2"})
    monkeypatch.setattr(runner, "validate_menu", lambda _: frame)
    # Profile parsing has its own tests; preserve synthetic source row order here.
    monkeypatch.setattr(runner, "load_strategy_profile",
                        lambda path: pd.read_csv(path).to_dict("records"))
    trace = tmp_path / "trace.csv"
    times = pd.date_range("2024-05-01", periods=744, freq="h")
    pd.DataFrame(dict(year=times.year, month=times.month, day=times.day,
                      time=times.strftime("%H:%M"), carbon=100)).to_csv(trace, index=False)
    args = SimpleNamespace(sweep=tmp_path, trace=trace, out=tmp_path / "out",
                           models=["gpt2"], tokens=None, deadline_hours=None,
                           oracle_token_quantum=None)
    return args, frame


def test_prepare_defaults_and_overrides(runner, prepare_inputs):
    import pandas as pd

    args, _ = prepare_inputs
    default, = runner.prepare(args)
    assert default["tokens"] == 20 * 1000
    assert default["deadline_h"] == 4
    assert default["global_batch_size"] * default["seq_length"] == 128
    args.tokens, args.deadline_hours, args.oracle_token_quantum = 12345, 2.5, 100
    override, = runner.prepare(args)
    assert override["tokens"] == 12345
    assert override["deadline_h"] == 2.5
    assert override["token_quantum"] == 100
    assert override["offset_h"] == 0 and override["wrap_trace"]
    mapping = pd.read_csv(args.out / "gpt2/action_mapping.csv")
    assert mapping.strategy_idx.tolist() == [0, 1]
    assert mapping.placement_id.tolist() == ["placement0", "placement1"]


def test_prepare_rejects_mixed_fixed_workloads(runner, prepare_inputs):
    args, frame = prepare_inputs
    frame.loc[1, "seq_length"] = 32
    with pytest.raises(ValueError, match="Inconsistent batches"):
        runner.prepare(args)


@pytest.mark.parametrize("hours, wraps", [(2, 0), (2.5, 1), (4, 1), (5, 2)])
def test_infeasible_summary_retained_and_wrap_boundaries(
    runner, prepare_inputs, monkeypatch, hours, wraps,
):
    import json
    import pandas as pd

    args, _ = prepare_inputs
    args.tokens, args.deadline_hours = 100000, hours
    workload, = runner.prepare(args)
    monkeypatch.setattr(runner, "load_carbon_trace", lambda _: (
        np.array([0.0, 3600.0]), np.array([100.0, 300.0]), None,
    ))

    def incomplete(cfg, trace_t, trace_ci, *pos, **kw):
        sim = CarbonSim(cfg, trace_t, trace_ci, wrap_trace=True)
        sim.step(0, decision_interval_s=hours * 3600)
        return sim

    monkeypatch.setattr(runner, "rollout_static", incomplete)
    monkeypatch.setattr(runner, "rollout_oracle_zero_overhead", incomplete)
    summary = runner.run_model(workload)
    assert not summary["feasible"] and not summary["completed"]
    assert summary["feasible_static_count"] == 0
    assert summary["best_fixed_idx"] is None
    assert summary["savings_pct"] is None
    assert summary["reason_if_infeasible"]
    assert summary["trace_wraps"] == wraps
    saved = json.loads((args.out / "gpt2/summary.json").read_text())
    assert saved == summary
    records = pd.read_csv(args.out / "gpt2/results.csv")
    assert len(records) == 3
    assert not records.feasible.any()
    assert records.trace_wraps.tolist() == [wraps] * 3


def test_prepare_coarsens_automatic_quantum_without_dropping_actions(
    runner, prepare_inputs, monkeypatch,
):
    args, _ = prepare_inputs
    args.tokens = 10_000_000
    monkeypatch.setattr(runner, "estimate_memory_bytes", lambda tokens, quantum, deadline, actions:
                        23 * 1024**3 if quantum < 3_000_000 else 2 * 1024**3)
    workload, = runner.prepare(args)
    assert workload["token_quantum"] == 3_000_000
    assert workload["action_count"] == 2
    assert workload["estimated_worker_bytes"] == 2 * 1024**3
    args.oracle_token_quantum = 100
    explicit, = runner.prepare(args)
    assert explicit["token_quantum"] == 100


def test_zero_savings_summary_is_retained(runner, prepare_inputs):
    args, frame = prepare_inputs
    frame["power_w"] = 1000.0
    frame["throughput_tok_s"] = 1.0
    args.tokens, args.deadline_hours, args.oracle_token_quantum = 3600, 2, 3600
    workload, = runner.prepare(args)
    summary = runner.run_model(workload)
    assert summary["feasible"] and summary["completed"]
    assert summary["feasible_static_count"] == 2
    assert summary["best_fixed_idx"] == 0
    assert summary["savings_pct"] == 0.0
    assert summary["reason_if_infeasible"] == ""


def test_prepare_rejects_minimum_bin_over_memory_budget(runner, prepare_inputs, monkeypatch):
    args, _ = prepare_inputs
    monkeypatch.setattr(runner, "estimate_memory_bytes", lambda *args: 23 * 1024**3)
    with pytest.raises(ValueError, match="Even one token bin"):
        runner.prepare(args)


def test_default_trace_is_recovered_workshop_west1(runner):
    import hashlib
    if not runner.DEFAULT_TRACE.exists():
        pytest.skip("Recovered research trace is not installed")
    _, carbon, _ = runner.load_carbon_trace(runner.DEFAULT_TRACE)
    assert len(carbon) == 744
    assert carbon.mean() == pytest.approx(142.7284946236559)
    assert hashlib.sha256(runner.DEFAULT_TRACE.read_bytes()).hexdigest() == (
        "8fb0da0d267112b666d726298746f4391804ca619b3924a9a1dec279666db96e"
    )

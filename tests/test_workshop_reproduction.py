import importlib.util
from pathlib import Path
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("workshop", Path(__file__).resolve().parents[1] / "experiments/workshop_reproduction.py")
workshop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workshop)


def test_warmup_tokens_count():
    row = dict(warmup_iters=2, warmup_wall_s=10, throughput_tok_s=100)
    assert workshop.static_seconds(row, 1000, 100) == 18
    assert workshop.static_seconds(row, 100, 100) == 5


def test_rounding_tolerance_and_invalid_metric():
    row = dict(tp=1, pp=1, dp=8, global_batch_size=16, seq_length=1024,
               tokens_per_iter=16384, power_w=706.7, step_time_s=.2677,
               throughput_tok_s=61205.97, joules_per_token=.011547,
               warmup_wall_s=30.15, warmup_iters=66, save_s=0)
    workshop.validate_row(row)
    with pytest.raises(ValueError, match="Power/energy"):
        workshop.validate_row(dict(row, joules_per_token=.02))
    with pytest.raises(ValueError, match="Invalid"):
        workshop.validate_row(dict(row, power_w=float("nan")))


def test_pairing_uses_same_best_carbon_row_and_drops_infeasible():
    rows = [dict(episode=0, offset_h=1, policy=p, carbon_g=c, energy_kj=e,
                 completion_delay_h=t, completed=True, feasible=f,
                 episode_best_feasible_static_policy="static(a)")
            for p, c, e, t, f in [("static(a)", 10, 100, 8, True),
                                  ("static(b)", 20, 50, 6, True),
                                  ("Greedy", 8, 80, 7, True),
                                  ("bad", 1, 1, 1, False)]]
    paired = workshop.paired_rows(pd.DataFrame(rows))
    assert paired.policy.tolist() == ["Greedy"]
    assert paired.baseline_energy_kj.iloc[0] == 100
    assert paired.energy_kj_savings_pct.iloc[0] == 20
    assert paired.completion_delay_h_savings_pct.iloc[0] == 12.5


def test_duplicate_regional_series_blocked(tmp_path):
    dates = pd.date_range("2024-01-01", periods=8784, freq="h")
    frame = pd.DataFrame(dict(year=dates.year, month=dates.month, day=dates.day,
                              time=dates.strftime("%H:%M:%S"), carbon=100))
    for region in workshop.REGIONS:
        frame.to_csv(tmp_path / f"carbon_2024_hourly_aws_{region}.csv", index=False)
    traces, warnings = workshop.audit_traces(tmp_path)
    assert len(traces) == 3
    assert len(warnings) == 3
    assert all("BLOCKED" in warning for warning in warnings)


def test_monthly_trace_validation(tmp_path):
    dates = pd.date_range("2024-05-01", periods=744, freq="h")
    frame = pd.DataFrame(dict(year=dates.year, month=dates.month, day=dates.day,
                              time=dates.strftime("%H:%M:%S"), carbon=100))
    for i, region in enumerate(workshop.REGIONS):
        frame["carbon"] = 100 + i
        frame.to_csv(tmp_path / f"carbon_2024_05_hourly_aws_{region}.csv", index=False)
    traces, warnings = workshop.audit_traces(tmp_path, "workshop-fixed-start")
    assert not warnings
    assert {t["hours"] for t in traces.values()} == {744}
    frame.iloc[:-1].to_csv(tmp_path / f"carbon_2024_05_hourly_aws_{region}.csv", index=False)
    with pytest.raises(ValueError, match="Incomplete"):
        workshop.audit_traces(tmp_path, "workshop-fixed-start")


def test_fixed_grid_preserved_and_commands_distinct(tmp_path):
    base = [dict(model=model, global_batch_size=16, seq_length=1024,
                 parameters=100, strategies=2, profile="profile.csv")
            for model in ("gpt2", "llama_1b")]
    selected = [dict(model=w["model"], warmup_iters=0, warmup_wall_s=0,
                     throughput_tok_s=100_000) for w in base]
    workloads = workshop.fixed_start_workloads(base, selected)
    assert [(w["model"], w["tokens"], w["deadline_h"], w["token_quantum"])
            for w in workloads] == list(workshop.WORKSHOP_GRID)
    assert all(w["feasible_static_strategies"] == 1 for w in workloads)
    selected[0]["throughput_tok_s"] = 1
    infeasible = workshop.fixed_start_workloads(base, selected)
    assert infeasible[0]["feasible_static_strategies"] == 0
    assert infeasible[0]["deadline_h"] == 12
    workshop.write_json(tmp_path / "workloads.json", workloads)
    workshop.write_json(tmp_path / "audit.json", dict(study="workshop-fixed-start",
                         traces={r: dict(path=r) for r in workshop.REGIONS}, max_offset_h=575))
    commands = list(workshop.commands(tmp_path, tmp_path / "results"))
    assert len(commands) == len({dest for dest, cmd in commands}) == 18
    for dest, cmd in commands:
        assert "--include-job-shift" not in cmd
        assert cmd[cmd.index("--n-episodes") + 1] == "32"
        assert cmd[cmd.index("--seed") + 1] == "0"
        assert cmd[cmd.index("--max-offset-h") + 1] == "575"

import numpy as np
import pytest

from carbon_aware_3d.carbon_aware_env import (
    CarbonSim, SimConfig, TraceOverrunError, load_carbon_trace,
)


def config():
    return SimConfig(
        strategies=[dict(tp=1, pp=1, power_w=1000.0,
                         throughput_tok_s=1.0, warmup_wall_s=0.0,
                         warmup_iters=0, save_s=0.0)],
        pretraining_tokens=7200, tokens_per_iter=1,
    )


def test_carbon_integration_across_hour_boundary():
    sim = CarbonSim(config(), np.array([0., 3600.]), np.array([100., 300.]))
    result = sim.step(0, decision_interval_s=7200)
    assert result['terminated']
    assert sim.tokens_completed == 7200
    assert sim.total_energy_j == 7_200_000
    assert sim.total_carbon_g == pytest.approx(400)


def test_trace_end_is_explicit_unless_wrapping():
    t, ci = np.array([0.]), np.array([100.])
    with pytest.raises(TraceOverrunError):
        CarbonSim(config(), t, ci).ci_at(3600)
    assert CarbonSim(config(), t, ci, wrap_trace=True).ci_at(3600) == 100


def test_load_hourly_trace(tmp_path):
    trace = tmp_path / 'trace.csv'
    trace.write_text('year,month,day,time,carbon\n2024,1,1,01:00,300\n2024,1,1,00:00,100\n')
    t, ci, _ = load_carbon_trace(trace)
    assert t.tolist() == [0, 3600]
    assert ci.tolist() == [100, 300]

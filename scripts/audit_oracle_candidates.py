"""Diagnostic snapshot of the zero-overhead oracle, not a production scheduler.

Records the retained rounded-DP winner before exact static comparison.
Run from the repository root; writes only runs/measured_models_audit/.
"""
import sys, json, time, math, argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from carbon_aware_3d.carbon_aware_env import *
from scripts.evaluate_measured_models import default_config,load_strategy_profile,load_carbon_trace
from typing import Sequence
def rollout_oracle_zero_overhead(
    sim_cfg: SimConfig,
    trace_t: np.ndarray,
    trace_ci: np.ndarray,
    trace_offset_s: float,
    decision_interval_s: float = 3600.0,
    max_hours: float = 200.0,
    token_quantum: float = 5_000_000.0,
    wrap_trace: bool = False,
) -> CarbonSim:
    """Fixed-start, deadline-constrained oracle for zero-overhead profiles.

    Rolling token-bin costs remove the previous-action dimension: every action
    remains available at every decision, including distinct placements. Work is
    rounded down after each full decision. Thus this is exact on the aligned
    token lattice, not a continuous-work optimum for arbitrary quanta. Unlike
    the legacy oracle it performs no replay-based local repair. Exact feasible
    static schedules are also compared, and the winner is replayed in CarbonSim.

    Trace samples must follow the simulator's regular hourly trace convention.
    Runtime is O(T Q K); parent storage is O(T Q). All startup, switch, warmup,
    and save durations must be zero; idle power is irrelevant without idle time.
    The final decision can be shorter than decision_interval_s to respect the
    deadline. Infeasible workloads return the fastest incomplete replay.
    """
    for name, value in (
        ('decision_interval_s', decision_interval_s),
        ('max_hours', max_hours), ('token_quantum', token_quantum),
        ('pretraining_tokens', sim_cfg.pretraining_tokens),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('container_startup_s', 'fresh_build_s',
                 'switch_time_s', 'first_switch_cache_miss_s'):
        if getattr(sim_cfg, name) != 0:
            raise ValueError(f'zero-overhead oracle requires {name}=0')
    if not sim_cfg.strategies:
        raise ValueError('zero-overhead oracle requires at least one strategy')
    for strategy in sim_cfg.strategies:
        for name in ('warmup_wall_s', 'warmup_iters', 'save_s', 'reshard_wall_time_s'):
            if strategy.get(name, 0) != 0:
                raise ValueError(f'zero-overhead oracle requires {name}=0')
        if (not math.isfinite(strategy['throughput_tok_s'])
                or strategy['throughput_tok_s'] <= 0
                or not math.isfinite(strategy['power_w'])
                or strategy['power_w'] < 0):
            raise ValueError('strategies require positive throughput and nonnegative power')

    probe = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    probe.reset(trace_offset_s)
    deadline = float(max_hours * 3600.0)
    # There is no useful work beyond the available trace in non-wrapping mode.
    horizon = deadline if wrap_trace else min(
        deadline, probe.trace_duration_s - trace_offset_s)
    rates = np.array([s['throughput_tok_s'] for s in sim_cfg.strategies], dtype=float)
    powers = np.array([s['power_w'] for s in sim_cfg.strategies], dtype=float)
    target = float(sim_cfg.pretraining_tokens)
    if horizon <= 0 or target > rates.max() * horizon + 1e-6:
        fastest_idx = int(np.argmax(rates))
        probe.reset(trace_offset_s, initial_strategy_idx=fastest_idx)
        for t in range(max(0, int(math.ceil(horizon / decision_interval_s)))):
            probe.step(fastest_idx, min(decision_interval_s, horizon - probe.cursor_s))
        return probe

    # Integral of CI at arbitrary local times, including offset and wrap edges.
    first_boundary = probe._next_hour_boundary_local(0.0)
    boundaries = np.concatenate(([0.0], np.arange(first_boundary, horizon, 3600.0), [horizon]))
    intensities = np.array([
        probe.ci_at(trace_offset_s + (a + b) / 2.0)
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ])
    integrated_ci = np.concatenate(([0.0], np.cumsum(np.diff(boundaries) * intensities)))

    def ci_integral(times):
        return np.interp(times, boundaries, integrated_ci)

    static_durations = target / rates
    static_costs = np.where(
        target <= rates * horizon + 1e-6,
        powers * ci_integral(np.minimum(static_durations, horizon)) / 3.6e6,
        np.inf,
    )
    static_idx = int(np.argmin(static_costs))
    n_steps = int(math.ceil(horizon / decision_interval_s))
    n_bins = int(math.ceil(target / token_quantum))
    remaining = target - np.arange(n_bins, dtype=float) * token_quantum
    costs = np.full(n_bins, np.inf)
    costs[0] = 0.0
    parent_dtype = np.int16 if len(rates) <= np.iinfo(np.int16).max else np.int32
    parents = np.full((n_steps, n_bins), -1, dtype=parent_dtype)
    increments = np.empty((n_steps, len(rates)), dtype=np.int64)
    best_cost = np.inf
    terminal = None

    for t in range(n_steps):
        start = t * decision_interval_s
        duration = min(decision_interval_s, horizon - start)
        full_ci = ci_integral(start + duration) - ci_integral(start)
        next_costs = np.full(n_bins, np.inf)
        for k, (rate, power) in enumerate(zip(rates, powers)):
            produced = rate * duration
            increment = int(math.floor(produced / token_quantum))
            increments[t, k] = increment
            # Terminal work is charged at its exact duration, not a full step.
            terminal_start = int(np.searchsorted(-remaining, -produced - 1e-6))
            if terminal_start < n_bins:
                finish_times = start + remaining[terminal_start:] / rate
                candidates = costs[terminal_start:] + power * (
                    ci_integral(np.minimum(finish_times, horizon)) - ci_integral(start)
                ) / 3.6e6
                local_q = int(np.argmin(candidates))
                if candidates[local_q] < best_cost:
                    best_cost = float(candidates[local_q])
                    terminal = (t, terminal_start + local_q, k)
            # Each action shifts a contiguous vector of nonterminal bins.
            count = min(terminal_start, n_bins - increment)
            if count > 0:
                candidate = costs[:count] + power * full_ci / 3.6e6
                destination = next_costs[increment:increment + count]
                improved = candidate < destination
                destination[improved] = candidate[improved]
                parents[t, increment:increment + count][improved] = k
        costs = next_costs
        if not np.any(np.isfinite(costs)):
            break

    actions = None
    if terminal is not None:
        t, q, k = terminal
        actions = [k]
        for previous_t in range(t - 1, -1, -1):
            k = int(parents[previous_t, q])
            actions.append(k)
            q -= int(increments[previous_t, k])
        actions.reverse()

    def replay(sequence: Sequence[int]) -> CarbonSim:
        sim = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
        sim.reset(trace_offset_s, initial_strategy_idx=int(sequence[0]))
        for k in sequence:
            budget = min(decision_interval_s, horizon - sim.cursor_s)
            if budget <= 0 or sim.done():
                break
            sim.step(int(k), budget)
        return sim

    best_sim = replay([static_idx] * n_steps)
    candidate_sim = replay(actions) if actions is not None else None
    static_sim = best_sim
    accepted = candidate_sim is not None and candidate_sim.done() and candidate_sim.cursor_s <= deadline + 1e-6 and (not static_sim.done() or candidate_sim.total_carbon_g < static_sim.total_carbon_g)
    def metrics(sim):
        if sim is None: return None
        return dict(carbon_g=sim.total_carbon_g, hours=sim.cursor_s/3600, tokens=sim.tokens_completed, done=sim.done(), switches=sim.reshard_count)
    return dict(dp_cost_g=best_cost, terminal=terminal, actions=actions, candidate=metrics(candidate_sim), static=metrics(static_sim), static_idx=static_idx, accepted=accepted, reason='lower exact replay carbon' if accepted else ('no terminal' if candidate_sim is None else 'incomplete' if not candidate_sim.done() else 'deadline exceeded' if candidate_sim.cursor_s>deadline+1e-6 else 'exact replay carbon not strictly lower than static'), static_infeasible_action_indices=np.flatnonzero(static_durations>horizon+1e-6).tolist(), floor_loss_per_hour=(rates*decision_interval_s % token_quantum).tolist(), parent_bytes=parents.nbytes)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--factor',type=int,default=1); ap.add_argument('--trace',choices=['old','recovered'],default='old'); args=ap.parse_args()
    rows=[]
    ws=json.loads((ROOT/'runs/measured_models_fast/workloads.json').read_text())['workloads']
    for w in ws:
        cfg=default_config(pretraining_tokens=w['tokens'],global_batch_size=w['global_batch_size'],seq_length=w['seq_length'],n_gpus=8)
        cfg.strategies=load_strategy_profile(w['profile'])
        cfg.container_startup_s=cfg.fresh_build_s=cfg.switch_time_s=cfg.first_switch_cache_miss_s=0.
        path=w['trace'] if args.trace=='old' else str(ROOT/'data/recovered_workshop_may2024/carbon_2024_05_hourly_aws_us-west-1.csv')
        tt,ci,_=load_carbon_trace(path)
        start=time.monotonic()
        r=rollout_oracle_zero_overhead(cfg,tt,ci,0,max_hours=w['deadline_h'],token_quantum=w['token_quantum']/args.factor,wrap_trace=True)
        r.update(model=w['model'],trace=args.trace,quantum=w['token_quantum']/args.factor,elapsed_s=time.monotonic()-start)
        rows.append(r)
        out=ROOT/f'runs/measured_models_audit/candidates_{args.trace}_qdiv{args.factor}.json'; out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(rows,indent=2))
        print(w['model'],r['candidate'],r['static'],r['reason'],r['elapsed_s'],flush=True)

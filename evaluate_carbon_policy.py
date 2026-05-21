"""Evaluate carbon-aware scheduling baselines.

Runs N episodes for each policy on identical trace-offset draws so the
comparison is paired. Produces:
  - aggregate.png: box/strip plot of total carbon per policy across N episodes
  - rollout_offset_<h>h.png: per-episode CI / strategy / cumulative-carbon plot
  - results.csv: raw per-episode numbers

Usage:
    python evaluate_carbon_policy.py \
        --trace /path/to/carbon_week.csv \
        --tokens 3000000000 \
        --n-episodes 32 \
        --out runs/static_smart_oracle/eval
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carbon_aware_3d.carbon_aware_env import (
    default_config, load_carbon_trace, make_env,
    rollout_static, rollout_job_shifting_static_fast,
    rollout_smart_switch, rollout_oracle_dp, rollout_oracle_shift_switch,
    plot_comparison, CarbonSim, _fastest_strategy_idx, _replay_action_sequence,
)


GREEDY_LABEL = 'Greedy'
BEST_CASE_SWITCHING_LABEL = 'Fixed Start Time Dynamic Parallelism'
JOB_SHIFT_LABEL = 'Job Shifting'
ORACLE_SHIFT_SWITCH_LABEL = 'Job Shifting + Dynamic Parallelism'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True)
    p.add_argument('--include-rl', action='store_true',
                   help='Evaluate saved RL policy as an additional baseline.')
    p.add_argument('--model', default=None, help='Path to SB3 model .zip')
    p.add_argument('--algo', choices=['ppo', 'dqn'], default=None)
    p.add_argument('--tokens', type=int, default=3_000_000_000)
    p.add_argument('--global-batch-size', type=int, default=16)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--model-name', default=None,
                   help='Model label for feasibility summaries')
    p.add_argument('--model-num-params', type=int, default=None,
                   help='Model parameter count for memory feasibility filtering')
    p.add_argument('--filter-strategies', action='store_true',
                   help='Filter TP/PP/DP strategies by model memory feasibility')
    p.add_argument('--gpu-memory-gb', type=float, default=48.0,
                   help='Physical memory per GPU for feasibility filtering')
    p.add_argument('--usable-memory-frac', type=float, default=0.90,
                   help='Usable fraction of GPU memory for feasibility filtering')
    p.add_argument('--bytes-per-param-training', type=float, default=18.0,
                   help='Training-state bytes per parameter before TP/PP sharding')
    p.add_argument('--activation-reserve-gb', type=float, default=4.0,
                   help='Extra per-GPU activation/workspace memory reserve')
    p.add_argument('--min-pp', type=int, default=1,
                   help='Minimum PP degree allowed by feasibility filtering')
    p.add_argument('--quiet-feasibility', action='store_true',
                   help='Suppress feasibility summary printing')
    p.add_argument('--forecast-horizon-h', type=int, default=6)
    p.add_argument('--max-episode-hours', type=float, default=72.0)
    p.add_argument('--n-episodes', type=int, default=32,
                   help='Number of trace-offset draws to evaluate on')
    p.add_argument('--smart-fast-idx', type=int, default=-1,
                   help='Fast/catch-up strategy index for Greedy baseline. '
                        'Use -1 to automatically pick the highest-throughput '
                        'strategy.')
    p.add_argument('--smart-lookahead-h', type=int, default=6,
                   help='Lookahead window for Greedy clean catch-up CI')
    p.add_argument('--smart-margin-g', type=float, default=0.0,
                   help='Extra required carbon benefit before Greedy '
                        'changes strategy')
    p.add_argument('--smart-candidates', type=str, default=None,
                   help='Comma-separated strategy indices for Greedy '
                        'dirty-window candidates, e.g. "1,2". Default None '
                        'means auto-select lower-power/slower candidates.')
    p.add_argument('--include-job-shift', action='store_true',
                   help='Evaluate temporal job-shifting baseline. When dynamic '
                        'scheduling is applicable, also evaluate '
                        'job shifting + dynamic parallelism.')
    p.add_argument('--shift-window-h', type=float, default=24.0,
                   help='Maximum allowed delay before starting the job.')
    p.add_argument('--shift-step-h', type=float, default=1.0,
                   help='Grid step for searching possible delayed start times.')
    p.add_argument('--shift-strategy',
                   choices=['fastest'],
                   default='fastest',
                   help='Strategy used after shifting. For now only fastest throughput.')
    p.add_argument('--oracle-token-quantum', type=float, default=5_000_000.0,
                   help='Token discretization for Fixed Start-Time dynamic '
                        'parallelism baseline')
    p.add_argument('--oracle-switch-cost-mode',
                   choices=['sim', 'fixed'],
                   default='sim',
                   help='Switch-cost model for Fixed Start-Time dynamic '
                        'parallelism baseline')
    p.add_argument('--debug-oracle', action='store_true',
                   help='Print Fixed Start-Time dynamic parallelism internals '
                        'and assert if it loses to completed static/smart schedules')
    p.add_argument('--debug-shift-switch', action='store_true',
                   help='Dump candidate-start diagnostics for job shifting + '
                        'dynamic parallelism and Job Shifting.')
    p.add_argument('--debug-shift-switch-episode', type=int, default=None,
                   help='1-based episode number for --debug-shift-switch. '
                        'If omitted, dumps every episode.')
    p.add_argument('--fixed-offset-h', type=float, default=None,
                   help='Evaluate every episode at this exact trace offset '
                        '(hours), useful for reproducing Fixed Start-Time '
                        'dynamic parallelism bugs')
    p.add_argument('--n-detail-rollouts', type=int, default=3,
                   help='How many per-episode detail plots to save')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wrap-trace', action='store_true',
                   help='Tile the trace cyclically during evaluation. Off by '
                        'default so eval episodes draw from the real trace only.')
    p.add_argument('--out', default='eval_out')
    return p.parse_args()


def run_rl_episode(model, env, deterministic: bool = True) -> CarbonSim:
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return env.sim


def detect_model_algo(model_path: str | Path) -> str | None:
    """Infer SB3 algorithm family from the saved policy class metadata."""
    path = Path(model_path)
    if not path.exists() and path.suffix != '.zip':
        path = path.with_suffix(path.suffix + '.zip')
    if not path.exists():
        return None

    try:
        with zipfile.ZipFile(path) as zf:
            data = json.loads(zf.read('data'))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return None

    policy_class = data.get('policy_class', {})
    module = ''
    if isinstance(policy_class, dict):
        module = str(policy_class.get('__module__', ''))
    if '.dqn.' in module:
        return 'dqn'
    if 'common.policies' in module:
        return 'ppo'
    return None


def main():
    args = parse_args()
    if args.include_rl:
        if args.model is None:
            raise ValueError('--model is required when --include-rl is set')
        if args.algo is None:
            raise ValueError('--algo is required when --include-rl is set')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_t, trace_ci, _ = load_carbon_trace(args.trace)
    cfg = default_config(
        pretraining_tokens=args.tokens,
        global_batch_size=args.global_batch_size,
        seq_length=args.seq_length,
        model_num_params=args.model_num_params,
        model_name=args.model_name,
        filter_strategies=args.filter_strategies,
        gpu_memory_gb=args.gpu_memory_gb,
        usable_memory_frac=args.usable_memory_frac,
        bytes_per_param_training=args.bytes_per_param_training,
        activation_reserve_gb=args.activation_reserve_gb,
        min_pp=args.min_pp,
        verbose_feasibility=not args.quiet_feasibility,
    )
    dynamic_scheduling_applicable = len(cfg.strategies) >= 2
    if not dynamic_scheduling_applicable:
        model_label = args.model_name or (
            f'{args.model_num_params / 1e9:.0f}B'
            if args.model_num_params is not None else 'model'
        )
        print(
            f"[skip] {model_label}: only {len(cfg.strategies)} feasible strategy. "
            "Skipping Fixed Start-Time dynamic parallelism/Greedy savings comparison."
        )
        print(
            "Only one feasible strategy remained for this model size on the "
            "target hardware, so dynamic scheduling is not applicable. We "
            "report this as a feasibility-boundary case rather than an "
            "Fixed Start-Time dynamic parallelism/Greedy win."
        )
    include_rl_eval = args.include_rl and dynamic_scheduling_applicable
    if args.include_rl and not include_rl_eval:
        print('[skip] RL baseline disabled for static-only feasible case.')
    if args.smart_candidates is None or args.smart_candidates.strip() == '':
        smart_candidates = None
    else:
        requested_smart_candidates = [
            int(x) for x in args.smart_candidates.split(',')
            if x.strip()
        ]
        original_to_filtered_idx = {
            int(s.get('original_strategy_idx', i)): i
            for i, s in enumerate(cfg.strategies)
        }
        smart_candidates = []
        removed_candidates = []
        for idx in requested_smart_candidates:
            if idx in original_to_filtered_idx:
                smart_candidates.append(original_to_filtered_idx[idx])
            else:
                removed_candidates.append(idx)
        if removed_candidates:
            print(
                '[warning] Smart-switch candidates removed by feasibility '
                f'filter or unavailable: {removed_candidates}'
            )
        if not smart_candidates:
            print(
                '[warning] No requested Greedy candidates remain; '
                'using auto-selected feasible lower-power candidates.'
            )
            smart_candidates = None

    model = None
    env = None
    algo = args.algo
    if include_rl_eval:
        # Load model only when requested, so the default static/smart/oracle
        # evaluation path has no Stable-Baselines dependency.
        detected_algo = detect_model_algo(args.model)
        if detected_algo is not None and detected_algo != algo:
            print(f'Model metadata says {detected_algo.upper()}, '
                  f'overriding --algo {algo}.')
            algo = detected_algo

        if algo == 'ppo':
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
        else:
            from stable_baselines3 import DQN
            model = DQN.load(args.model)

        # Build a single env for evaluation (we control offsets manually).
        env = make_env(
            cfg, trace_t, trace_ci,
            forecast_horizon_h=args.forecast_horizon_h,
            max_episode_hours=args.max_episode_hours,
            random_start=True,
            wrap_trace=args.wrap_trace,
            seed=args.seed,
        )

    # Pre-draw N trace offsets so every policy sees the same conditions
    rng = np.random.default_rng(args.seed)
    trace_dur = trace_t[-1] + 3600.0
    baseline_max_hours = args.max_episode_hours * 1.5
    shift_slack_s = args.shift_window_h * 3600.0 if args.include_job_shift else 0.0
    slack = (
        baseline_max_hours * 3600.0
        + shift_slack_s
        + 3600.0
        + max(args.forecast_horizon_h, args.smart_lookahead_h) * 3600.0
    )
    max_offset = max(0.0, trace_dur - slack)
    if args.fixed_offset_h is None:
        offsets = rng.uniform(0.0, max_offset, size=args.n_episodes)
    else:
        offsets = np.full(args.n_episodes, args.fixed_offset_h * 3600.0)

    records = []
    detail_rollouts = []  # list of (offset, {label: CarbonSim}, {label: wait_h})
    synergy_records = []
    shift_switch_diagnostic_records = []

    def append_record(ep_i, offset, policy, sim, wait_h=0.0, **extra):
        wall_h = sim.cursor_s / 3600.0
        completion_delay_h = wait_h + wall_h
        strategy_sequence = [
            int(rec.strategy_idx)
            for rec in sim.history
        ]
        completed = sim.tokens_completed >= args.tokens - 1e-6
        feasible = (
            completed
            and completion_delay_h <= args.max_episode_hours + 1e-9
        )
        row = {
            'episode': ep_i,
            'offset_h': offset / 3600.0,
            'policy': policy,
            'carbon_g': sim.total_carbon_g,
            'energy_kj': sim.total_energy_j / 1000,
            'wall_h': wall_h,
            'wait_h': wait_h,
            'completion_delay_h': completion_delay_h,
            'start_offset_h': offset / 3600.0 + wait_h,
            'reshards': sim.reshard_count,
            'tokens_done': sim.tokens_completed,
            'completed': completed,
            'feasible': feasible,
            'strategy_sequence': json.dumps(strategy_sequence, separators=(',', ':')),
        }
        row.update(extra)
        records.append(row)

    def deadline_feasible(sim, wait_h=0.0):
        completion_delay_h = wait_h + sim.cursor_s / 3600.0
        return (
            sim.tokens_completed >= args.tokens - 1e-6
            and completion_delay_h <= args.max_episode_hours + 1e-9
        )

    def _sequence_from_sim(sim):
        return [int(rec.strategy_idx) for rec in sim.history]

    def _is_wait_on_grid(wait_s, limit_s, step_s, inclusive_limit=True):
        tol_s = 1e-6
        if wait_s < -tol_s:
            return False
        if inclusive_limit:
            if wait_s > limit_s + tol_s:
                return False
        elif wait_s >= limit_s - tol_s:
            return False
        if step_s <= 0:
            return False
        j = round(wait_s / step_s)
        return abs(wait_s - j * step_s) <= tol_s

    def build_shift_switch_candidate_table(offset, shift_strategy_idx):
        rows = []
        decision_interval_s = 3600.0
        deadline_s = args.max_episode_hours * 3600.0
        shift_window_s = args.shift_window_h * 3600.0
        shift_step_s = args.shift_step_h * 3600.0

        wait_keys = set()
        n_shift_steps = int(shift_window_s // shift_step_s)
        for j in range(n_shift_steps + 1):
            wait_keys.add(int(round(j * shift_step_s * 1_000_000)))
        n_oracle_steps = int(np.ceil(deadline_s / decision_interval_s))
        for j in range(n_oracle_steps):
            wait_s = j * decision_interval_s
            if wait_s < deadline_s - 1e-6:
                wait_keys.add(int(round(wait_s * 1_000_000)))

        for wait_key in sorted(wait_keys):
            wait_s = wait_key / 1_000_000.0
            wait_h = wait_s / 3600.0
            start_offset_s = float(offset) + wait_s
            remaining_h = (deadline_s - wait_s) / 3600.0
            job_considered = _is_wait_on_grid(
                wait_s,
                shift_window_s,
                shift_step_s,
                inclusive_limit=True,
            )
            oracle_considered = _is_wait_on_grid(
                wait_s,
                deadline_s,
                decision_interval_s,
                inclusive_limit=False,
            )

            row = {
                'candidate_wait_h': wait_h,
                'candidate_start_time': start_offset_s / 3600.0,
                'remaining_horizon_h': remaining_h,
                'job_shift_considered': job_considered,
                'oracle_considered': oracle_considered,
                'job_shift_feasible': False,
                'job_shift_dynamic_parallelism_feasible': False,
                'job_shift_carbon_if_feasible': np.nan,
                'oracle_dp_carbon_if_feasible': np.nan,
                'oracle_strategy_sequence': '',
                'oracle_num_reshards': np.nan,
                'completion_time': np.nan,
                'job_shift_completion_time_h': np.nan,
                'oracle_completion_time_h': np.nan,
                'reason_if_infeasible': '',
            }
            reasons = []

            if remaining_h <= 0:
                if job_considered:
                    reasons.append('job:no_remaining_horizon')
                if oracle_considered:
                    reasons.append('oracle:no_remaining_horizon')
                row['reason_if_infeasible'] = '; '.join(reasons)
                rows.append(row)
                continue

            if job_considered:
                try:
                    job_sim = rollout_static(
                        cfg,
                        trace_t,
                        trace_ci,
                        start_offset_s,
                        strategy_idx=shift_strategy_idx,
                        max_hours=remaining_h,
                        wrap_trace=args.wrap_trace,
                    )
                    job_completion_h = wait_h + job_sim.cursor_s / 3600.0
                    row['job_shift_completion_time_h'] = job_completion_h
                    job_feasible = (
                        job_sim.done()
                        and job_completion_h <= args.max_episode_hours + 1e-9
                    )
                    row['job_shift_feasible'] = job_feasible
                    if job_feasible:
                        row['job_shift_carbon_if_feasible'] = job_sim.total_carbon_g
                    elif not job_sim.done():
                        reasons.append(
                            f'job:incomplete_tokens={job_sim.tokens_completed:.0f}'
                        )
                    else:
                        reasons.append(f'job:miss_deadline={job_completion_h:.4f}h')
                except Exception as exc:
                    reasons.append(f'job:{type(exc).__name__}')
            else:
                reasons.append('job:not_considered')

            if oracle_considered:
                try:
                    oracle_sim = rollout_oracle_dp(
                        cfg,
                        trace_t,
                        trace_ci,
                        start_offset_s,
                        decision_interval_s=decision_interval_s,
                        max_hours=remaining_h,
                        token_quantum=args.oracle_token_quantum,
                        switch_cost_mode=args.oracle_switch_cost_mode,
                        wrap_trace=args.wrap_trace,
                        debug=False,
                    )
                    oracle_completion_h = wait_h + oracle_sim.cursor_s / 3600.0
                    oracle_actions = _sequence_from_sim(oracle_sim)
                    row['oracle_completion_time_h'] = oracle_completion_h
                    row['oracle_strategy_sequence'] = json.dumps(
                        oracle_actions,
                        separators=(',', ':'),
                    )
                    row['oracle_num_reshards'] = oracle_sim.reshard_count
                    oracle_feasible = (
                        oracle_sim.done()
                        and oracle_completion_h <= args.max_episode_hours + 1e-9
                    )
                    row['job_shift_dynamic_parallelism_feasible'] = oracle_feasible
                    if oracle_feasible:
                        row['oracle_dp_carbon_if_feasible'] = oracle_sim.total_carbon_g
                        row['completion_time'] = oracle_completion_h
                    elif not oracle_sim.done():
                        reasons.append(
                            f'oracle:incomplete_tokens={oracle_sim.tokens_completed:.0f}'
                        )
                    else:
                        reasons.append(
                            f'oracle:miss_deadline={oracle_completion_h:.4f}h'
                        )
                except Exception as exc:
                    reasons.append(f'oracle:{type(exc).__name__}')
            else:
                reasons.append('oracle:not_considered')

            row['reason_if_infeasible'] = '; '.join(reasons)
            rows.append(row)

        return pd.DataFrame(rows)

    def dump_shift_switch_candidate_table(ep_i, offset, shift_strategy_idx, reason):
        table = build_shift_switch_candidate_table(offset, shift_strategy_idx)
        path = out_dir / f'shift_switch_candidates_ep{ep_i + 1}.csv'
        table.to_csv(path, index=False)
        shift_strategy = cfg.strategies[shift_strategy_idx]
        print('\n[job shifting + dynamic parallelism diagnostics]')
        print('reason:', reason)
        print('episode:', ep_i + 1)
        print('offset_h:', offset / 3600.0)
        print('deadline_h:', args.max_episode_hours)
        print(
            'job_shift_strategy:',
            shift_strategy_idx,
            (shift_strategy['tp'], shift_strategy['pp']),
        )
        print('candidate table:', path)
        display_cols = [
            'candidate_wait_h',
            'candidate_start_time',
            'remaining_horizon_h',
            'job_shift_considered',
            'oracle_considered',
            'job_shift_feasible',
            'job_shift_dynamic_parallelism_feasible',
            'job_shift_carbon_if_feasible',
            'oracle_dp_carbon_if_feasible',
            'oracle_strategy_sequence',
            'oracle_num_reshards',
            'completion_time',
            'reason_if_infeasible',
        ]
        with pd.option_context(
            'display.max_rows',
            None,
            'display.max_colwidth',
            96,
            'display.width',
            240,
        ):
            print(table[display_cols].to_string(index=False))
        return table

    for ep_i, offset in enumerate(offsets):
        episode_sims = {}
        episode_wait_h = {}
        static_sims_by_idx = {}

        # ----- RL policy -----
        rl_sim = None
        if include_rl_eval:
            obs, _ = env.reset(options={'trace_offset_s': float(offset)})
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            rl_sim = env.sim
            episode_sims['RL'] = rl_sim
            append_record(ep_i, offset, 'RL', rl_sim)

        # ----- Static baselines (one per strategy) -----
        for i, s in enumerate(cfg.strategies):
            sim = rollout_static(cfg, trace_t, trace_ci, float(offset),
                                 strategy_idx=i,
                                 max_hours=args.max_episode_hours * 1.5,
                                 wrap_trace=args.wrap_trace)
            label = f"static({s['tp']},{s['pp']})"
            static_sims_by_idx[i] = sim
            episode_sims[label] = sim
            append_record(ep_i, offset, label, sim)

        # ----- Temporal job-shifting baseline -----
        sim_shift = None
        delay_s = None
        shift_label = None
        if args.include_job_shift:
            sim_shift, delay_s, shift_idx = rollout_job_shifting_static_fast(
                cfg,
                trace_t,
                trace_ci,
                float(offset),
                shift_window_h=args.shift_window_h,
                shift_step_h=args.shift_step_h,
                max_hours=args.max_episode_hours,
                wrap_trace=args.wrap_trace,
                respect_deadline=True,
            )

            shift_label = JOB_SHIFT_LABEL
            wait_h = delay_s / 3600.0
            episode_sims[shift_label] = sim_shift
            episode_wait_h[shift_label] = wait_h
            append_record(
                ep_i,
                offset,
                shift_label,
                sim_shift,
                wait_h=wait_h,
                shift_strategy_idx=shift_idx,
            )

        sim_smart = None
        sim_oracle = None
        sim_joint = None
        joint_delay_s = None
        if dynamic_scheduling_applicable:
            # ----- Smart-switch baseline -----
            sim_smart = rollout_smart_switch(
                cfg,
                trace_t,
                trace_ci,
                float(offset),
                fast_idx=args.smart_fast_idx,
                candidate_indices=smart_candidates,
                lookahead_h=args.smart_lookahead_h,
                margin_g=args.smart_margin_g,
                max_hours=args.max_episode_hours * 1.5,
                wrap_trace=args.wrap_trace,
            )
            label = GREEDY_LABEL
            episode_sims[label] = sim_smart
            smart_actions = [rec.strategy_idx for rec in sim_smart.history]
            smart_initial_idx = (
                _fastest_strategy_idx(cfg)
                if args.smart_fast_idx is None or args.smart_fast_idx < 0
                else args.smart_fast_idx
            )
            if args.debug_oracle:
                smart_replay = _replay_action_sequence(
                    cfg,
                    trace_t,
                    trace_ci,
                    float(offset),
                    smart_actions,
                    initial_strategy_idx=smart_initial_idx,
                    decision_interval_s=3600.0,
                    max_hours=args.max_episode_hours * 1.5,
                    wrap_trace=args.wrap_trace,
                )
                print('[Fixed Start-Time dynamic parallelism debug] Greedy rollout')
                print('Greedy actions:', smart_actions)
                print(
                    'Greedy strategy tuples:',
                    [
                        (cfg.strategies[a]['tp'], cfg.strategies[a]['pp'])
                        for a in smart_actions
                    ],
                )
                print('Greedy carbon:', sim_smart.total_carbon_g)
                print('Greedy wall_h:', sim_smart.cursor_s / 3600.0)
                print('Greedy reshards:', sim_smart.reshard_count)
                print('Greedy replay carbon:', smart_replay.total_carbon_g)
                print('Greedy replay wall_h:', smart_replay.cursor_s / 3600.0)
                print('Greedy replay reshards:', smart_replay.reshard_count)
            append_record(ep_i, offset, label, sim_smart)

            # ----- Oracle DP baseline -----
            sim_oracle = rollout_oracle_dp(
                cfg,
                trace_t,
                trace_ci,
                float(offset),
                decision_interval_s=3600.0,
                max_hours=args.max_episode_hours * 1.5,
                token_quantum=args.oracle_token_quantum,
                switch_cost_mode=args.oracle_switch_cost_mode,
                wrap_trace=args.wrap_trace,
                candidate_action_sequences=[smart_actions],
                candidate_initial_strategy_indices=[smart_initial_idx],
                debug=args.debug_oracle,
            )
            label = BEST_CASE_SWITCHING_LABEL
            episode_sims[label] = sim_oracle
            append_record(ep_i, offset, label, sim_oracle)

            # ----- Joint shifting-and-switching oracle -----
            if args.include_job_shift:
                sim_joint, joint_delay_s = rollout_oracle_shift_switch(
                    cfg,
                    trace_t,
                    trace_ci,
                    float(offset),
                    decision_interval_s=3600.0,
                    max_episode_hours=args.max_episode_hours,
                    token_quantum=args.oracle_token_quantum,
                    switch_cost_mode=args.oracle_switch_cost_mode,
                    wrap_trace=args.wrap_trace,
                    debug=args.debug_oracle,
                )
                label = ORACLE_SHIFT_SWITCH_LABEL
                wait_h = joint_delay_s / 3600.0
                episode_sims[label] = sim_joint
                episode_wait_h[label] = wait_h
                append_record(ep_i, offset, label, sim_joint, wait_h=wait_h)

        if ep_i < args.n_detail_rollouts:
            detail_rollouts.append((offset, episode_sims, dict(episode_wait_h)))

        static_sims = [
            s for k, s in episode_sims.items()
            if k.startswith('static')
        ]
        completed_static_sims = [
            s for s in static_sims
            if s.tokens_completed >= args.tokens - 1e-6
        ]
        best_static_pool = completed_static_sims or static_sims
        best_static_note = '' if completed_static_sims else ' (incomplete)'
        best_static_carbon = min(s.total_carbon_g for s in best_static_pool)

        if completed_static_sims:
            best_completed_static_carbon = min(
                sim.total_carbon_g for sim in completed_static_sims
            )
        else:
            best_completed_static_carbon = float('inf')

        smart_label = None
        smart_carbon = float('inf')
        if dynamic_scheduling_applicable:
            smart_labels = [
                name for name in episode_sims
                if name == GREEDY_LABEL
            ]
            smart_label = smart_labels[0] if smart_labels else None
            if smart_label is not None:
                smart_sim = episode_sims[smart_label]
                smart_completed = smart_sim.tokens_completed >= args.tokens - 1e-6
                smart_carbon = (
                    smart_sim.total_carbon_g if smart_completed else float('inf')
                )

        rl_carbon = None
        if include_rl_eval and rl_sim is not None:
            rl_carbon = (
                rl_sim.total_carbon_g
                if rl_sim.tokens_completed >= args.tokens - 1e-6
                else float('inf')
            )
        oracle_carbon = (
            episode_sims[BEST_CASE_SWITCHING_LABEL].total_carbon_g
            if dynamic_scheduling_applicable
            else float('inf')
        )
        available = []
        if np.isfinite(best_completed_static_carbon):
            available.append(best_completed_static_carbon)
        if np.isfinite(smart_carbon):
            available.append(smart_carbon)
        if rl_carbon is not None and np.isfinite(rl_carbon):
            available.append(rl_carbon)
        # Job shifting has a different feasible set because it can delay the
        # start time. Do not include it in the oracle-DP invariant.
        available_best = min(available) if available else float('inf')
        if dynamic_scheduling_applicable and args.debug_oracle:
            print('[Fixed Start-Time dynamic parallelism debug] comparison')
            print('best_static_carbon:', best_completed_static_carbon)
            print('Greedy_label:', smart_label)
            print('Greedy_carbon:', smart_carbon)
            if include_rl_eval:
                print('rl_carbon:', rl_carbon)
            print('Fixed Start-Time dynamic parallelism carbon:', oracle_carbon)
        oracle_tolerance_g = 1e-3
        if (dynamic_scheduling_applicable
                and oracle_carbon > available_best + oracle_tolerance_g):
            print(
                '\n[FIXED START-TIME DYNAMIC PARALLELISM BUG] '
                'Fixed Start-Time dynamic parallelism is worse than an available schedule'
            )
            print('episode:', ep_i)
            print('offset_h:', offset / 3600.0)
            print('best_static_carbon:', best_completed_static_carbon)
            print('Greedy_label:', smart_label)
            print('Greedy_carbon:', smart_carbon)
            if include_rl_eval:
                print('rl_carbon:', rl_carbon)
            print('Fixed Start-Time dynamic parallelism carbon:', oracle_carbon)

            for name, sim in episode_sims.items():
                print(
                    name,
                    'carbon=', sim.total_carbon_g,
                    'wall_h=', sim.cursor_s / 3600.0,
                    'reshards=', sim.reshard_count,
                    'tokens=', sim.tokens_completed,
                    'completed=', sim.tokens_completed >= args.tokens - 1e-6,
                )

            if args.debug_oracle:
                raise AssertionError(
                    'Fixed Start-Time dynamic parallelism is worse than a '
                    'completed static/Greedy schedule'
                )

        if (args.include_job_shift
                and dynamic_scheduling_applicable
                and sim_shift is not None
                and sim_oracle is not None
                and sim_joint is not None):
            feasible_static_sims = [
                sim for sim in static_sims_by_idx.values()
                if deadline_feasible(sim)
            ]
            c0 = (
                min(sim.total_carbon_g for sim in feasible_static_sims)
                if feasible_static_sims else np.nan
            )
            shift_wait_h = float(delay_s or 0.0) / 3600.0
            joint_wait_h = float(joint_delay_s or 0.0) / 3600.0
            cshift = (
                sim_shift.total_carbon_g
                if deadline_feasible(sim_shift, shift_wait_h) else np.nan
            )
            cswitch = (
                sim_oracle.total_carbon_g
                if deadline_feasible(sim_oracle) else np.nan
            )
            cjoint = (
                sim_joint.total_carbon_g
                if deadline_feasible(sim_joint, joint_wait_h) else np.nan
            )
            synergy = np.nan
            if np.all(np.isfinite([c0, cshift, cswitch, cjoint])):
                gshift = c0 - cshift
                gswitch = c0 - cswitch
                gjoint = c0 - cjoint
                synergy = cshift + cswitch - cjoint - c0
            else:
                gshift = np.nan
                gswitch = np.nan
                gjoint = np.nan
            synergy_records.append({
                'episode': ep_i,
                'offset_h': offset / 3600.0,
                'C0_static_immediate_g': c0,
                'Cshift_g': cshift,
                'Cswitch_g': cswitch,
                'Cjoint_g': cjoint,
                'Gshift_g': gshift,
                'Gswitch_g': gswitch,
                'Gjoint_g': gjoint,
                'synergy_g': synergy,
                'positive_synergy': (
                    bool(synergy > 1e-6) if np.isfinite(synergy) else np.nan
                ),
            })

            shift_feasible = deadline_feasible(sim_shift, shift_wait_h)
            joint_feasible = deadline_feasible(sim_joint, joint_wait_h)
            shift_switch_diagnostic_records.append({
                'episode': ep_i,
                'offset_h': offset / 3600.0,
                'job_shifting_wait_h': shift_wait_h,
                'job_shifting_carbon_g': sim_shift.total_carbon_g,
                'job_shifting_feasible': shift_feasible,
                'job_shifting_strategy_sequence': json.dumps(
                    _sequence_from_sim(sim_shift),
                    separators=(',', ':'),
                ),
                'job_shift_dynamic_parallelism_wait_h': joint_wait_h,
                'job_shift_dynamic_parallelism_carbon_g': sim_joint.total_carbon_g,
                'job_shift_dynamic_parallelism_feasible': joint_feasible,
                'job_shift_dynamic_parallelism_num_reshards': sim_joint.reshard_count,
                'job_shift_dynamic_parallelism_strategy_sequence': json.dumps(
                    _sequence_from_sim(sim_joint),
                    separators=(',', ':'),
                ),
            })

            should_dump_shift_switch = (
                args.debug_shift_switch
                and (
                    args.debug_shift_switch_episode is None
                    or args.debug_shift_switch_episode == ep_i + 1
                )
            )
            if should_dump_shift_switch:
                dump_shift_switch_candidate_table(
                    ep_i,
                    offset,
                    shift_idx,
                    'requested debug dump',
                )

            if shift_feasible and (
                    not joint_feasible
                    or sim_joint.total_carbon_g > sim_shift.total_carbon_g + 1e-6):
                dump_shift_switch_candidate_table(
                    ep_i,
                    offset,
                    shift_idx,
                    'invariant failure: job shifting + dynamic parallelism '
                    'worse than Job Shifting',
                )
                raise AssertionError(
                    'job shifting + dynamic parallelism must be no worse than '
                    'deadline-feasible Job Shifting'
                )

        msg = (
            f'episode {ep_i+1:>2}/{args.n_episodes} '
            f'offset={offset/3600.0:7.2f}h  '
        )
        if include_rl_eval and rl_sim is not None:
            msg += f'RL={rl_sim.total_carbon_g:7.1f}g  '
        msg += (
            f'best-static{best_static_note}={best_static_carbon:7.1f}g  '
        )
        if args.include_job_shift and sim_shift is not None:
            msg += (
                f'{JOB_SHIFT_LABEL}={sim_shift.total_carbon_g:7.1f}g  '
                f'wait={delay_s/3600.0:5.1f}h  '
            )
        if dynamic_scheduling_applicable:
            msg += (
                f'{GREEDY_LABEL}={sim_smart.total_carbon_g:7.1f}g  '
                f'{BEST_CASE_SWITCHING_LABEL}={sim_oracle.total_carbon_g:7.1f}g'
            )
            if args.include_job_shift and sim_joint is not None:
                msg += (
                    f'  {ORACLE_SHIFT_SWITCH_LABEL}='
                    f'{sim_joint.total_carbon_g:7.1f}g  '
                    f'wait={float(joint_delay_s or 0.0)/3600.0:5.1f}h'
                )
        else:
            msg += 'dynamic=not-applicable'
        if include_rl_eval and rl_sim is not None:
            msg += f'  reshards(RL)={rl_sim.reshard_count}'
        print(msg)

    # ---- Save raw records ----
    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'results.csv', index=False)
    print(f'\nSaved per-episode results to {out_dir / "results.csv"}')
    if synergy_records:
        synergy_df = pd.DataFrame(synergy_records)
        synergy_df.to_csv(out_dir / 'superadditivity.csv', index=False)
        valid_synergy = synergy_df[np.isfinite(synergy_df['synergy_g'])].copy()
        print(f'Saved super-additivity metrics to {out_dir / "superadditivity.csv"}')
        if len(valid_synergy):
            print(
                'Super-additivity synergy: '
                f'mean={valid_synergy["synergy_g"].mean():.2f}g, '
                f'median={valid_synergy["synergy_g"].median():.2f}g, '
                f'positive episodes={int(valid_synergy["positive_synergy"].sum())}'
                f'/{len(valid_synergy)}'
            )
        else:
            print(
                'Super-additivity metrics were written, but no episode had all '
                'four deadline-feasible components.'
            )
    if shift_switch_diagnostic_records:
        shift_switch_diag_df = pd.DataFrame(shift_switch_diagnostic_records)
        shift_switch_diag_df.to_csv(
            out_dir / 'shift_switch_diagnostics.csv',
            index=False,
        )
        print(
            f'Saved shift/switch diagnostics to '
            f'{out_dir / "shift_switch_diagnostics.csv"}'
        )
    completed_df = df[df['completed']].copy()
    incomplete = df[~df['completed']].copy()
    if len(incomplete):
        incomplete_counts = incomplete.groupby('policy').size().sort_values(ascending=False)
        print('\nIncomplete runs excluded from carbon summaries:')
        print(incomplete_counts.to_string())

    # ---- Aggregate plot: box + strip per policy ----
    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    policies = list(df['policy'].unique())
    carbon_policies = [
        p for p in policies
        if len(completed_df.loc[completed_df['policy'] == p])
    ]
    carbon_plot_df = completed_df if carbon_policies else df
    if not carbon_policies:
        carbon_policies = policies
    carbons_per_policy = [
        carbon_plot_df.loc[carbon_plot_df['policy'] == p, 'carbon_g'].values
        for p in carbon_policies
    ]
    completion_per_policy = [
        df.loc[df['policy'] == p, 'completion_delay_h'].values
        for p in policies
    ]

    bp = ax_c.boxplot(carbons_per_policy, tick_labels=carbon_policies,
                      patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(carbon_policies)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, vals in enumerate(carbons_per_policy):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        ax_c.scatter(x, vals, alpha=0.6, s=18, color='k')
    ax_c.set_ylabel('Total carbon (gCO₂)')
    if len(completed_df):
        ax_c.set_title('Carbon over completed fixed-token runs')
    else:
        ax_c.set_title('Carbon (no completed fixed-token runs)')
    ax_c.grid(True, alpha=0.3, axis='y')
    plt.setp(ax_c.get_xticklabels(), rotation=25, ha='right')

    bp2 = ax_w.boxplot(completion_per_policy, tick_labels=policies,
                       patch_artist=True, showfliers=False)
    for patch, color in zip(bp2['boxes'], plt.cm.tab10(np.linspace(0, 1, len(policies)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, vals in enumerate(completion_per_policy):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        ax_w.scatter(x, vals, alpha=0.6, s=18, color='k')
    ax_w.set_ylabel('Completion delay (hours)')
    ax_w.set_title('Completion delay')
    ax_w.grid(True, alpha=0.3, axis='y')
    plt.setp(ax_w.get_xticklabels(), rotation=25, ha='right')

    plt.tight_layout()
    fig.savefig(out_dir / 'aggregate.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved aggregate plot to {out_dir / "aggregate.png"}')

    # ---- Per-episode detail plots ----
    for offset, episode_sims, episode_wait_h in detail_rollouts:
        # Drop non-best static rollouts to keep the figure readable.
        static_labels = [k for k in episode_sims if k.startswith('static')]
        completed_static_labels = [
            k for k in static_labels
            if episode_sims[k].tokens_completed >= args.tokens - 1e-6
        ]
        best_static_label = min(
            completed_static_labels or static_labels,
            key=lambda k: episode_sims[k].total_carbon_g,
        )
        keep = {
            f'Best Static = {best_static_label}': episode_sims[best_static_label],
        }
        smart_plot_label = GREEDY_LABEL
        if smart_plot_label in episode_sims:
            keep[smart_plot_label] = episode_sims[smart_plot_label]
        if args.include_job_shift:
            shift_plot_label = JOB_SHIFT_LABEL
            if shift_plot_label in episode_sims:
                wait_h = episode_wait_h.get(shift_plot_label, 0.0)
                keep[
                    f'{JOB_SHIFT_LABEL} (wait={wait_h:.1f}h)'
                ] = episode_sims[shift_plot_label]
        if BEST_CASE_SWITCHING_LABEL in episode_sims:
            keep[BEST_CASE_SWITCHING_LABEL] = episode_sims[BEST_CASE_SWITCHING_LABEL]
        if ORACLE_SHIFT_SWITCH_LABEL in episode_sims:
            wait_h = episode_wait_h.get(ORACLE_SHIFT_SWITCH_LABEL, 0.0)
            keep[
                f'{ORACLE_SHIFT_SWITCH_LABEL} (wait={wait_h:.1f}h)'
            ] = episode_sims[ORACLE_SHIFT_SWITCH_LABEL]
        if include_rl_eval:
            keep = {
                f'RL ({algo.upper()})': episode_sims['RL'],
                **keep,
            }
        fig = plot_comparison(keep, cfg,
                              title=f'Episode at trace offset {offset/3600.0:.2f}h')
        fname = out_dir / f'rollout_offset_{offset/3600.0:.1f}h.png'
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

    # ---- Summary table ----
    if len(completed_df):
        summary = (completed_df.groupby('policy')
                     .agg(completed_runs=('completed', 'size'),
                          carbon_mean=('carbon_g', 'mean'),
                          carbon_std=('carbon_g', 'std'),
                          carbon_min=('carbon_g', 'min'),
                          carbon_max=('carbon_g', 'max'),
                          wall_mean=('wall_h', 'mean'),
                          wait_mean=('wait_h', 'mean'),
                          wait_median=('wait_h', 'median'),
                          completion_delay_mean=('completion_delay_h', 'mean'),
                          reshards_med=('reshards', 'median'),
                          tokens_min=('tokens_done', 'min'))
                     .round(2)
                     .sort_values('carbon_mean'))
        print('\nSummary across completed fixed-token episodes:')
        print(summary.to_string())
        summary.to_csv(out_dir / 'summary.csv')
    else:
        summary = (df.groupby('policy')
                     .agg(incomplete_runs=('completed', 'size'),
                          tokens_max=('tokens_done', 'max'),
                          wall_mean=('wall_h', 'mean'),
                          wait_mean=('wait_h', 'mean'),
                          completion_delay_mean=('completion_delay_h', 'mean'),
                          carbon_mean_incomplete=('carbon_g', 'mean'))
                     .round(2))
        print('\nNo policy completed the requested fixed-token budget.')
        print(summary.to_string())
        summary.to_csv(out_dir / 'summary.csv')


if __name__ == '__main__':
    main()

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
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from carbon_aware_env import (
    default_config, load_carbon_trace, make_env,
    rollout_static, rollout_job_shifting_static_fast,
    rollout_smart_switch, rollout_oracle_dp,
    plot_comparison, CarbonSim, _fastest_strategy_idx, _replay_action_sequence,
)


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
                   help='Fast/catch-up strategy index for smart-switch baseline. '
                        'Use -1 to automatically pick the highest-throughput '
                        'strategy.')
    p.add_argument('--smart-lookahead-h', type=int, default=6,
                   help='Lookahead window for smart-switch clean catch-up CI')
    p.add_argument('--smart-margin-g', type=float, default=0.0,
                   help='Extra required carbon benefit before smart-switch '
                        'changes strategy')
    p.add_argument('--smart-candidates', type=str, default=None,
                   help='Comma-separated strategy indices for smart-switch '
                        'dirty-window candidates, e.g. "1,2". Default None '
                        'means auto-select lower-power/slower candidates.')
    p.add_argument('--include-job-shift', action='store_true',
                   help='Evaluate temporal job-shifting baseline.')
    p.add_argument('--shift-window-h', type=float, default=24.0,
                   help='Maximum allowed delay before starting the job.')
    p.add_argument('--shift-step-h', type=float, default=1.0,
                   help='Grid step for searching possible delayed start times.')
    p.add_argument('--shift-strategy',
                   choices=['fastest'],
                   default='fastest',
                   help='Strategy used after shifting. For now only fastest throughput.')
    p.add_argument('--oracle-token-quantum', type=float, default=5_000_000.0,
                   help='Token discretization for oracle DP baseline')
    p.add_argument('--oracle-switch-cost-mode',
                   choices=['sim', 'fixed'],
                   default='sim',
                   help='Switch-cost model for oracle DP baseline')
    p.add_argument('--debug-oracle', action='store_true',
                   help='Print oracle DP internals and assert if oracle loses '
                        'to completed static/smart schedules')
    p.add_argument('--fixed-offset-h', type=float, default=None,
                   help='Evaluate every episode at this exact trace offset '
                        '(hours), useful for reproducing oracle bugs')
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
            "Skipping oracle/scheduler savings comparison."
        )
        print(
            "Only one feasible strategy remained for this model size on the "
            "target hardware, so dynamic scheduling is not applicable. We "
            "report this as a feasibility-boundary case rather than an "
            "oracle/scheduler win."
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
                '[warning] No requested smart-switch candidates remain; '
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
    episode_metrics = []
    detail_rollouts = []  # list of (offset, {label: CarbonSim}, {label: wait_h})

    def append_record(ep_i, offset, policy, sim, wait_h=0.0, **extra):
        wall_h = sim.cursor_s / 3600.0
        row = {
            'episode': ep_i,
            'offset_h': offset / 3600.0,
            'policy': policy,
            'carbon_g': sim.total_carbon_g,
            'energy_kj': sim.total_energy_j / 1000,
            'wall_h': wall_h,
            'wait_h': wait_h,
            'completion_delay_h': wait_h + wall_h,
            'reshards': sim.reshard_count,
            'tokens_done': sim.tokens_completed,
            'completed': sim.tokens_completed >= args.tokens - 1e-6,
        }
        row.update(extra)
        records.append(row)

    for ep_i, offset in enumerate(offsets):
        episode_sims = {}
        episode_wait_h = {}

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
            )

            shift_label = f'job-shift-fast({args.shift_window_h:g}h)'
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
            label = f'smart-switch({args.smart_lookahead_h}h)'
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
                print('[oracle debug] smart-switch rollout')
                print('smart actions:', smart_actions)
                print(
                    'smart strategy tuples:',
                    [
                        (cfg.strategies[a]['tp'], cfg.strategies[a]['pp'])
                        for a in smart_actions
                    ],
                )
                print('smart carbon:', sim_smart.total_carbon_g)
                print('smart wall_h:', sim_smart.cursor_s / 3600.0)
                print('smart reshards:', sim_smart.reshard_count)
                print('smart replay carbon:', smart_replay.total_carbon_g)
                print('smart replay wall_h:', smart_replay.cursor_s / 3600.0)
                print('smart replay reshards:', smart_replay.reshard_count)
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
            label = 'oracle-dp'
            episode_sims[label] = sim_oracle
            append_record(ep_i, offset, label, sim_oracle)

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
                if name.startswith('smart-switch')
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
            episode_sims['oracle-dp'].total_carbon_g
            if dynamic_scheduling_applicable
            else float('inf')
        )
        job_shift_carbon = float('inf')
        job_shift_wait_h = float('nan')
        job_shift_completion_delay_h = float('nan')
        if args.include_job_shift and sim_shift is not None:
            job_shift_wait_h = delay_s / 3600.0
            job_shift_completion_delay_h = job_shift_wait_h + sim_shift.cursor_s / 3600.0
            if sim_shift.tokens_completed >= args.tokens - 1e-6:
                job_shift_carbon = sim_shift.total_carbon_g
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
            print('[oracle debug] comparison')
            print('best_static_carbon:', best_completed_static_carbon)
            print('smart_label:', smart_label)
            print('smart_carbon:', smart_carbon)
            if include_rl_eval:
                print('rl_carbon:', rl_carbon)
            print('oracle_carbon:', oracle_carbon)
        oracle_tolerance_g = 1e-3
        if (dynamic_scheduling_applicable
                and oracle_carbon > available_best + oracle_tolerance_g):
            print('\n[ORACLE BUG] oracle-dp is worse than an available schedule')
            print('episode:', ep_i)
            print('offset_h:', offset / 3600.0)
            print('best_static_carbon:', best_completed_static_carbon)
            print('smart_label:', smart_label)
            print('smart_carbon:', smart_carbon)
            if include_rl_eval:
                print('rl_carbon:', rl_carbon)
            print('oracle_carbon:', oracle_carbon)

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
                    'oracle-dp is worse than a completed static/smart schedule'
                )

        def savings_pct(base_carbon, policy_carbon):
            if not (np.isfinite(base_carbon) and np.isfinite(policy_carbon)):
                return float('nan')
            return 100.0 * (base_carbon - policy_carbon) / base_carbon

        oracle_metric_carbon = float('nan')
        if (dynamic_scheduling_applicable and sim_oracle is not None
                and sim_oracle.tokens_completed >= args.tokens - 1e-6):
            oracle_metric_carbon = sim_oracle.total_carbon_g
        smart_metric_carbon = smart_carbon if np.isfinite(smart_carbon) else float('nan')
        best_static_metric_carbon = (
            best_completed_static_carbon
            if np.isfinite(best_completed_static_carbon)
            else float('nan')
        )
        job_shift_metric_carbon = (
            job_shift_carbon if np.isfinite(job_shift_carbon) else float('nan')
        )
        episode_metrics.append({
            'episode': ep_i,
            'offset_h': offset / 3600.0,
            'best_static_carbon': best_static_metric_carbon,
            'job_shift_fast_carbon': job_shift_metric_carbon,
            'smart_switch_carbon': smart_metric_carbon,
            'oracle_dp_carbon': oracle_metric_carbon,
            'shift_savings_pct': savings_pct(
                best_static_metric_carbon,
                job_shift_metric_carbon,
            ),
            'oracle_savings_pct': savings_pct(
                best_static_metric_carbon,
                oracle_metric_carbon,
            ),
            'smart_savings_pct': savings_pct(
                best_static_metric_carbon,
                smart_metric_carbon,
            ),
            'job_shift_wait_h': job_shift_wait_h,
            'job_shift_completion_delay_h': job_shift_completion_delay_h,
        })

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
                f'shift={sim_shift.total_carbon_g:7.1f}g  '
                f'wait={delay_s/3600.0:5.1f}h  '
            )
        if dynamic_scheduling_applicable:
            msg += (
                f'smart={sim_smart.total_carbon_g:7.1f}g  '
                f'oracle={sim_oracle.total_carbon_g:7.1f}g'
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
    episode_metrics_df = pd.DataFrame(episode_metrics)
    if len(episode_metrics_df):
        episode_metrics_df.to_csv(out_dir / 'episode_metrics.csv', index=False)
        print(f'Saved paired episode metrics to {out_dir / "episode_metrics.csv"}')
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

    bp = ax_c.boxplot(carbons_per_policy, labels=carbon_policies, patch_artist=True,
                       showfliers=False)
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

    bp2 = ax_w.boxplot(completion_per_policy, labels=policies, patch_artist=True,
                        showfliers=False)
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
            f'best static = {best_static_label}': episode_sims[best_static_label],
        }
        smart_plot_label = f'smart-switch({args.smart_lookahead_h}h)'
        if smart_plot_label in episode_sims:
            keep[smart_plot_label] = episode_sims[smart_plot_label]
        if args.include_job_shift:
            shift_plot_label = f'job-shift-fast({args.shift_window_h:g}h)'
            if shift_plot_label in episode_sims:
                wait_h = episode_wait_h.get(shift_plot_label, 0.0)
                keep[
                    f'job-shift-fast({args.shift_window_h:g}h, '
                    f'wait={wait_h:.1f}h)'
                ] = episode_sims[shift_plot_label]
        if 'oracle-dp' in episode_sims:
            keep['oracle-dp'] = episode_sims['oracle-dp']
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

    if len(episode_metrics_df):
        def comparison_policy_rows(policy):
            if policy == 'job-shift-fast':
                mask = completed_df['policy'].str.startswith('job-shift-fast')
            elif policy == 'smart-switch':
                mask = completed_df['policy'].str.startswith('smart-switch')
            else:
                mask = completed_df['policy'] == policy
            return completed_df.loc[mask]

        comparison_specs = [
            ('job-shift-fast', 'job_shift_fast_carbon', 'shift_savings_pct'),
            ('smart-switch', 'smart_switch_carbon', 'smart_savings_pct'),
            ('oracle-dp', 'oracle_dp_carbon', 'oracle_savings_pct'),
        ]
        comparison_rows = []
        for policy, carbon_col, savings_col in comparison_specs:
            valid = (
                episode_metrics_df['best_static_carbon'].notna()
                & episode_metrics_df[carbon_col].notna()
            )
            if not valid.any():
                continue
            subset = episode_metrics_df.loc[valid]
            row = {
                'policy': policy,
                'paired_runs': int(valid.sum()),
                'best_static_carbon_mean': subset['best_static_carbon'].mean(),
                'policy_carbon_mean': subset[carbon_col].mean(),
                'savings_pct_mean': subset[savings_col].mean(),
                'savings_pct_median': subset[savings_col].median(),
            }
            policy_rows = comparison_policy_rows(policy)
            policy_rows = policy_rows[policy_rows['episode'].isin(subset['episode'])]
            if len(policy_rows):
                row.update({
                    'mean_wait_h': policy_rows['wait_h'].mean(),
                    'median_wait_h': policy_rows['wait_h'].median(),
                    'mean_completion_delay_h': policy_rows['completion_delay_h'].mean(),
                })
            comparison_rows.append(row)

        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows).round(2)
            print('\nPaired savings versus best completed static:')
            print(comparison_df.to_string(index=False))
            comparison_df.to_csv(out_dir / 'savings_summary.csv', index=False)


if __name__ == '__main__':
    main()

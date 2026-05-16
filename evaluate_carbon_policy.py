"""Evaluate a trained carbon-aware policy against baselines.

Runs N episodes for each policy on identical trace-offset draws so the
comparison is paired. Produces:
  - aggregate.png: box/strip plot of total carbon per policy across N episodes
  - rollout_offset_<h>h.png: per-episode CI / strategy / cumulative-carbon plot
  - results.csv: raw per-episode numbers

Usage:
    python evaluate_carbon_policy.py \
        --trace /path/to/carbon_week.csv \
        --model runs/ppo_carbon/model.zip \
        --tokens 3000000000 \
        --n-episodes 32 \
        --out runs/ppo_carbon/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from carbon_aware_env import (
    SimConfig, default_config, load_carbon_trace, make_env,
    rollout_static, rollout_ci_threshold, plot_comparison, CarbonSim,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True)
    p.add_argument('--model', required=True, help='Path to SB3 model .zip')
    p.add_argument('--algo', choices=['ppo', 'dqn'], default='ppo')
    p.add_argument('--tokens', type=int, default=3_000_000_000)
    p.add_argument('--global-batch-size', type=int, default=16)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--forecast-horizon-h', type=int, default=6)
    p.add_argument('--max-episode-hours', type=float, default=72.0)
    p.add_argument('--n-episodes', type=int, default=32,
                   help='Number of trace-offset draws to evaluate on')
    p.add_argument('--ci-threshold', type=float, default=60.0,
                   help='Threshold for CI-threshold baseline (g/kWh)')
    p.add_argument('--low-idx', type=int, default=0,
                   help='Strategy index for low-CI in threshold baseline')
    p.add_argument('--high-idx', type=int, default=2,
                   help='Strategy index for high-CI in threshold baseline')
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


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_t, trace_ci, _ = load_carbon_trace(args.trace)
    cfg = default_config(
        pretraining_tokens=args.tokens,
        global_batch_size=args.global_batch_size,
        seq_length=args.seq_length,
    )

    # Load model
    if args.algo == 'ppo':
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
    else:
        from stable_baselines3 import DQN
        model = DQN.load(args.model)

    # Build a single env for evaluation (we control the seed/offsets manually)
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
    slack = args.max_episode_hours * 3600.0 + 3600.0
    max_offset = max(0.0, trace_dur - slack)
    offsets = rng.uniform(0.0, max_offset, size=args.n_episodes)

    records = []
    detail_rollouts = []  # list of (offset, {label: CarbonSim})

    for ep_i, offset in enumerate(offsets):
        episode_sims = {}

        # ----- RL policy -----
        obs, _ = env.reset(options={'trace_offset_s': float(offset)})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        rl_sim = env.sim
        episode_sims['RL'] = rl_sim
        records.append({
            'episode': ep_i, 'offset_h': offset / 3600.0, 'policy': 'RL',
            'carbon_g': rl_sim.total_carbon_g,
            'energy_kj': rl_sim.total_energy_j / 1000,
            'wall_h': rl_sim.cursor_s / 3600.0,
            'reshards': rl_sim.reshard_count,
            'tokens_done': rl_sim.tokens_completed,
        })

        # ----- Static baselines (one per strategy) -----
        for i, s in enumerate(cfg.strategies):
            sim = rollout_static(cfg, trace_t, trace_ci, float(offset),
                                 strategy_idx=i,
                                 max_hours=args.max_episode_hours * 1.5,
                                 wrap_trace=args.wrap_trace)
            label = f"static({s['tp']},{s['pp']})"
            episode_sims[label] = sim
            records.append({
                'episode': ep_i, 'offset_h': offset / 3600.0, 'policy': label,
                'carbon_g': sim.total_carbon_g,
                'energy_kj': sim.total_energy_j / 1000,
                'wall_h': sim.cursor_s / 3600.0,
                'reshards': sim.reshard_count,
                'tokens_done': sim.tokens_completed,
            })

        # ----- CI-threshold baseline -----
        sim_th = rollout_ci_threshold(cfg, trace_t, trace_ci, float(offset),
                                      low_ci_idx=args.low_idx,
                                      high_ci_idx=args.high_idx,
                                      threshold_g_per_kwh=args.ci_threshold,
                                      max_hours=args.max_episode_hours * 1.5,
                                      wrap_trace=args.wrap_trace)
        label = f'threshold@{args.ci_threshold:.0f}'
        episode_sims[label] = sim_th
        records.append({
            'episode': ep_i, 'offset_h': offset / 3600.0, 'policy': label,
            'carbon_g': sim_th.total_carbon_g,
            'energy_kj': sim_th.total_energy_j / 1000,
            'wall_h': sim_th.cursor_s / 3600.0,
            'reshards': sim_th.reshard_count,
            'tokens_done': sim_th.tokens_completed,
        })

        if ep_i < args.n_detail_rollouts:
            detail_rollouts.append((offset, episode_sims))

        print(f'episode {ep_i+1:>2}/{args.n_episodes} '
              f'offset={offset/3600.0:7.2f}h  '
              f'RL={rl_sim.total_carbon_g:7.1f}g  '
              f'best-static={min(s.total_carbon_g for k, s in episode_sims.items() if k.startswith("static")):7.1f}g  '
              f'threshold={sim_th.total_carbon_g:7.1f}g  '
              f'reshards(RL)={rl_sim.reshard_count}')

    # ---- Save raw records ----
    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'results.csv', index=False)
    print(f'\nSaved per-episode results to {out_dir / "results.csv"}')

    # ---- Aggregate plot: box + strip per policy ----
    fig, (ax_c, ax_w) = plt.subplots(1, 2, figsize=(14, 6))
    policies = list(df['policy'].unique())
    carbons_per_policy = [df.loc[df['policy'] == p, 'carbon_g'].values for p in policies]
    walls_per_policy = [df.loc[df['policy'] == p, 'wall_h'].values for p in policies]

    bp = ax_c.boxplot(carbons_per_policy, labels=policies, patch_artist=True,
                       showfliers=False)
    for patch, color in zip(bp['boxes'], plt.cm.tab10(np.linspace(0, 1, len(policies)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, vals in enumerate(carbons_per_policy):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        ax_c.scatter(x, vals, alpha=0.6, s=18, color='k')
    ax_c.set_ylabel('Total carbon (gCO₂)')
    ax_c.set_title(f'Carbon over {args.n_episodes} trace-offset draws')
    ax_c.grid(True, alpha=0.3, axis='y')
    plt.setp(ax_c.get_xticklabels(), rotation=25, ha='right')

    bp2 = ax_w.boxplot(walls_per_policy, labels=policies, patch_artist=True,
                        showfliers=False)
    for patch, color in zip(bp2['boxes'], plt.cm.tab10(np.linspace(0, 1, len(policies)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for i, vals in enumerate(walls_per_policy):
        x = np.random.normal(i + 1, 0.05, size=len(vals))
        ax_w.scatter(x, vals, alpha=0.6, s=18, color='k')
    ax_w.set_ylabel('Wall time (hours)')
    ax_w.set_title('Wall time')
    ax_w.grid(True, alpha=0.3, axis='y')
    plt.setp(ax_w.get_xticklabels(), rotation=25, ha='right')

    plt.tight_layout()
    fig.savefig(out_dir / 'aggregate.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved aggregate plot to {out_dir / "aggregate.png"}')

    # ---- Per-episode detail plots ----
    for offset, episode_sims in detail_rollouts:
        # Drop the non-best static rollouts to keep the figure readable
        # — keep RL, best-static, threshold
        best_static_label = min(
            (k for k in episode_sims if k.startswith('static')),
            key=lambda k: episode_sims[k].total_carbon_g,
        )
        keep = {
            f'RL ({args.algo.upper()})': episode_sims['RL'],
            f'best static = {best_static_label}': episode_sims[best_static_label],
            f'threshold@{args.ci_threshold:.0f}': episode_sims[f'threshold@{args.ci_threshold:.0f}'],
        }
        fig = plot_comparison(keep, cfg,
                              title=f'Episode at trace offset {offset/3600.0:.2f}h')
        fname = out_dir / f'rollout_offset_{offset/3600.0:.1f}h.png'
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {fname}')

    # ---- Summary table ----
    summary = (df.groupby('policy')
                 .agg(carbon_mean=('carbon_g', 'mean'),
                      carbon_std=('carbon_g', 'std'),
                      carbon_min=('carbon_g', 'min'),
                      carbon_max=('carbon_g', 'max'),
                      wall_mean=('wall_h', 'mean'),
                      reshards_med=('reshards', 'median'))
                 .round(2)
                 .sort_values('carbon_mean'))
    print('\nSummary across episodes:')
    print(summary.to_string())
    summary.to_csv(out_dir / 'summary.csv')


if __name__ == '__main__':
    main()
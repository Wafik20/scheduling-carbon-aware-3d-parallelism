"""Train a PPO policy on the carbon-aware training-strategy env.

Usage:
    python train_carbon_policy.py \
        --trace carbon_trace.csv \
        --tokens 3000000000 \
        --total-timesteps 200000 \
        --out runs/ppo_carbon

Requires `stable-baselines3` and `gymnasium`. Install with:
    pip install stable-baselines3 gymnasium
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from carbon_aware_env import (
    default_config, load_carbon_trace, make_env,
    plot_training_curve,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', required=True, help='Path to hourly carbon CSV')
    p.add_argument('--tokens', type=int, default=3_000_000_000,
                   help='Pretraining token budget per episode')
    p.add_argument('--global-batch-size', type=int, default=16)
    p.add_argument('--seq-length', type=int, default=1024)
    p.add_argument('--forecast-horizon-h', type=int, default=6,
                   help='Hours of CI forecast the agent sees in obs')
    p.add_argument('--max-episode-hours', type=float, default=72.0)
    p.add_argument('--total-timesteps', type=int, default=200_000,
                   help='Total environment steps to train for')
    p.add_argument('--n-envs', type=int, default=8,
                   help='Parallel env workers for SB3 SubprocVecEnv')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wrap-trace', action='store_true',
                   help='Tile the trace cyclically — required if total episode '
                        'time exceeds trace length. Useful when training on a '
                        'short real trace.')
    p.add_argument('--out', default='runs/ppo_carbon',
                   help='Directory to write model + logs')
    p.add_argument('--algo', choices=['ppo', 'dqn'], default='ppo')
    p.add_argument('--learning-rate', type=float, default=3e-4)
    return p.parse_args()


class EpisodeReturnCallback:
    """Lightweight callback shim: SB3 callbacks subclass BaseCallback. We do
    that below in `build_callback`. This class exists only as a placeholder so
    that the module can be imported without sb3 installed.
    """
    pass


def build_callback(log_dir: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class _Cb(BaseCallback):
        """Records per-episode return (== -gCO₂) and dumps to numpy."""
        def __init__(self):
            super().__init__()
            self.episode_returns: list[float] = []
            self.episode_lengths: list[int] = []
            self.episode_carbons: list[float] = []
            self.episode_reshards: list[int] = []
            self.episode_progress: list[float] = []

        def _on_step(self) -> bool:
            infos = self.locals.get('infos', [])
            dones = self.locals.get('dones', [])
            for info, done in zip(infos, dones):
                if not done:
                    continue
                ep = info.get('episode')
                if ep is not None:
                    self.episode_returns.append(float(ep['r']))
                    self.episode_lengths.append(int(ep['l']))
                # Final-step infos are passed through by VecEnv; pull our fields
                self.episode_carbons.append(float(info.get('total_carbon_g', np.nan)))
                self.episode_reshards.append(int(info.get('reshard_count', 0)))
                self.episode_progress.append(float(info.get('progress', np.nan)))
            return True

        def _on_training_end(self):
            np.savez(log_dir / 'training_log.npz',
                     returns=np.array(self.episode_returns),
                     lengths=np.array(self.episode_lengths),
                     carbons=np.array(self.episode_carbons),
                     reshards=np.array(self.episode_reshards),
                     progress=np.array(self.episode_progress))

    return _Cb()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading carbon trace from {args.trace}')
    trace_t, trace_ci, _ = load_carbon_trace(args.trace)
    print(f'  {len(trace_t)} hourly samples, '
          f'{(trace_t[-1] + 3600) / 86400:.1f} days, '
          f'CI {trace_ci.min():.1f}–{trace_ci.max():.1f} g/kWh')

    cfg = default_config(
        pretraining_tokens=args.tokens,
        global_batch_size=args.global_batch_size,
        seq_length=args.seq_length,
    )
    print(f'  {cfg.pretraining_tokens:,} tokens, {len(cfg.strategies)} strategies')

    # Estimated reference: best static (1,1) carbon for offset 0
    from carbon_aware_env import rollout_static
    static_sims = [rollout_static(cfg, trace_t, trace_ci, 0.0, i)
                   for i in range(len(cfg.strategies))]
    print('Reference static rollouts at offset 0:')
    for i, sim in enumerate(static_sims):
        s = cfg.strategies[i]
        print(f"  ({s['tp']},{s['pp']}): wall={sim.cursor_s/3600.0:.2f}h "
              f'carbon={sim.total_carbon_g:.1f} gCO₂')

    # Build the vectorized env (subproc for true parallelism)
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    def env_factory():
        env = make_env(
            cfg, trace_t, trace_ci,
            forecast_horizon_h=args.forecast_horizon_h,
            max_episode_hours=args.max_episode_hours,
            random_start=True,
            wrap_trace=args.wrap_trace,
        )
        return Monitor(env)

    vec_env = make_vec_env(env_factory, n_envs=args.n_envs,
                           seed=args.seed, vec_env_cls=SubprocVecEnv)

    # Build the algorithm
    if args.algo == 'ppo':
        from stable_baselines3 import PPO
        model = PPO(
            'MlpPolicy', vec_env,
            learning_rate=args.learning_rate,
            n_steps=128,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(out_dir / 'tb'),
        )
    elif args.algo == 'dqn':
        from stable_baselines3 import DQN
        model = DQN(
            'MlpPolicy', vec_env,
            learning_rate=args.learning_rate,
            buffer_size=50_000,
            learning_starts=2000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(out_dir / 'tb'),
        )

    cb = build_callback(out_dir)
    print(f'\nTraining {args.algo.upper()} for {args.total_timesteps} timesteps...')
    model.learn(total_timesteps=args.total_timesteps, callback=cb)

    model.save(str(out_dir / 'model'))
    print(f'\nSaved model to {out_dir / "model.zip"}')

    # Save plot of training curve
    if cb.episode_returns:
        fig = plot_training_curve(cb.episode_returns,
                                  title=f'{args.algo.upper()} training curve '
                                        f'({args.tokens/1e9:.1f}B tokens/episode)')
        fig.savefig(out_dir / 'training_curve.png', dpi=120, bbox_inches='tight')
        print(f'Saved training curve to {out_dir / "training_curve.png"}')

        # Print summary
        ret_arr = np.array(cb.episode_returns)
        carbon_arr = np.array(cb.episode_carbons)
        print(f'\nTraining summary ({len(ret_arr)} episodes):')
        print(f'  final 10% mean return: {ret_arr[-len(ret_arr)//10:].mean():.2f}')
        print(f'  final 10% mean carbon: {carbon_arr[-len(carbon_arr)//10:].mean():.2f} gCO₂')
        print(f'  best episode carbon:   {carbon_arr.min():.2f} gCO₂')
        print(f'  median reshards:       {int(np.median(cb.episode_reshards))}')


if __name__ == '__main__':
    main()

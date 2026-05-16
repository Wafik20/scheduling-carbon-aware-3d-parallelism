"""Carbon-aware training-strategy selection as a Gymnasium environment.

The environment wraps the same calibration model used in the static simulator:
each hour-long decision step, the agent picks a TP/PP strategy. Strategy
switches pay a reshard cost (idle power for ~14s, plus a one-time cache-miss
bump on the first switch). Energy and carbon are integrated over the real
hour boundaries of the carbon-intensity trace, so the simulator stays accurate
for long episodes.

The simulation core (`CarbonSim`) is pure-numpy and runs without `gymnasium`,
so baselines (static, CI-threshold) and unit tests can use it directly. The
`CarbonAwareTrainingEnv` class is a thin Gym wrapper on top.

Reward is `-Δcarbon_g` over the step, so the cumulative return of an episode
equals `-total_carbon_g`. RL agents that maximize return therefore minimize
gCO₂.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


class TraceOverrunError(RuntimeError):
    """Raised when a sim looks past the end of a non-wrapping carbon trace."""


# ---------------------------------------------------------------------------
# Carbon-trace loader
# ---------------------------------------------------------------------------

def load_carbon_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Load a carbon-intensity CSV with year/month/day/time/carbon columns.

    Returns `(t_seconds_from_first_row, carbon_g_per_kwh, datetimes)`. Rows are
    sorted and must be hourly.
    """
    df = pd.read_csv(path)
    required = {'year', 'month', 'day', 'time', 'carbon'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Carbon trace missing columns: {sorted(missing)}')

    datetimes = pd.to_datetime(
        df['year'].astype(str) + '-' +
        df['month'].astype(str).str.zfill(2) + '-' +
        df['day'].astype(str).str.zfill(2) + ' ' +
        df['time'].astype(str)
    )
    df = df.assign(datetime=datetimes).sort_values('datetime').reset_index(drop=True)

    t = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    c = df['carbon'].to_numpy(dtype=float)

    if len(t) >= 2:
        dt = np.diff(t)
        if not np.allclose(dt, 3600.0):
            raise ValueError('Carbon trace must be hourly')

    return t, c, df['datetime']


# ---------------------------------------------------------------------------
# Simulation core (no gymnasium dependency)
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All knobs the simulator needs. Match this to your `profile_strategies.py`."""
    strategies: list[dict]
    pretraining_tokens: int
    tokens_per_iter: int
    n_gpus: int = 4
    idle_power_per_gpu_w: float = 50.0
    container_startup_s: float = 35.0
    fresh_build_s: float = 0.23
    reshard_time_s: float = 13.68
    first_reshard_cache_miss_s: float = 20.0

    @property
    def idle_cluster_w(self) -> float:
        return self.idle_power_per_gpu_w * self.n_gpus


@dataclass
class StepRecord:
    """One row of the per-decision-step history (for plotting / debugging)."""
    step: int
    t_start_s: float
    t_end_s: float
    strategy_idx: int
    reshard: bool
    tokens_completed: float
    energy_j: float        # cumulative
    carbon_g: float        # cumulative
    ci_at_step_start: float


class CarbonSim:
    """Pure-Python simulator. Reused by the Gym env, baselines, and tests.

    Args:
        wrap_trace: If True, time is taken modulo trace duration, so a finite
            trace tiles cyclically into an effectively infinite one. Useful for
            training when token budgets exceed the real trace length. Default
            False, which raises if a lookup goes past the trace end.
    """

    def __init__(self, cfg: SimConfig, trace_t: np.ndarray, trace_ci: np.ndarray,
                 wrap_trace: bool = False):
        self.cfg = cfg
        self.trace_t = trace_t
        self.trace_ci = trace_ci
        self.trace_duration_s = float(trace_t[-1] + 3600.0)  # last hour is covered
        self.wrap_trace = bool(wrap_trace)

        # Validate strategies have the fields we need
        required = {'tp', 'pp', 'power_w', 'step_time_s',
                    'warmup_wall_s', 'warmup_iters', 'save_s'}
        for s in cfg.strategies:
            missing = required - set(s.keys())
            if missing:
                raise ValueError(f'Strategy missing fields {missing}: {s}')

        self.reset(trace_offset_s=0.0)

    # ---- carbon-trace lookup ------------------------------------------------

    def ci_at(self, trace_time_s: float) -> float:
        """Carbon intensity at an absolute trace time (seconds from trace[0]).

        If `wrap_trace=True`, time wraps modulo trace duration. Otherwise,
        going past the trace end raises a `TraceOverrunError` (so silent
        clamping never produces a misleading constant-CI episode).
        """
        if self.wrap_trace:
            trace_time_s = trace_time_s % self.trace_duration_s
        if trace_time_s < self.trace_t[0]:
            return float(self.trace_ci[0])
        if trace_time_s >= self.trace_t[-1] + 3600.0:
            raise TraceOverrunError(
                f'Lookup at trace_time={trace_time_s:.1f}s '
                f'({trace_time_s/3600.0:.2f}h) exceeds trace duration '
                f'({self.trace_duration_s/3600.0:.2f}h). '
                f'Either shorten the episode, use a longer trace, or pass '
                f'wrap_trace=True to tile cyclically.'
            )
        idx = np.searchsorted(self.trace_t, trace_time_s, side='right') - 1
        return float(self.trace_ci[int(np.clip(idx, 0, len(self.trace_ci) - 1))])

    def ci_now(self) -> float:
        return self.ci_at(self.trace_offset_s + self.cursor_s)

    def ci_forecast(self, horizon_h: int) -> np.ndarray:
        """Hourly CI values for the next `horizon_h` hours starting from now."""
        out = np.empty(horizon_h, dtype=float)
        for h in range(horizon_h):
            out[h] = self.ci_at(self.trace_offset_s + self.cursor_s + (h + 1) * 3600.0)
        return out

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, trace_offset_s: float, initial_strategy_idx: int = 0):
        self.trace_offset_s = float(trace_offset_s)
        self.cursor_s = 0.0
        self.tokens_completed = 0.0
        self.total_energy_j = 0.0
        self.total_carbon_g = 0.0
        self.current_strategy_idx = int(initial_strategy_idx)
        self.reshard_count = 0
        # After init, the first chosen strategy needs warmup. We treat the
        # container-startup phase as a "warmup wipe" so the very first decision
        # also pays the warmup cost — that matches what the static sim does.
        self.warmup_pending = True
        self.history: list[StepRecord] = []
        self.step_idx = 0

        # Charge container startup + fresh build (idle power)
        self._integrate_idle(self.cfg.container_startup_s + self.cfg.fresh_build_s)

    def done(self) -> bool:
        return self.tokens_completed >= self.cfg.pretraining_tokens - 1e-6

    # ---- integration helpers ----------------------------------------------

    def _next_hour_boundary_local(self, t_local: float) -> float:
        """Local time of the next trace-hour boundary strictly after t_local."""
        trace_pos = self.trace_offset_s + t_local
        # floor → next multiple of 3600 strictly greater than trace_pos
        next_trace_hour = (math.floor(trace_pos / 3600.0) + 1.0) * 3600.0
        if next_trace_hour - trace_pos < 1e-9:
            next_trace_hour += 3600.0
        return next_trace_hour - self.trace_offset_s

    def _integrate_constant_power(self, duration_s: float, power_w: float,
                                   throughput_tok_s: float = 0.0,
                                   token_cap: float | None = None) -> float:
        """Advance time by up to `duration_s` at constant power and throughput.

        Splits at hour boundaries to integrate carbon accurately across CI
        transitions. Stops early if `token_cap` would be exceeded.

        Returns the duration actually consumed.
        """
        if duration_s <= 1e-9:
            return 0.0

        target_end = self.cursor_s + duration_s
        consumed = 0.0

        while self.cursor_s < target_end - 1e-9:
            hour_end = self._next_hour_boundary_local(self.cursor_s)
            chunk_end = min(target_end, hour_end)
            chunk_dt = chunk_end - self.cursor_s

            # Cap by token budget if applicable
            if token_cap is not None and throughput_tok_s > 0:
                remaining = token_cap - self.tokens_completed
                if remaining <= 0:
                    break
                time_to_cap = remaining / throughput_tok_s
                chunk_dt = min(chunk_dt, time_to_cap)

            if chunk_dt <= 1e-9:
                self.cursor_s = chunk_end
                continue

            # CI at midpoint of this chunk (chunk is within a single trace hour
            # so CI is in fact constant; midpoint is exact).
            ci = self.ci_at(self.trace_offset_s + self.cursor_s + chunk_dt / 2.0)
            e_j = power_w * chunk_dt
            c_g = e_j * ci / 3.6e6

            self.cursor_s += chunk_dt
            self.tokens_completed += throughput_tok_s * chunk_dt
            self.total_energy_j += e_j
            self.total_carbon_g += c_g
            consumed += chunk_dt

        return consumed

    def _integrate_idle(self, duration_s: float):
        self._integrate_constant_power(duration_s, self.cfg.idle_cluster_w)

    # ---- one decision step ------------------------------------------------

    def step(self, action: int, decision_interval_s: float = 3600.0) -> dict:
        """Run one decision step.

        - If `action` differs from current strategy, pay reshard cost.
        - Run training (warmup phase first if pending) for up to
          `decision_interval_s`, stopping if the token budget is hit.
        - On termination, charge the chosen strategy's `save_s`.

        Returns a dict with `delta_carbon_g`, `delta_energy_j`, `terminated`,
        `truncated_local`, plus a `StepRecord` snapshot.
        """
        cfg = self.cfg
        t_start = self.cursor_s
        ci_start = self.ci_now()
        carbon_start = self.total_carbon_g
        energy_start = self.total_energy_j

        # Are we resharding?
        reshard = action != self.current_strategy_idx
        if reshard:
            cost = cfg.reshard_time_s
            if self.reshard_count == 0:
                cost += cfg.first_reshard_cache_miss_s
            self._integrate_idle(cost)
            self.reshard_count += 1
            self.current_strategy_idx = action
            self.warmup_pending = True

        strategy = cfg.strategies[action]

        # Hour budget for this decision step. Reshard time counts against it
        # so the agent is incentivized to plan ahead.
        budget_remaining = max(0.0, decision_interval_s - (self.cursor_s - t_start))

        # Warmup phase
        if self.warmup_pending and budget_remaining > 1e-9 and not self.done():
            warmup_throughput = (
                strategy['warmup_iters'] * cfg.tokens_per_iter / strategy['warmup_wall_s']
                if strategy['warmup_wall_s'] > 0 else 0.0
            )
            warmup_dt = min(strategy['warmup_wall_s'], budget_remaining)
            used = self._integrate_constant_power(
                warmup_dt,
                strategy['power_w'],
                throughput_tok_s=warmup_throughput,
                token_cap=cfg.pretraining_tokens,
            )
            budget_remaining -= used
            if used >= strategy['warmup_wall_s'] - 1e-6:
                self.warmup_pending = False

        # Steady-state phase
        if budget_remaining > 1e-9 and not self.warmup_pending and not self.done():
            throughput = (cfg.tokens_per_iter / strategy['step_time_s']
                          if strategy['step_time_s'] > 0 else 0.0)
            self._integrate_constant_power(
                budget_remaining,
                strategy['power_w'],
                throughput_tok_s=throughput,
                token_cap=cfg.pretraining_tokens,
            )

        terminated = self.done()
        if terminated:
            # Final save (idle power for save_s of the strategy we ended on).
            self._integrate_idle(strategy['save_s'])

        delta_carbon = self.total_carbon_g - carbon_start
        delta_energy = self.total_energy_j - energy_start

        record = StepRecord(
            step=self.step_idx,
            t_start_s=t_start,
            t_end_s=self.cursor_s,
            strategy_idx=self.current_strategy_idx,
            reshard=reshard,
            tokens_completed=self.tokens_completed,
            energy_j=self.total_energy_j,
            carbon_g=self.total_carbon_g,
            ci_at_step_start=ci_start,
        )
        self.history.append(record)
        self.step_idx += 1

        return {
            'delta_carbon_g': delta_carbon,
            'delta_energy_j': delta_energy,
            'terminated': terminated,
            'record': record,
        }


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = None  # type: ignore
    spaces = None  # type: ignore


if _HAS_GYM:

    class CarbonAwareTrainingEnv(gym.Env):
        """Pick a TP/PP strategy each hour to minimize gCO₂ for a token budget.

        Observation (Box, all roughly in [0, 1]):
            [ ci_now / ci_max,
              ci_forecast[+1h] / ci_max, ..., ci_forecast[+Hh] / ci_max,
              one_hot(current_strategy_idx),
              tokens_completed / pretraining_tokens,
              hours_elapsed / max_episode_hours ]

        Action (Discrete(n_strategies)): which strategy to run for the next hour.

        Reward: -Δcarbon_g for this step (so episode return == -total gCO₂).

        Episode termination:
            terminated -> token budget reached
            truncated  -> max_episode_hours exceeded (with optional penalty
                          per un-trained token, off by default)
        """

        metadata = {'render_modes': []}

        def __init__(self,
                     sim_cfg: SimConfig,
                     trace_t: np.ndarray,
                     trace_ci: np.ndarray,
                     forecast_horizon_h: int = 6,
                     decision_interval_s: float = 3600.0,
                     max_episode_hours: float = 72.0,
                     random_start: bool = True,
                     truncation_penalty_per_remaining_token: float = 0.0,
                     ci_max: float | None = None,
                     wrap_trace: bool = False,
                     seed: int | None = None):
            super().__init__()
            self.sim_cfg = sim_cfg
            self.trace_t = trace_t
            self.trace_ci = trace_ci
            self.forecast_horizon_h = int(forecast_horizon_h)
            self.decision_interval_s = float(decision_interval_s)
            self.max_episode_hours = float(max_episode_hours)
            self.random_start = bool(random_start)
            self.truncation_penalty = float(truncation_penalty_per_remaining_token)
            self.ci_max = float(ci_max) if ci_max is not None else float(trace_ci.max())
            self.wrap_trace = bool(wrap_trace)
            self._rng = np.random.default_rng(seed)

            self.n_strategies = len(sim_cfg.strategies)
            self.action_space = spaces.Discrete(self.n_strategies)
            obs_dim = 1 + self.forecast_horizon_h + self.n_strategies + 2
            self.observation_space = spaces.Box(
                low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.sim: CarbonSim | None = None
            self._trace_offset_s = 0.0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            options = options or {}
            trace_dur = self.trace_t[-1] + 3600.0
            if 'trace_offset_s' in options:
                offset = float(options['trace_offset_s'])
            elif self.random_start:
                if self.wrap_trace:
                    # With wrapping, any offset in the trace is fine
                    max_offset = trace_dur
                else:
                    # Leave room for the longest possible episode
                    slack = self.max_episode_hours * 3600.0 + 3600.0
                    max_offset = max(0.0, trace_dur - slack)
                    if max_offset < 3600.0:
                        raise ValueError(
                            f'Trace duration ({trace_dur/3600.0:.1f}h) is too '
                            f'short for max_episode_hours={self.max_episode_hours}. '
                            f'Either shorten the episode, extend the trace, or '
                            f'pass wrap_trace=True.'
                        )
                offset = float(self._rng.uniform(0.0, max_offset))
            else:
                offset = 0.0
            self._trace_offset_s = offset

            self.sim = CarbonSim(self.sim_cfg, self.trace_t, self.trace_ci,
                                 wrap_trace=self.wrap_trace)
            self.sim.reset(trace_offset_s=offset, initial_strategy_idx=0)
            return self._make_obs(), {'trace_offset_s': offset}

        def step(self, action):
            assert self.sim is not None, 'Must call reset() before step()'
            result = self.sim.step(int(action), self.decision_interval_s)
            reward = -result['delta_carbon_g']

            terminated = result['terminated']
            hours_elapsed = self.sim.cursor_s / 3600.0
            truncated = (not terminated) and (hours_elapsed >= self.max_episode_hours)
            if truncated and self.truncation_penalty > 0:
                remaining = max(
                    0.0,
                    self.sim_cfg.pretraining_tokens - self.sim.tokens_completed,
                )
                reward -= self.truncation_penalty * remaining

            obs = self._make_obs()
            info = {
                'tokens_completed': self.sim.tokens_completed,
                'total_carbon_g': self.sim.total_carbon_g,
                'total_energy_j': self.sim.total_energy_j,
                'cursor_s': self.sim.cursor_s,
                'reshard_count': self.sim.reshard_count,
                'progress': self.sim.tokens_completed / self.sim_cfg.pretraining_tokens,
            }
            return obs, float(reward), bool(terminated), bool(truncated), info

        def _make_obs(self) -> np.ndarray:
            sim = self.sim
            ci_now = sim.ci_now() / self.ci_max
            forecast = sim.ci_forecast(self.forecast_horizon_h) / self.ci_max
            onehot = np.zeros(self.n_strategies, dtype=np.float32)
            onehot[sim.current_strategy_idx] = 1.0
            progress = sim.tokens_completed / self.sim_cfg.pretraining_tokens
            time_used = (sim.cursor_s / 3600.0) / self.max_episode_hours
            return np.concatenate([
                np.array([ci_now], dtype=np.float32),
                forecast.astype(np.float32),
                onehot,
                np.array([progress, time_used], dtype=np.float32),
            ])

else:
    class CarbonAwareTrainingEnv:  # type: ignore[no-redef]
        """Placeholder when gymnasium isn't installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                'gymnasium is required for CarbonAwareTrainingEnv. '
                'Install it with: pip install gymnasium'
            )


def make_env(sim_cfg: SimConfig,
             trace_t: np.ndarray,
             trace_ci: np.ndarray,
             **env_kwargs):
    """Build a CarbonAwareTrainingEnv. Raises ImportError if gymnasium absent."""
    return CarbonAwareTrainingEnv(sim_cfg, trace_t, trace_ci, **env_kwargs)


# ---------------------------------------------------------------------------
# Baseline policies (work on top of CarbonSim directly — no gym needed)
# ---------------------------------------------------------------------------

def rollout_static(sim_cfg: SimConfig,
                   trace_t: np.ndarray,
                   trace_ci: np.ndarray,
                   trace_offset_s: float,
                   strategy_idx: int,
                   decision_interval_s: float = 3600.0,
                   max_hours: float = 200.0,
                   wrap_trace: bool = False) -> CarbonSim:
    """Run a static-strategy episode end-to-end. Returns the finished sim."""
    sim = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    sim.reset(trace_offset_s, initial_strategy_idx=strategy_idx)
    max_steps = int(max_hours * 3600.0 / decision_interval_s) + 2
    for _ in range(max_steps):
        r = sim.step(strategy_idx, decision_interval_s)
        if r['terminated']:
            break
    return sim


def rollout_ci_threshold(sim_cfg: SimConfig,
                         trace_t: np.ndarray,
                         trace_ci: np.ndarray,
                         trace_offset_s: float,
                         low_ci_idx: int,
                         high_ci_idx: int,
                         threshold_g_per_kwh: float,
                         decision_interval_s: float = 3600.0,
                         max_hours: float = 200.0,
                         wrap_trace: bool = False) -> CarbonSim:
    """CI-threshold baseline: pick high_ci_idx when CI > threshold, else low."""
    sim = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    initial = high_ci_idx if sim.ci_at(trace_offset_s) > threshold_g_per_kwh else low_ci_idx
    sim.reset(trace_offset_s, initial_strategy_idx=initial)
    max_steps = int(max_hours * 3600.0 / decision_interval_s) + 2
    for _ in range(max_steps):
        ci = sim.ci_now()
        action = high_ci_idx if ci > threshold_g_per_kwh else low_ci_idx
        r = sim.step(action, decision_interval_s)
        if r['terminated']:
            break
    return sim


def rollout_policy(env, policy_fn: Callable[[np.ndarray], int]) -> dict:
    """Run one episode in `env` using a callable `obs -> action`. Returns
    final info dict augmented with `history` (list of StepRecord) for plotting.
    """
    obs, info = env.reset()
    done = False
    while not done:
        action = int(policy_fn(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    info['history'] = list(env.sim.history)
    info['trace_offset_s'] = env._trace_offset_s
    info['total_carbon_g'] = env.sim.total_carbon_g
    info['total_energy_j'] = env.sim.total_energy_j
    info['wall_s'] = env.sim.cursor_s
    info['tokens_completed'] = env.sim.tokens_completed
    info['reshard_count'] = env.sim.reshard_count
    return info


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_rollout(sim: CarbonSim,
                 sim_cfg: SimConfig,
                 title: str = 'Rollout',
                 ax_ci=None, ax_strategy=None, ax_carbon=None,
                 color: str = 'crimson'):
    """Three stacked axes: CI(t), strategy choice(t), cumulative carbon(t).

    Pass pre-made axes to overlay multiple rollouts on the same figure.
    """
    import matplotlib.pyplot as plt
    own_fig = ax_ci is None
    if own_fig:
        fig, (ax_ci, ax_strategy, ax_carbon) = plt.subplots(
            3, 1, figsize=(13, 10), sharex=True,
            gridspec_kw={'height_ratios': [1, 1, 2]}
        )

    end_h = sim.cursor_s / 3600.0
    sample_t = np.linspace(0.0, sim.cursor_s, max(200, int(end_h * 4)))
    sample_ci = np.array([sim.ci_at(sim.trace_offset_s + t) for t in sample_t])
    ax_ci.plot(sample_t / 3600.0, sample_ci, color='darkgreen',
               drawstyle='steps-post', linewidth=1.5)
    ax_ci.fill_between(sample_t / 3600.0, sample_ci, alpha=0.15,
                       color='darkgreen', step='post')
    ax_ci.set_ylabel('CI (g/kWh)')
    ax_ci.grid(True, alpha=0.3)
    ax_ci.set_ylim(bottom=0)

    if sim.history:
        for rec in sim.history:
            ax_strategy.plot([rec.t_start_s / 3600.0, rec.t_end_s / 3600.0],
                             [rec.strategy_idx, rec.strategy_idx],
                             color=color, linewidth=4, solid_capstyle='butt')
            if rec.reshard:
                ax_strategy.axvspan(rec.t_start_s / 3600.0,
                                    rec.t_start_s / 3600.0 + 0.02,
                                    color=color, alpha=0.25)
        ax_strategy.set_yticks(range(len(sim_cfg.strategies)))
        ax_strategy.set_yticklabels(
            [f"({s['tp']},{s['pp']})" for s in sim_cfg.strategies]
        )
    ax_strategy.set_ylabel('Strategy')
    ax_strategy.grid(True, alpha=0.3)

    # Cumulative carbon — pull from history (per-step end values)
    t_hist = np.array([rec.t_end_s / 3600.0 for rec in sim.history])
    c_hist = np.array([rec.carbon_g for rec in sim.history])
    if len(t_hist):
        t_hist = np.concatenate([[0.0], t_hist])
        c_hist = np.concatenate([[0.0], c_hist])
        ax_carbon.plot(t_hist, c_hist, color=color, linewidth=2.2,
                       label=f'{title}: {sim.total_carbon_g:.1f} gCO₂, '
                             f'{sim.cursor_s/3600.0:.2f}h, '
                             f'{sim.reshard_count} reshards')
        ax_carbon.plot(t_hist[-1], c_hist[-1], 'o', color=color, markersize=7)
    ax_carbon.set_xlabel('Hours from episode start')
    ax_carbon.set_ylabel('Cumulative gCO₂')
    ax_carbon.grid(True, alpha=0.3)
    ax_carbon.legend(loc='upper left', fontsize=9)

    if own_fig:
        fig.suptitle(title)
        import matplotlib.pyplot as plt
        plt.tight_layout()
        return fig
    return None


def plot_comparison(rollouts: dict[str, CarbonSim],
                    sim_cfg: SimConfig,
                    title: str = 'RL vs baselines'):
    """Overlay multiple rollouts on one figure. `rollouts` = {name: CarbonSim}."""
    import matplotlib.pyplot as plt
    fig, (ax_ci, ax_strategy, ax_carbon) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.2, 2]}
    )

    # CI panel uses the longest rollout's window
    longest = max(rollouts.values(), key=lambda s: s.cursor_s)
    sample_t = np.linspace(0.0, longest.cursor_s, 1500)
    sample_ci = np.array([longest.ci_at(longest.trace_offset_s + t) for t in sample_t])
    ax_ci.plot(sample_t / 3600.0, sample_ci, color='darkgreen',
               drawstyle='steps-post', linewidth=1.5)
    ax_ci.fill_between(sample_t / 3600.0, sample_ci, alpha=0.15,
                       color='darkgreen', step='post')
    ax_ci.set_ylabel('CI (g/kWh)')
    ax_ci.set_title(title)
    ax_ci.grid(True, alpha=0.3)
    ax_ci.set_ylim(bottom=0)

    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(rollouts), 1)))
    n_strats = len(sim_cfg.strategies)
    # Stagger strategy lines vertically per-rollout so they don't overlap
    n_roll = len(rollouts)
    for i, ((name, sim), color) in enumerate(zip(rollouts.items(), cmap)):
        offset = (i - (n_roll - 1) / 2) * 0.18
        for rec in sim.history:
            ax_strategy.plot([rec.t_start_s / 3600.0, rec.t_end_s / 3600.0],
                             [rec.strategy_idx + offset, rec.strategy_idx + offset],
                             color=color, linewidth=3, solid_capstyle='butt')
        t_hist = np.array([0.0] + [rec.t_end_s / 3600.0 for rec in sim.history])
        c_hist = np.array([0.0] + [rec.carbon_g for rec in sim.history])
        ax_carbon.plot(t_hist, c_hist, color=color, linewidth=2.2,
                       label=f'{name}: {sim.total_carbon_g:.1f} gCO₂  '
                             f'({sim.cursor_s/3600.0:.2f}h, '
                             f'{sim.reshard_count} reshards)')
        ax_carbon.plot(t_hist[-1], c_hist[-1], 'o', color=color, markersize=7)

    ax_strategy.set_yticks(range(n_strats))
    ax_strategy.set_yticklabels([f"({s['tp']},{s['pp']})" for s in sim_cfg.strategies])
    ax_strategy.set_ylabel('Strategy')
    ax_strategy.grid(True, alpha=0.3)

    ax_carbon.set_xlabel('Hours from episode start')
    ax_carbon.set_ylabel('Cumulative gCO₂')
    ax_carbon.legend(loc='upper left', fontsize=9)
    ax_carbon.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_training_curve(returns: list[float], window: int = 20,
                        title: str = 'Training reward (negative gCO₂)'):
    """Plot per-episode return with a rolling mean. Returns are -gCO₂."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = np.arange(len(returns))
    ax.plot(ep, returns, color='steelblue', alpha=0.3, linewidth=1,
            label='per-episode return')
    if len(returns) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(returns, kernel, mode='valid')
        ax.plot(np.arange(window - 1, len(returns)), smooth,
                color='crimson', linewidth=2.2,
                label=f'{window}-episode rolling mean')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return (= -gCO₂)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Default config matching the user's notebook
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIES = [
    {'tp': 1, 'pp': 1, 'power_w': 802.0, 'step_time_s': 0.1875,
     'warmup_wall_s': 10.1, 'warmup_iters': 37, 'save_s': 11.00},
    {'tp': 2, 'pp': 1, 'power_w': 612.0, 'step_time_s': 0.4158,
     'warmup_wall_s': 10.6, 'warmup_iters': 21, 'save_s':  8.80},
    {'tp': 4, 'pp': 1, 'power_w': 598.0, 'step_time_s': 0.8163,
     'warmup_wall_s': 18.9, 'warmup_iters': 21, 'save_s':  7.82},
]


def default_config(pretraining_tokens: int = 3_000_000_000,
                   global_batch_size: int = 16,
                   seq_length: int = 1024) -> SimConfig:
    return SimConfig(
        strategies=[dict(s) for s in DEFAULT_STRATEGIES],
        pretraining_tokens=pretraining_tokens,
        tokens_per_iter=global_batch_size * seq_length,
    )
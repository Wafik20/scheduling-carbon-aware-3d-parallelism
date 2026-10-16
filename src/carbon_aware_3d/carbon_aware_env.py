"""Carbon-aware training-strategy selection as a Gymnasium environment.

The environment wraps an synthetic demonstration strategy profile by default: each hour-long
decision step, the agent picks a TP/PP strategy. Strategy switches pay the
target strategy's measured reshard cost when available. Energy and carbon are
integrated over the real hour boundaries of the carbon-intensity trace, so the
simulator stays accurate for long episodes.

The simulation core (`CarbonSim`) is pure-numpy and runs without `gymnasium`,
so baselines (static, smart switching) and unit tests can use it directly. The
`CarbonAwareTrainingEnv` class is a thin Gym wrapper on top.

Reward is `-Δcarbon_g` over the step, so the cumulative return of an episode
equals `-total_carbon_g`. RL agents that maximize return therefore minimize
gCO₂.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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
    n_gpus: int = 8
    idle_power_per_gpu_w: float = 50.0
    container_startup_s: float = 0.0
    fresh_build_s: float = 0.0
    switch_time_s: float = 0.0
    first_switch_cache_miss_s: float = 0.0

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


def strategy_label(strategy: dict, *, include_dp: bool = True) -> str:
    """Compact display label for TP/PP strategies, including MBS when present."""
    parts = [
        f"tp={int(strategy['tp'])}",
        f"pp={int(strategy['pp'])}",
    ]
    if include_dp and 'dp' in strategy:
        parts.append(f"dp={int(strategy['dp'])}")
    if 'micro_batch_size' in strategy:
        parts.append(f"mbs={int(strategy['micro_batch_size'])}")
    return "(" + ",".join(parts) + ")"


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
        required = {
            'tp', 'pp', 'power_w', 'throughput_tok_s',
            'warmup_wall_s', 'warmup_iters', 'save_s',
        }
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
        plus a `StepRecord` snapshot.
        """
        cfg = self.cfg
        t_start = self.cursor_s
        ci_start = self.ci_now()
        carbon_start = self.total_carbon_g
        energy_start = self.total_energy_j

        strategy = cfg.strategies[action]

        # Are we switching configs?
        reshard = action != self.current_strategy_idx
        if reshard:
            cost = strategy.get('reshard_wall_time_s', cfg.switch_time_s)
            if self.reshard_count == 0:
                cost += cfg.first_switch_cache_miss_s
            self._integrate_idle(cost)
            self.reshard_count += 1
            self.current_strategy_idx = action
            self.warmup_pending = True

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
            throughput = float(strategy['throughput_tok_s'])
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
                    # Leave room for the longest possible episode, one final
                    # decision-step overshoot, and the next observation's CI
                    # forecast horizon.
                    slack = (
                        self.max_episode_hours * 3600.0
                        + self.decision_interval_s
                        + self.forecast_horizon_h * 3600.0
                    )
                    max_offset = max(0.0, trace_dur - slack)
                    if max_offset < 3600.0:
                        raise ValueError(
                            f'Trace duration ({trace_dur/3600.0:.1f}h) is too '
                            f'short for max_episode_hours={self.max_episode_hours} '
                            f'and forecast_horizon_h={self.forecast_horizon_h}. '
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
            tokens_before = self.sim.tokens_completed
            result = self.sim.step(int(action), self.decision_interval_s)
            delta_tokens = self.sim.tokens_completed - tokens_before
            delta_carbon = result['delta_carbon_g']

            reward = -delta_carbon
            terminated = result['terminated']
            hours_elapsed = self.sim.cursor_s / 3600.0
            truncated = (not terminated) and (hours_elapsed >= self.max_episode_hours)
            if truncated and self.truncation_penalty > 0:
                remaining = max(
                    0.0,
                    self.sim_cfg.pretraining_tokens - self.sim.tokens_completed,
                )
                reward -= self.truncation_penalty * remaining

            obs = self._make_obs(pad_forecast=terminated or truncated)
            info = {
                'tokens_completed': self.sim.tokens_completed,
                'total_carbon_g': self.sim.total_carbon_g,
                'total_energy_j': self.sim.total_energy_j,
                'cursor_s': self.sim.cursor_s,
                'reshard_count': self.sim.reshard_count,
                'progress': self.sim.tokens_completed / self.sim_cfg.pretraining_tokens,
                'delta_tokens': delta_tokens,
                'delta_carbon_g': delta_carbon,
            }
            return obs, float(reward), bool(terminated), bool(truncated), info

        def _make_obs(self, pad_forecast: bool = False) -> np.ndarray:
            sim = self.sim
            try:
                ci_now = sim.ci_now() / self.ci_max
            except TraceOverrunError:
                if not pad_forecast:
                    raise
                ci_now = float(self.trace_ci[-1]) / self.ci_max
            if pad_forecast:
                try:
                    forecast = sim.ci_forecast(self.forecast_horizon_h) / self.ci_max
                except TraceOverrunError:
                    forecast = np.full(
                        self.forecast_horizon_h,
                        ci_now,
                        dtype=np.float32,
                    )
            else:
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

def _replay_action_sequence(sim_cfg: SimConfig,
                            trace_t: np.ndarray,
                            trace_ci: np.ndarray,
                            trace_offset_s: float,
                            action_seq: Sequence[int],
                            initial_strategy_idx: int | None = None,
                            decision_interval_s: float = 3600.0,
                            max_hours: float = 200.0,
                            wrap_trace: bool = False) -> CarbonSim:
    """Replay a fixed action sequence through CarbonSim exact accounting.

    By default the first action is treated as the initial strategy, so it does
    not pay a reshard cost. This matches the DP oracle's free
    initial-configuration convention. Pass `initial_strategy_idx` to replay
    policies, such as smart-switch, that intentionally start from a different
    initial strategy.
    """
    if not action_seq:
        raise ValueError('action_seq must contain at least one action')

    sim = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    if initial_strategy_idx is None:
        initial_strategy_idx = int(action_seq[0])
    sim.reset(trace_offset_s, initial_strategy_idx=int(initial_strategy_idx))
    max_steps = int(max_hours * 3600.0 / decision_interval_s) + 2
    for action in list(action_seq)[:max_steps]:
        r = sim.step(int(action), decision_interval_s)
        if r['terminated']:
            break
    return sim


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


def rollout_job_shifting_static_fast(
    sim_cfg: SimConfig,
    trace_t: np.ndarray,
    trace_ci: np.ndarray,
    arrival_offset_s: float,
    shift_window_h: float,
    shift_step_h: float,
    max_hours: float,
    wrap_trace: bool = False,
    respect_deadline: bool = False,
) -> tuple[CarbonSim, float, int]:
    """Delay the job start, then run the highest-throughput static strategy.

    The waiting period is intentionally not simulated as cluster activity:
    returned carbon and energy include only the training run. The selected
    delay is returned separately so evaluators can report completion delay. If
    `respect_deadline` is true, `max_hours` is treated as an arrival-relative
    deadline and each delayed start receives only its remaining horizon.
    """
    if shift_window_h < 0:
        raise ValueError(f'shift_window_h must be non-negative, got {shift_window_h}')
    if shift_step_h <= 0:
        raise ValueError(f'shift_step_h must be positive, got {shift_step_h}')

    fastest_idx = fastest_strategy_idx(sim_cfg)

    best_sim = None
    best_delay_s = None
    best_carbon = float('inf')

    shift_window_s = shift_window_h * 3600.0
    shift_step_s = shift_step_h * 3600.0
    n_steps = int(shift_window_s // shift_step_s)

    for j in range(n_steps + 1):
        delay_s = j * shift_step_s
        start_offset_s = arrival_offset_s + delay_s
        horizon_h = max_hours
        if respect_deadline:
            horizon_h = max_hours - delay_s / 3600.0
            if horizon_h <= 0:
                continue

        try:
            sim = rollout_static(
                sim_cfg,
                trace_t,
                trace_ci,
                start_offset_s,
                strategy_idx=fastest_idx,
                max_hours=horizon_h,
                wrap_trace=wrap_trace,
            )
        except TraceOverrunError:
            continue

        if not sim.done():
            continue
        if respect_deadline and delay_s + sim.cursor_s > max_hours * 3600.0 + 1e-6:
            continue

        if sim.total_carbon_g < best_carbon:
            best_carbon = sim.total_carbon_g
            best_sim = sim
            best_delay_s = delay_s

    if best_sim is None:
        best_delay_s = 0.0
        best_sim = rollout_static(
            sim_cfg,
            trace_t,
            trace_ci,
            arrival_offset_s,
            strategy_idx=fastest_idx,
            max_hours=max_hours,
            wrap_trace=wrap_trace,
        )

    return best_sim, float(best_delay_s), fastest_idx


def _future_clean_ci(sim: CarbonSim,
                     lookahead_h: int,
                     catchup_time_s: float) -> float:
    """Estimate the clean CI available for catch-up.

    For now, this is the minimum hourly CI in the next `lookahead_h` hours.
    `catchup_time_s` is accepted for future refinements that model the length
    of the catch-up window.
    """
    del catchup_time_s
    vals = []
    for h in range(1, lookahead_h + 1):
        vals.append(sim.ci_at(sim.trace_offset_s + sim.cursor_s + h * 3600.0))
    return min(vals)


def _switch_cost_g(sim: CarbonSim,
                   target_strategy_idx: int,
                   ci_g_per_kwh: float) -> float:
    """Carbon cost of switching into `target_strategy_idx` at the given CI."""
    if target_strategy_idx == sim.current_strategy_idx:
        return 0.0
    strategy = sim.cfg.strategies[target_strategy_idx]
    switch_s = strategy.get('reshard_wall_time_s', sim.cfg.switch_time_s)
    return sim.cfg.idle_cluster_w * switch_s * ci_g_per_kwh / 3.6e6


def fastest_strategy_idx(sim_cfg: SimConfig) -> int:
    """Return the strategy index with highest measured throughput."""
    return max(
        range(len(sim_cfg.strategies)),
        key=lambda i: float(sim_cfg.strategies[i]['throughput_tok_s']),
    )


def _fastest_strategy_idx(sim_cfg: SimConfig) -> int:
    """Return the strategy index with highest measured throughput.

    Tie-break by lower J/token, then lower power.
    """
    best_idx = 0
    best_key = None

    for i, s in enumerate(sim_cfg.strategies):
        throughput = float(s['throughput_tok_s'])
        power = float(s['power_w'])
        j_per_tok = power / throughput

        # Maximize throughput. If tied, prefer lower J/token, then lower power.
        key = (throughput, -j_per_tok, -power)

        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    return best_idx


def rollout_smart_switch(sim_cfg: SimConfig,
                         trace_t: np.ndarray,
                         trace_ci: np.ndarray,
                         trace_offset_s: float,
                         fast_idx: int | None = None,
                         candidate_indices: list[int] | None = None,
                         lookahead_h: int = 6,
                         decision_interval_s: float = 3600.0,
                         max_hours: float = 200.0,
                         margin_g: float = 0.0,
                         wrap_trace: bool = False) -> CarbonSim:
    """Carbon-arbitrage baseline using measured power, throughput, and reshard.

    Run the highest-throughput clean-window catch-up mode by default. During
    dirty periods, switch to a lower-power, lower-throughput strategy only when
    estimated dirty-hour savings exceed clean catch-up carbon, switching
    overhead, and `margin_g`.
    """
    if fast_idx is None or fast_idx < 0:
        fast_idx = _fastest_strategy_idx(sim_cfg)

    sim = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    sim.reset(trace_offset_s, initial_strategy_idx=fast_idx)

    fast = sim_cfg.strategies[fast_idx]
    p_a = float(fast['power_w'])
    r_a = float(fast['throughput_tok_s'])

    if candidate_indices is None:
        candidate_indices = [
            i for i, s in enumerate(sim_cfg.strategies)
            if (float(s['power_w']) < p_a
                and float(s['throughput_tok_s']) < r_a)
        ]

    max_steps = int(max_hours * 3600.0 / decision_interval_s) + 2
    for _ in range(max_steps):
        ci_now = sim.ci_now()
        best_action = fast_idx
        best_benefit_g = 0.0

        for i in candidate_indices:
            candidate = sim_cfg.strategies[i]
            p_b = float(candidate['power_w'])
            r_b = float(candidate['throughput_tok_s'])
            tau = decision_interval_s

            catchup_time_s = (1.0 - r_b / r_a) * tau
            ci_catchup = _future_clean_ci(sim, lookahead_h, catchup_time_s)

            dirty_savings_g = (p_a - p_b) * tau * ci_now / 3.6e6
            clean_catchup_cost_g = p_a * catchup_time_s * ci_catchup / 3.6e6
            switch_cost_now_g = _switch_cost_g(
                sim,
                target_strategy_idx=i,
                ci_g_per_kwh=ci_now,
            )

            if sim.current_strategy_idx == fast_idx:
                switch_back_s = fast.get('reshard_wall_time_s', sim.cfg.switch_time_s)
                switch_back_cost_g = (
                    sim.cfg.idle_cluster_w * switch_back_s * ci_catchup / 3.6e6
                )
            else:
                switch_back_cost_g = 0.0

            benefit_g = (
                dirty_savings_g
                - clean_catchup_cost_g
                - switch_cost_now_g
                - switch_back_cost_g
                - margin_g
            )

            if benefit_g > best_benefit_g:
                best_benefit_g = benefit_g
                best_action = i

        r = sim.step(best_action, decision_interval_s)
        if r['terminated']:
            break

    return sim


def rollout_oracle_dp(sim_cfg: SimConfig,
                      trace_t: np.ndarray,
                      trace_ci: np.ndarray,
                      trace_offset_s: float,
                      decision_interval_s: float = 3600.0,
                      max_hours: float = 200.0,
                      token_quantum: float = 5_000_000.0,
                      switch_cost_mode: str = 'sim',
                      wrap_trace: bool = False,
                      candidate_action_sequences: list[Sequence[int]] | None = None,
                      candidate_initial_strategy_indices: list[int | None] | None = None,
                      respect_deadline: bool = False,
                      debug: bool = False) -> CarbonSim:
    """Discretized dynamic-programming oracle for the fixed-token problem.

    The oracle assumes access to the future carbon trace, minimizes carbon over
    token-progress bins and strategy state, then replays the recovered action
    sequence through `CarbonSim` so reported metrics use the simulator's exact
    accounting.
    """
    if token_quantum <= 0:
        raise ValueError(f'token_quantum must be positive, got {token_quantum}')
    if switch_cost_mode not in ('sim', 'fixed'):
        raise ValueError(
            f"switch_cost_mode must be 'sim' or 'fixed', got {switch_cost_mode!r}"
        )

    strategies = sim_cfg.strategies
    n_strategies = len(strategies)
    target_tokens = float(sim_cfg.pretraining_tokens)
    target_bin = int(math.ceil(target_tokens / token_quantum))
    deadline_s = max_hours * 3600.0
    deadline_tol_s = 1e-6

    ci_probe = CarbonSim(sim_cfg, trace_t, trace_ci, wrap_trace=wrap_trace)
    ci_probe.reset(trace_offset_s, initial_strategy_idx=0)
    initial_idle_s = sim_cfg.container_startup_s + sim_cfg.fresh_build_s
    dp_horizon_s = deadline_s
    if respect_deadline:
        dp_horizon_s = max(0.0, deadline_s - initial_idle_s)
    t_num = int(math.ceil(dp_horizon_s / decision_interval_s))

    def next_hour_boundary_local(t_local: float) -> float:
        trace_pos = trace_offset_s + t_local
        next_trace_hour = (math.floor(trace_pos / 3600.0) + 1.0) * 3600.0
        if next_trace_hour - trace_pos < 1e-9:
            next_trace_hour += 3600.0
        return next_trace_hour - trace_offset_s

    def integrate_constant_power(t_local: float,
                                 duration_s: float,
                                 power: float,
                                 throughput_tok_s: float = 0.0,
                                 token_cap: float | None = None) -> tuple[float, float, float]:
        """CarbonSim-equivalent integration for a local interval.

        Returns `(duration_used_s, tokens_produced, carbon_g)`.
        """
        if duration_s <= 1e-9:
            return 0.0, 0.0, 0.0

        cursor = float(t_local)
        target_end = cursor + float(duration_s)
        consumed = 0.0
        produced = 0.0
        carbon_g = 0.0
        remaining = token_cap

        while cursor < target_end - 1e-9:
            hour_end = next_hour_boundary_local(cursor)
            chunk_end = min(target_end, hour_end)
            chunk_dt = chunk_end - cursor

            if remaining is not None and throughput_tok_s > 0:
                if remaining <= 0:
                    break
                chunk_dt = min(chunk_dt, remaining / throughput_tok_s)

            if chunk_dt <= 1e-9:
                cursor = chunk_end
                continue

            ci = ci_probe.ci_at(trace_offset_s + cursor + chunk_dt / 2.0)
            carbon_g += power * chunk_dt * ci / 3.6e6
            produced_now = throughput_tok_s * chunk_dt
            produced += produced_now
            if remaining is not None:
                remaining -= produced_now
            cursor += chunk_dt
            consumed += chunk_dt

        return consumed, produced, carbon_g

    initial_cost = integrate_constant_power(
        0.0,
        initial_idle_s,
        sim_cfg.idle_cluster_w,
    )[2]

    ci_steps = np.array([
        ci_probe.ci_at(trace_offset_s + initial_idle_s + t * decision_interval_s)
        for t in range(t_num)
    ], dtype=float)

    throughput = np.array([
        float(s['throughput_tok_s'])
        for s in strategies
    ], dtype=float)
    power_w = np.array([
        float(s['power_w'])
        for s in strategies
    ], dtype=float)
    warmup_wall_s = np.array([
        float(s['warmup_wall_s'])
        for s in strategies
    ], dtype=float)
    warmup_throughput = np.array([
        (
            float(s['warmup_iters']) * sim_cfg.tokens_per_iter
            / float(s['warmup_wall_s'])
        )
        if float(s['warmup_wall_s']) > 0 else 0.0
        for s in strategies
    ], dtype=float)

    switch_cost = np.zeros(
        (t_num, n_strategies, n_strategies),
        dtype=float,
    )
    switch_duration = np.zeros(
        (n_strategies, n_strategies),
        dtype=float,
    )
    if switch_cost_mode == 'sim':
        for t in range(t_num):
            decision_start_s = initial_idle_s + t * decision_interval_s
            for prev_k in range(n_strategies):
                for nxt_k in range(n_strategies):
                    if prev_k == nxt_k:
                        continue
                    switch_s = strategies[nxt_k].get(
                        'reshard_wall_time_s',
                        sim_cfg.switch_time_s,
                    )
                    switch_duration[prev_k, nxt_k] = switch_s
                    switch_cost[t, prev_k, nxt_k] = integrate_constant_power(
                        decision_start_s,
                        switch_s,
                        sim_cfg.idle_cluster_w,
                    )[2]
    else:
        fixed_cost = (
            sim_cfg.idle_cluster_w
            * sim_cfg.switch_time_s
            * float(ci_steps.mean())
            / 3.6e6
        )
        switch_cost[:, :, :] = fixed_cost
        for k in range(n_strategies):
            switch_cost[:, k, k] = 0.0
        switch_duration[:, :] = sim_cfg.switch_time_s
        for k in range(n_strategies):
            switch_duration[k, k] = 0.0

    if debug:
        print('[oracle debug] strategies')
        for i, s in enumerate(strategies):
            print(
                i,
                strategy_label(s),
                'P=', s['power_w'],
                'r=', s['throughput_tok_s'],
                'step=', s.get('step_time_s'),
                'reshard=', s.get('reshard_wall_time_s'),
            )
        print('[oracle debug] switch costs at first step')
        for i, s in enumerate(strategies):
            print(
                'target', i, strategy_label(s),
                'reshard_s=', s.get('reshard_wall_time_s', sim_cfg.switch_time_s),
                'switch_g_from_0=', switch_cost[0, 0, i],
            )

    step_cost = np.zeros(
        (t_num, n_strategies, n_strategies),
        dtype=float,
    )
    step_tokens = np.zeros_like(step_cost)
    warmup_start_s = np.zeros_like(step_cost)
    train_start_s = np.zeros_like(step_cost)
    warmup_cost_full = np.zeros_like(step_cost)
    warmup_tokens_full = np.zeros_like(step_cost)
    train_cost_full = np.zeros_like(step_cost)
    train_tokens_full = np.zeros_like(step_cost)

    for t in range(t_num):
        decision_start_s = initial_idle_s + t * decision_interval_s
        for prev_k in range(n_strategies):
            for nxt_k in range(n_strategies):
                switch_s = float(switch_duration[prev_k, nxt_k])
                cursor_s = decision_start_s + switch_s
                budget = max(0.0, decision_interval_s - switch_s)

                warmup_start_s[t, prev_k, nxt_k] = cursor_s

                if (t == 0 or prev_k != nxt_k) and budget > 1e-9:
                    warmup_dt = min(float(warmup_wall_s[nxt_k]), budget)
                    warmup_cost = integrate_constant_power(
                        cursor_s,
                        warmup_dt,
                        float(power_w[nxt_k]),
                    )[2]
                    warmup_tokens = float(warmup_throughput[nxt_k]) * warmup_dt
                    cursor_s += warmup_dt
                    budget -= warmup_dt
                else:
                    warmup_cost = 0.0
                    warmup_tokens = 0.0

                train_start_s[t, prev_k, nxt_k] = cursor_s
                train_dt = max(0.0, budget)
                train_cost = integrate_constant_power(
                    cursor_s,
                    train_dt,
                    float(power_w[nxt_k]),
                )[2]
                train_tokens = float(throughput[nxt_k]) * train_dt

                warmup_cost_full[t, prev_k, nxt_k] = warmup_cost
                warmup_tokens_full[t, prev_k, nxt_k] = warmup_tokens
                train_cost_full[t, prev_k, nxt_k] = train_cost
                train_tokens_full[t, prev_k, nxt_k] = train_tokens
                step_cost[t, prev_k, nxt_k] = (
                    switch_cost[t, prev_k, nxt_k]
                    + warmup_cost
                    + train_cost
                )
                step_tokens[t, prev_k, nxt_k] = warmup_tokens + train_tokens

    def transition(t: int, q: int, prev_k: int, nxt_k: int) -> tuple[int, float]:
        """Approximate one CarbonSim decision step from a token-bin state."""
        q_tokens = min(float(q) * token_quantum, target_tokens)
        remaining = target_tokens - q_tokens
        if remaining <= 0:
            return target_bin, 0.0

        produced = float(step_tokens[t, prev_k, nxt_k])
        if remaining > produced + 1e-6:
            q2 = int(math.floor((q_tokens + produced) / token_quantum))
            q2 = min(target_bin, max(q, q2))
            return q2, float(step_cost[t, prev_k, nxt_k])

        cost = float(switch_cost[t, prev_k, nxt_k])
        warmup_tokens = float(warmup_tokens_full[t, prev_k, nxt_k])
        remaining_in_step = remaining

        if warmup_tokens > 1e-9:
            warmup_thru = float(warmup_throughput[nxt_k])
            if remaining_in_step <= warmup_tokens + 1e-6:
                warmup_dt = remaining_in_step / warmup_thru
                cursor_s = float(warmup_start_s[t, prev_k, nxt_k]) + warmup_dt
                cost += integrate_constant_power(
                    float(warmup_start_s[t, prev_k, nxt_k]),
                    warmup_dt,
                    float(power_w[nxt_k]),
                )[2]
                save_s = float(strategies[nxt_k]['save_s'])
                cost += integrate_constant_power(
                    cursor_s,
                    save_s,
                    sim_cfg.idle_cluster_w,
                )[2]
                return target_bin, cost

            cost += float(warmup_cost_full[t, prev_k, nxt_k])
            remaining_in_step -= warmup_tokens

        train_thru = float(throughput[nxt_k])
        train_dt = remaining_in_step / train_thru
        cursor_s = float(train_start_s[t, prev_k, nxt_k]) + train_dt
        cost += integrate_constant_power(
            float(train_start_s[t, prev_k, nxt_k]),
            train_dt,
            float(power_w[nxt_k]),
        )[2]
        save_s = float(strategies[nxt_k]['save_s'])
        cost += integrate_constant_power(
            cursor_s,
            save_s,
            sim_cfg.idle_cluster_w,
        )[2]
        return target_bin, cost

    dp = np.full(
        (t_num + 1, target_bin + 1, n_strategies),
        np.inf,
        dtype=float,
    )
    parent_q = np.full(
        (t_num + 1, target_bin + 1, n_strategies),
        -1,
        dtype=np.int32,
    )
    parent_k = np.full(
        (t_num + 1, target_bin + 1, n_strategies),
        -1,
        dtype=np.int16,
    )

    dp[0, 0, :] = initial_cost

    for t in range(t_num):
        for q in range(target_bin + 1):
            for prev_k in range(n_strategies):
                base_cost = dp[t, q, prev_k]
                if not np.isfinite(base_cost):
                    continue

                if q >= target_bin:
                    if base_cost < dp[t + 1, q, prev_k]:
                        dp[t + 1, q, prev_k] = base_cost
                        parent_q[t + 1, q, prev_k] = q
                        parent_k[t + 1, q, prev_k] = prev_k
                    continue

                for nxt_k in range(n_strategies):
                    if t == 0 and prev_k != nxt_k:
                        continue
                    q2, transition_cost = transition(t, q, prev_k, nxt_k)
                    cost = base_cost + transition_cost
                    if cost < dp[t + 1, q2, nxt_k]:
                        dp[t + 1, q2, nxt_k] = cost
                        parent_q[t + 1, q2, nxt_k] = q
                        parent_k[t + 1, q2, nxt_k] = prev_k

    best_t = -1
    best_k = -1
    best_cost = np.inf
    for t in range(t_num + 1):
        for k in range(n_strategies):
            cost = dp[t, target_bin, k]
            if cost < best_cost:
                best_cost = cost
                best_t = t
                best_k = k

    if not np.isfinite(best_cost):
        fastest_idx = _fastest_strategy_idx(sim_cfg)
        actions = [fastest_idx] * max(1, t_num)
        initial_k = fastest_idx
    else:
        actions = []
        q = target_bin
        k = best_k
        for t in range(best_t, 0, -1):
            actions.append(k)
            pq = int(parent_q[t, q, k])
            pk = int(parent_k[t, q, k])
            if pq < 0 or pk < 0:
                break
            q, k = pq, pk
        actions.reverse()
        initial_k = k

    if debug:
        from collections import Counter

        print('[oracle debug] terminal solution')
        print('best_t:', best_t)
        print('best_k:', best_k)
        print('best_q:', target_bin)
        print('best_dp_carbon:', best_cost)
        print('initial_k:', initial_k)
        print('[oracle debug] reconstructed actions before repair')
        print('length:', len(actions))
        print('counts:', Counter(actions))
        print('actions:', actions)
        print(
            'strategy tuples:',
            [strategy_label(strategies[a]) for a in actions],
        )

    if not actions:
        fastest_idx = _fastest_strategy_idx(sim_cfg)
        actions = [fastest_idx] * max(1, t_num)
        initial_k = fastest_idx

    def replay_actions(action_seq: Sequence[int],
                       initial_strategy_idx: int | None = None) -> CarbonSim:
        return _replay_action_sequence(
            sim_cfg,
            trace_t,
            trace_ci,
            trace_offset_s,
            action_seq,
            initial_strategy_idx=initial_strategy_idx,
            decision_interval_s=decision_interval_s,
            max_hours=max_hours,
            wrap_trace=wrap_trace,
        )

    def exact_deadline_feasible(candidate_sim: CarbonSim) -> bool:
        return (
            candidate_sim.done()
            and (
                not respect_deadline
                or candidate_sim.cursor_s <= deadline_s + deadline_tol_s
            )
        )

    def candidate_is_better(candidate_sim: CarbonSim,
                            current_sim: CarbonSim | None) -> bool:
        candidate_feasible = exact_deadline_feasible(candidate_sim)
        current_feasible = (
            current_sim is not None and exact_deadline_feasible(current_sim)
        )
        if candidate_feasible:
            return (
                not current_feasible
                or candidate_sim.total_carbon_g
                < current_sim.total_carbon_g - 1e-9
            )
        if current_feasible or not candidate_sim.done():
            return False
        if current_sim is None or not current_sim.done():
            return True
        return candidate_sim.total_carbon_g < current_sim.total_carbon_g - 1e-9

    def replay_improves(candidate_actions: list[int],
                        current_sim: CarbonSim) -> tuple[bool, CarbonSim]:
        candidate_sim = replay_actions(candidate_actions)
        improves = candidate_is_better(candidate_sim, current_sim)
        return improves, candidate_sim

    sim = replay_actions(actions)
    original_sim = sim

    # Repair local token-bin artifacts using exact CarbonSim replay. This keeps
    # the DP policy as the starting point, but rejects edits unless they improve
    # exact fixed-token carbon accounting.
    improved = True
    while improved:
        improved = False
        for i in range(1, len(actions) - 1):
            if actions[i - 1] == actions[i + 1] and actions[i] != actions[i - 1]:
                trial = list(actions)
                trial[i] = actions[i - 1]
                ok, trial_sim = replay_improves(trial, sim)
                if ok:
                    actions = trial
                    sim = trial_sim
                    improved = True

    for _ in range(3):
        improved = False
        for i in range(len(actions)):
            current_action = actions[i]
            for k2 in range(n_strategies):
                if k2 == current_action:
                    continue
                trial = list(actions)
                trial[i] = k2
                ok, trial_sim = replay_improves(trial, sim)
                if ok:
                    actions = trial
                    sim = trial_sim
                    current_action = k2
                    improved = True
        if not improved:
            break

    if debug:
        from collections import Counter

        print('[oracle debug] carbon before repair:', original_sim.total_carbon_g)
        print('[oracle debug] replayed oracle before repair')
        print('carbon:', original_sim.total_carbon_g)
        print('wall_h:', original_sim.cursor_s / 3600.0)
        print('reshards:', original_sim.reshard_count)
        print('[oracle debug] reconstructed actions after repair')
        print('length:', len(actions))
        print('counts:', Counter(actions))
        print('actions:', actions)
        print(
            'strategy tuples:',
            [strategy_label(strategies[a]) for a in actions],
        )
        print('[oracle debug] replayed oracle after repair')
        print('carbon:', sim.total_carbon_g)
        print('wall_h:', sim.cursor_s / 3600.0)
        print('reshards:', sim.reshard_count)

    # Static schedules are feasible oracle schedules. Because the DP optimizes
    # a token-quantized approximation and replay uses exact simulator accounting
    # (including warmup/save), keep the best exact replay.
    best_sim = sim
    if candidate_action_sequences:
        for seq_i, candidate_actions in enumerate(candidate_action_sequences):
            if not candidate_actions:
                continue
            initial_strategy_idx = None
            if (candidate_initial_strategy_indices is not None
                    and seq_i < len(candidate_initial_strategy_indices)):
                initial_strategy_idx = candidate_initial_strategy_indices[seq_i]
            candidate_sim = replay_actions(
                candidate_actions,
                initial_strategy_idx=initial_strategy_idx,
            )
            if debug:
                print(f'[oracle debug] candidate sequence {seq_i} replay')
                print('carbon:', candidate_sim.total_carbon_g)
                print('wall_h:', candidate_sim.cursor_s / 3600.0)
                print('reshards:', candidate_sim.reshard_count)
                print('actions:', list(candidate_actions))
            if not candidate_sim.done():
                continue
            if candidate_is_better(candidate_sim, best_sim):
                best_sim = candidate_sim

    for strategy_idx in range(n_strategies):
        static_sim = rollout_static(
            sim_cfg,
            trace_t,
            trace_ci,
            trace_offset_s,
            strategy_idx=strategy_idx,
            decision_interval_s=decision_interval_s,
            max_hours=max_hours,
            wrap_trace=wrap_trace,
        )
        if not static_sim.done():
            continue
        if candidate_is_better(static_sim, best_sim):
            best_sim = static_sim

    if debug:
        static_carbons = []
        for strategy_idx in range(n_strategies):
            static_sim = rollout_static(
                sim_cfg,
                trace_t,
                trace_ci,
                trace_offset_s,
                strategy_idx=strategy_idx,
                decision_interval_s=decision_interval_s,
                max_hours=max_hours,
                wrap_trace=wrap_trace,
            )
            if static_sim.done():
                static_carbons.append(static_sim.total_carbon_g)
        if static_carbons:
            print('[oracle debug] best static carbon:', min(static_carbons))
        print('[oracle debug] returned oracle carbon:', best_sim.total_carbon_g)
        print('[oracle debug] returned oracle exact-deadline feasible:',
              exact_deadline_feasible(best_sim))

    return best_sim


def rollout_oracle_shift_switch(
    sim_cfg: SimConfig,
    trace_t: np.ndarray,
    trace_ci: np.ndarray,
    arrival_offset_s: float,
    decision_interval_s: float = 3600.0,
    max_episode_hours: float = 72.0,
    token_quantum: float = 5_000_000.0,
    switch_cost_mode: str = 'sim',
    wrap_trace: bool = False,
    debug: bool = False,
) -> tuple[CarbonSim, float]:
    """Oracle that jointly chooses start time and switching schedule.

    This wraps the fixed-start DP oracle in an hourly outer loop. For each
    candidate start time, the fixed-start oracle receives the remaining time
    to the original deadline. The returned simulator is already an exact
    CarbonSim replay; the wrapper only selects the best feasible replay.

    Returns `(sim, wait_s)`, where `wait_s` is the delay from arrival to the
    selected start. If no candidate completes before the deadline, the best
    infeasible replay found is returned with its wait so callers can report it.
    """
    if decision_interval_s <= 0:
        raise ValueError(
            f'decision_interval_s must be positive, got {decision_interval_s}'
        )
    if max_episode_hours <= 0:
        raise ValueError(
            f'max_episode_hours must be positive, got {max_episode_hours}'
        )

    deadline_s = float(arrival_offset_s) + float(max_episode_hours) * 3600.0
    total_horizon_s = deadline_s - float(arrival_offset_s)
    n_candidates = int(math.ceil(total_horizon_s / decision_interval_s))
    deadline_tol_s = 1e-6

    best_sim: CarbonSim | None = None
    best_wait_s = 0.0
    best_infeasible_sim: CarbonSim | None = None
    best_infeasible_wait_s = 0.0
    best_infeasible_key: tuple[float, float] | None = None

    for j in range(n_candidates):
        start_offset_s = float(arrival_offset_s) + j * decision_interval_s
        if start_offset_s >= deadline_s - deadline_tol_s:
            break

        horizon_h = (deadline_s - start_offset_s) / 3600.0
        if horizon_h <= 0:
            continue

        try:
            sim = rollout_oracle_dp(
                sim_cfg,
                trace_t,
                trace_ci,
                start_offset_s,
                decision_interval_s=decision_interval_s,
                max_hours=horizon_h,
                token_quantum=token_quantum,
                switch_cost_mode=switch_cost_mode,
                wrap_trace=wrap_trace,
                respect_deadline=True,
                debug=debug,
            )
        except TraceOverrunError:
            continue

        wait_s = start_offset_s - float(arrival_offset_s)
        completion_delay_s = wait_s + sim.cursor_s
        feasible = (
            sim.done()
            and completion_delay_s <= total_horizon_s + deadline_tol_s
        )

        if feasible:
            if best_sim is None or sim.total_carbon_g < best_sim.total_carbon_g:
                best_sim = sim
                best_wait_s = wait_s
            continue

        infeasible_key = (
            float(sim.tokens_completed),
            -float(sim.total_carbon_g),
        )
        if best_infeasible_key is None or infeasible_key > best_infeasible_key:
            best_infeasible_sim = sim
            best_infeasible_wait_s = wait_s
            best_infeasible_key = infeasible_key

    if best_sim is not None:
        return best_sim, float(best_wait_s)

    if best_infeasible_sim is not None:
        return best_infeasible_sim, float(best_infeasible_wait_s)

    fallback_sim = rollout_static(
        sim_cfg,
        trace_t,
        trace_ci,
        float(arrival_offset_s),
        strategy_idx=_fastest_strategy_idx(sim_cfg),
        decision_interval_s=decision_interval_s,
        max_hours=max_episode_hours,
        wrap_trace=wrap_trace,
    )
    return fallback_sim, 0.0


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
            [strategy_label(s, include_dp=False) for s in sim_cfg.strategies]
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
    ax_strategy.set_yticklabels([
        strategy_label(s, include_dp=False) for s in sim_cfg.strategies
    ])
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
# Synthetic demonstration profile; not hardware measurements.
# ---------------------------------------------------------------------------

DEFAULT_WARMUP_WALL_S = 0.0
DEFAULT_SAVE_S = 0.0
DEFAULT_STRATEGIES = [
    {'tp': 1, 'pp': 1, 'dp': 8, 'power_w': 2000.0,
     'throughput_tok_s': 100000.0, 'reshard_wall_time_s': 0.0},
    {'tp': 1, 'pp': 2, 'dp': 4, 'power_w': 1000.0,
     'throughput_tok_s': 40000.0, 'reshard_wall_time_s': 0.0},
]


def filter_feasible_strategies(strategies: list[dict],
                               *,
                               model_num_params: int,
                               n_gpus: int = 8,
                               gpu_memory_gb: float = 48.0,
                               usable_memory_frac: float = 0.90,
                               bytes_per_param_training: float = 18.0,
                               activation_reserve_gb: float = 4.0,
                               min_pp: int = 1,
                               verbose: bool = False,
                               return_removed: bool = False):
    """Filter strategies by per-GPU model-memory feasibility.

    The token budget controls episode length and is intentionally not part of
    this filter. DP does not shard model memory; TP and PP do, so the model
    memory estimate is divided by `tp * pp`.
    """
    feasible = []
    removed = []
    usable_memory_gb = gpu_memory_gb * usable_memory_frac

    for original_idx, raw in enumerate(strategies):
        s = deepcopy(raw)
        s['original_strategy_idx'] = raw.get('original_strategy_idx', original_idx)
        tp = int(s.get('tp', 1))
        pp = int(s.get('pp', 1))
        dp = s.get('dp')
        if dp is None and tp * pp > 0 and n_gpus % (tp * pp) == 0:
            dp = n_gpus // (tp * pp)
            s['dp'] = dp

        shard_factor = max(1, tp * pp)
        param_memory_gb = (
            float(model_num_params)
            * float(bytes_per_param_training)
            / shard_factor
            / 1e9
        )
        estimated_memory_gb = param_memory_gb + float(activation_reserve_gb)
        s['estimated_memory_gb'] = estimated_memory_gb

        reason = 'ok'
        if pp < min_pp:
            reason = f'pp={pp} < min_pp={min_pp}'
        elif tp * pp <= 0:
            reason = 'invalid tp/pp'
        elif tp * pp > n_gpus:
            reason = f'tp*pp={tp * pp} exceeds n_gpus={n_gpus}'
        elif n_gpus % (tp * pp) != 0:
            reason = f'tp*pp={tp * pp} does not divide n_gpus={n_gpus}'
        elif estimated_memory_gb > usable_memory_gb:
            reason = (
                f'estimated_memory_gb={estimated_memory_gb:.1f} exceeds '
                f'usable_gpu_memory_gb={usable_memory_gb:.1f}'
            )

        s['feasibility_reason'] = reason
        if reason == 'ok':
            feasible.append(s)
        else:
            removed.append(s)

    if verbose:
        print(
            f'Feasible strategies: {len(feasible)}/{len(strategies)} '
            f'for model_num_params={model_num_params:,}'
        )

    if return_removed:
        return feasible, removed
    return feasible


def print_feasibility_summary(model_name: str,
                              model_num_params: int,
                              strategies_before: list[dict],
                              strategies_after: list[dict],
                              removed_strategies: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Feasibility summary: {model_name}")
    print("=" * 72)
    print(f"model_num_params: {model_num_params:,}")
    print(f"candidate strategies: {len(strategies_before)}")
    print(f"feasible strategies:  {len(strategies_after)}")
    print(f"removed strategies:   {len(removed_strategies)}")

    print("\nFeasible:")
    for s in strategies_after:
        print(
            f"  {strategy_label(s)}, "
            f"est_mem={s.get('estimated_memory_gb', float('nan')):.1f}GB"
        )

    if removed_strategies:
        print("\nRemoved:")
        for s in removed_strategies:
            print(
                f"  {strategy_label(s)}, "
                f"est_mem={s.get('estimated_memory_gb', float('nan')):.1f}GB, "
                f"reason={s.get('feasibility_reason', 'unknown')}"
            )

    if len(strategies_after) == 0:
        print(
            "\n[error] No feasible strategies remain for this model/hardware setting."
        )

    elif len(strategies_after) == 1:
        s = strategies_after[0]
        print(
            "\n[warning] Only one feasible strategy remains. "
            "This is a static-only feasible case, not a meaningful dynamic "
            "scheduling/oracle comparison."
        )
        print(
            f"[warning] Remaining strategy: "
            f"{strategy_label(s)}"
        )

    else:
        print(
            "\n[ok] Multiple feasible strategies remain. "
            "Dynamic scheduling comparison is meaningful."
        )


def strategy_dominance_report(cfg: SimConfig, ci_samples=(40, 110, 260)) -> dict:
    """Print and return a dominance analysis for the strategies in `cfg`.

    Reports:
      - per-strategy J/tok, kWh per pretraining run, wall hours per run
      - Pareto-non-dominated strategies on (throughput, J/tok)
      - whether a single strategy is the carbon-optimum at every CI sample

    Returns a dict with `pareto_labels`, `dominant_at_all_ci` (bool), and
    `optimal_per_ci` (dict ci → label).
    """
    rows = []
    for s in cfg.strategies:
        thru = float(s['throughput_tok_s'])
        j_per_tok = s['power_w'] / thru
        rows.append({
            'label': strategy_label(s),
            'tp': s['tp'], 'pp': s['pp'],
            'micro_batch_size': s.get('micro_batch_size'),
            'power_w': s['power_w'],
            'throughput': thru,
            'j_per_tok': j_per_tok,
            'wh_per_run': j_per_tok * cfg.pretraining_tokens / 3600.0,
            'wall_h': cfg.pretraining_tokens / thru / 3600.0,
        })

    print(f"\n{'strategy':<20} {'P (W)':>7} {'tok/s':>9} {'J/tok':>8} "
          f"{'kWh/run':>9} {'wall (h)':>10}")
    print('-' * 70)
    for r in sorted(rows, key=lambda r: r['j_per_tok']):
        print(f"{r['label']:<20} {r['power_w']:>7.0f} {r['throughput']:>9,.0f} "
              f"{r['j_per_tok']:>8.4f} {r['wh_per_run']/1000:>9.3f} "
              f"{r['wall_h']:>10.2f}")

    pareto = []
    for ri in rows:
        dominated = any(
            rj['throughput'] >= ri['throughput']
            and rj['j_per_tok'] <= ri['j_per_tok']
            and (rj['throughput'] > ri['throughput']
                 or rj['j_per_tok'] < ri['j_per_tok'])
            for rj in rows
        )
        if not dominated:
            pareto.append(ri['label'])
    print(f"\nPareto frontier on (throughput, J/tok): {pareto}")

    optimal_per_ci = {}
    for ci in ci_samples:
        best = min(rows, key=lambda r: r['j_per_tok'] * ci)
        optimal_per_ci[ci] = best['label']
    dominant_at_all_ci = len(set(optimal_per_ci.values())) == 1
    print(f'Carbon-optimal strategy by CI sample: {optimal_per_ci}')
    if dominant_at_all_ci:
        print(f"  ⚠  One strategy ({list(optimal_per_ci.values())[0]}) wins at "
              f"every CI sample. Pure-carbon RL has no learning signal here; "
              f"add another source of scheduling flexibility to make the "
              f"problem nontrivial.")
    return {
        'pareto_labels': pareto,
        'dominant_at_all_ci': dominant_at_all_ci,
        'optimal_per_ci': optimal_per_ci,
    }


def default_config(pretraining_tokens: int = 3_000_000_000,
                   global_batch_size: int = 16,
                   seq_length: int = 1024,
                   n_gpus: int = 8,
                   model_num_params: int | None = None,
                   model_name: str | None = None,
                   filter_strategies: bool = False,
                   gpu_memory_gb: float = 48.0,
                   usable_memory_frac: float = 0.90,
                   bytes_per_param_training: float = 18.0,
                   activation_reserve_gb: float = 4.0,
                   min_pp: int = 1,
                   verbose_feasibility: bool = True) -> SimConfig:
    tokens_per_iter = global_batch_size * seq_length
    strategies = deepcopy(DEFAULT_STRATEGIES)
    if filter_strategies and model_num_params is not None:
        original_strategies = deepcopy(strategies)
        strategies, removed = filter_feasible_strategies(
            strategies,
            model_num_params=model_num_params,
            n_gpus=n_gpus,
            gpu_memory_gb=gpu_memory_gb,
            usable_memory_frac=usable_memory_frac,
            bytes_per_param_training=bytes_per_param_training,
            activation_reserve_gb=activation_reserve_gb,
            min_pp=min_pp,
            verbose=False,
            return_removed=True,
        )

        if verbose_feasibility:
            if model_name is None:
                model_name = f"{model_num_params / 1e9:.0f}B"
            print_feasibility_summary(
                model_name=model_name,
                model_num_params=model_num_params,
                strategies_before=original_strategies,
                strategies_after=strategies,
                removed_strategies=removed,
            )

        if not strategies:
            raise ValueError(
                'No feasible strategies remain for this model/hardware setting'
            )

    for s in strategies:
        s['warmup_wall_s'] = s.get('warmup_wall_s', DEFAULT_WARMUP_WALL_S)
        s['save_s'] = s.get('save_s', DEFAULT_SAVE_S)
        s['warmup_iters'] = s.get(
            'warmup_iters',
            max(0, round(
                s['warmup_wall_s'] * s['throughput_tok_s'] / tokens_per_iter
            )),
        )

    return SimConfig(
        strategies=strategies,
        pretraining_tokens=pretraining_tokens,
        tokens_per_iter=tokens_per_iter,
        n_gpus=n_gpus,
        switch_time_s=0.0,
        first_switch_cache_miss_s=0.0,
    )

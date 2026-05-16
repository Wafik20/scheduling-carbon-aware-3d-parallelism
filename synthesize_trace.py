"""Generate synthetic carbon-intensity traces for training/evaluation.

Real grid CI traces show:
  - Strong diurnal swing (low when solar is producing, high at night/morning)
  - Modest weekly cycle (lower industrial demand on weekends)
  - Seasonal drift (winter higher than summer in solar-heavy grids)
  - AR(1) "weather" noise (cloudy week pushes CI up for several days)
  - Occasional extreme events (heat wave, grid stress)

This module produces a CSV with the same schema as the real carbon_week.csv
(year,month,day,time,carbon) but spans an arbitrary number of days. Calibrated
by default to match the distribution of the user's real trace
(min~40, max~260, mean~110 g/kWh).

CLI:
    python synthesize_trace.py --days 365 --out carbon_year.csv --seed 0
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def synthesize_trace(
    days: int = 365,
    start_date: str = '2026-01-01',
    *,
    base_ci: float = 110.0,
    diurnal_amplitude: float = 65.0,
    diurnal_amplitude_jitter: float = 20.0,
    diurnal_phase_hours: float = 9.0,
    weekly_amplitude: float = 8.0,
    seasonal_amplitude: float = 25.0,
    weather_sigma: float = 18.0,
    weather_persistence: float = 0.92,
    extreme_event_rate_per_week: float = 0.4,
    extreme_event_size_g: float = 60.0,
    clip_min: float = 30.0,
    clip_max: float = 280.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Generate an hourly carbon-intensity trace as a DataFrame.

    The CI at hour h is approximately:

        ci[h] = base
              + seasonal(day_of_year)              # slow ~±25 swing
              + weekly(day_of_week)                # small dip on weekends
              + diurnal(hour_of_day) * day_amp     # the dominant signal
              + ar1_weather[h]                     # autocorrelated noise
              + extreme_events[h]                  # occasional bumps

    Args:
        days: number of days to generate (hourly resolution).
        start_date: ISO date string for the first row.
        base_ci: mean CI level. Defaults to ~110 g/kWh.
        diurnal_amplitude: peak-to-mean of the daily cycle.
        diurnal_amplitude_jitter: stddev of per-day amplitude (some days have
            stronger solar than others — cloudy days → small amplitude).
        diurnal_phase_hours: hour-of-day where the daily MIN sits (default 15,
            i.e. mid-afternoon when solar peaks). Set via the cosine offset.
        weekly_amplitude: weekday-weekend swing in g/kWh.
        seasonal_amplitude: peak-to-mean of the annual cycle.
        weather_sigma: stddev of the innovation feeding the AR(1) weather noise.
        weather_persistence: AR(1) coefficient. 0.92 → ~12-hour decorrelation.
        extreme_event_rate_per_week: expected number of grid-stress events per
            week. Each event is a brief positive bump.
        extreme_event_size_g: typical bump size in g/kWh.
        clip_min, clip_max: hard floor/ceiling.
        seed: numpy RNG seed.

    Returns:
        DataFrame with columns ['year','month','day','time','carbon'].
    """
    rng = np.random.default_rng(seed)
    n = days * 24
    ts = pd.date_range(start_date, periods=n, freq='1h')

    hour_of_day = ts.hour.to_numpy()
    day_of_week = ts.dayofweek.to_numpy()  # Mon=0
    day_of_year = ts.dayofyear.to_numpy()
    day_idx = np.arange(n) // 24

    # --- Diurnal: minimum near `diurnal_phase_hours` (afternoon solar peak)
    # cos(2π * (h - phase) / 24) → max at phase, so we negate
    diurnal_shape = -np.cos(2 * np.pi * (hour_of_day - diurnal_phase_hours) / 24)
    # Per-day random amplitude (cloudy days → smaller solar dip)
    day_amp = rng.normal(diurnal_amplitude, diurnal_amplitude_jitter, size=days)
    day_amp = np.clip(day_amp, 5.0, None)  # keep some swing even on bad days
    diurnal = diurnal_shape * day_amp[day_idx]

    # --- Seasonal: cosine peaking in winter (around day-of-year 15)
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # --- Weekly: small dip on Saturday/Sunday
    weekend = ((day_of_week == 5) | (day_of_week == 6)).astype(float)
    weekly = -weekly_amplitude * weekend  # subtract on weekend hours

    # --- AR(1) weather noise: persistent multi-day drift
    weather = np.zeros(n)
    for i in range(1, n):
        weather[i] = (weather_persistence * weather[i - 1]
                      + rng.normal(0, weather_sigma))

    # --- Extreme events: Poisson process of brief positive bumps
    expected_events = extreme_event_rate_per_week * (days / 7.0)
    n_events = rng.poisson(expected_events)
    event_starts = rng.integers(0, n - 24, size=n_events) if n_events else np.array([], int)
    event_durations = rng.integers(2, 12, size=n_events) if n_events else np.array([], int)
    event_sizes = rng.normal(extreme_event_size_g,
                              extreme_event_size_g * 0.3,
                              size=n_events) if n_events else np.array([])
    events = np.zeros(n)
    for s, d, mag in zip(event_starts, event_durations, event_sizes):
        # Triangular bump (ramp up, ramp down)
        ramp = np.minimum(np.arange(d), np.arange(d)[::-1])
        ramp = ramp / max(1, ramp.max()) if ramp.max() > 0 else ramp
        events[s:s + d] += mag * ramp

    ci = base_ci + diurnal + seasonal + weekly + weather + events
    ci = np.clip(ci, clip_min, clip_max)
    ci = np.round(ci).astype(int)

    return pd.DataFrame({
        'year': ts.year,
        'month': ts.month,
        'day': ts.day,
        'time': ts.strftime('%H:%M:%S'),
        'carbon': ci,
    })


def summarize(df: pd.DataFrame) -> dict:
    ci = df['carbon'].to_numpy()
    return {
        'n_hours': len(ci),
        'n_days': len(ci) / 24,
        'min': int(ci.min()),
        'max': int(ci.max()),
        'mean': float(ci.mean()),
        'median': float(np.median(ci)),
        'p10': float(np.percentile(ci, 10)),
        'p90': float(np.percentile(ci, 90)),
        'std': float(ci.std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=365)
    p.add_argument('--start-date', default='2026-01-01')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='carbon_synth.csv')
    p.add_argument('--base-ci', type=float, default=110.0)
    p.add_argument('--diurnal-amplitude', type=float, default=65.0)
    p.add_argument('--seasonal-amplitude', type=float, default=25.0)
    p.add_argument('--weather-sigma', type=float, default=18.0)
    args = p.parse_args()

    df = synthesize_trace(
        days=args.days,
        start_date=args.start_date,
        base_ci=args.base_ci,
        diurnal_amplitude=args.diurnal_amplitude,
        seasonal_amplitude=args.seasonal_amplitude,
        weather_sigma=args.weather_sigma,
        seed=args.seed,
    )
    df.to_csv(args.out, index=False)

    s = summarize(df)
    print(f"Wrote {args.out}: {s['n_hours']} hours ({s['n_days']:.0f} days)")
    print(f"  CI g/kWh: min={s['min']} max={s['max']} mean={s['mean']:.1f} "
          f"median={s['median']:.0f}  p10/p90={s['p10']:.0f}/{s['p90']:.0f}  "
          f"std={s['std']:.1f}")


if __name__ == '__main__':
    main()
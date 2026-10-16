"""Generate illustrative carbon intensity, not real grid measurements."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def generate_carbon_dataset(
    start="2024-01-01",
    end="2024-02-01",
    freq="1h",
    seed=42
):
    rng = np.random.default_rng(seed)

    time_index = pd.date_range(start=start, end=end, freq=freq)
    df = pd.DataFrame({"timestamp": time_index})

    base = 400

    hour = df["timestamp"].dt.hour
    daily = -150 * np.cos((hour - 12) / 24 * 2 * np.pi)

    dayofweek = df["timestamp"].dt.dayofweek
    weekly = np.where(dayofweek < 5, 0, -80)

    day_of_year = df["timestamp"].dt.dayofyear
    seasonal = 100 * np.sin(2 * np.pi * day_of_year / 365)

    noise = np.zeros(len(df))
    for i in range(1, len(df)):
        noise[i] = 0.8 * noise[i-1] + rng.normal(0, 20)

    spikes = np.zeros(len(df))
    spike_indices = rng.choice(len(df), size=int(0.03 * len(df)), replace=False)
    spikes[spike_indices] = rng.uniform(150, 400, len(spike_indices))

    regime = np.zeros(len(df))
    if len(df) > 48:
        for _ in range(3):
            start_idx = rng.integers(0, len(df) - 48)
            duration = rng.integers(24, 72)
            regime[start_idx:start_idx+duration] += rng.choice([-150, 150])

    df["carbon"] = base + daily + weekly + seasonal + noise + spikes + regime

    df["carbon"] = np.clip(df["carbon"], 50, None)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

    df_final = df[["year", "month", "day", "time", "carbon"]]

    return df_final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', default='2024-01-01')
    parser.add_argument('--end', default='2024-02-01')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = generate_carbon_dataset(start=args.start, end=args.end, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f'Saved synthetic trace to {args.out}')


if __name__ == '__main__':
    main()

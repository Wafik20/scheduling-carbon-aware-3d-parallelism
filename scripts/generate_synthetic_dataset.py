import numpy as np
import pandas as pd

def generate_carbon_dataset(
    start="2024-01-01",
    end="2024-02-01",
    freq="1h",
    seed=42
):
    np.random.seed(seed)

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
        noise[i] = 0.8 * noise[i-1] + np.random.normal(0, 20)

    spikes = np.zeros(len(df))
    spike_indices = np.random.choice(len(df), size=int(0.03 * len(df)), replace=False)
    spikes[spike_indices] = np.random.uniform(150, 400, len(spike_indices))

    regime = np.zeros(len(df))
    for _ in range(3):  # 3 random regimes
        start_idx = np.random.randint(0, len(df) - 48)
        duration = np.random.randint(24, 72)
        shift = np.random.choice([-150, 150])
        regime[start_idx:start_idx+duration] += shift

    df["carbon"] = base + daily + weekly + seasonal + noise + spikes + regime

    df["carbon"] = np.clip(df["carbon"], 50, None)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

    df_final = df[["year", "month", "day", "time", "carbon"]]

    return df_final


df = generate_carbon_dataset()

df.to_csv("data/carbon_synthetic_dataset.csv", index=False)

print("Saved to carbon_dataset.csv")
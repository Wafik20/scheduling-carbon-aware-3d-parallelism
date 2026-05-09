import os
import csv
import requests

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# ============================================
# LOAD ENV VARIABLES
# ============================================

load_dotenv()

EMAPS_API_KEY = os.getenv("EMAPS_API_KEY")

if EMAPS_API_KEY is None:
    raise ValueError("EMAPS_API_KEY not found in .env")

# ============================================
# PROJECT PATHS
# ============================================

# script lives in:
# project_root/scripts/this_script.py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = DATA_DIR / "carbon_week.csv"

# ============================================
# CONFIG
# ============================================

# High-variation California region
DATA_CENTER_REGION = "us-west-1"
DATA_CENTER_PROVIDER = "aws"

# ============================================
# TIME RANGE (LAST 7 DAYS)
# ============================================

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=7)

start_str = start_time.isoformat(timespec="milliseconds").replace("+00:00", "Z")
end_str = end_time.isoformat(timespec="milliseconds").replace("+00:00", "Z")

# ============================================
# API REQUEST
# ============================================

url = "https://api.electricitymaps.com/v3/carbon-intensity/history"

params = {
    "dataCenterRegion": DATA_CENTER_REGION,
    "dataCenterProvider": DATA_CENTER_PROVIDER,
    "start": start_str,
    "end": end_str,
}

headers = {
    "auth-token": EMAPS_API_KEY
}

print("Requesting hourly carbon intensity data...")

response = requests.get(url, params=params, headers=headers)

response.raise_for_status()

data = response.json()

# ============================================
# SAVE CSV
# ============================================

history = data["history"]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow(["year", "month", "day", "time", "carbon"])

    for entry in history:
        dt = datetime.fromisoformat(
            entry["datetime"].replace("Z", "+00:00")
        )

        carbon = entry["carbonIntensity"]

        writer.writerow([
            dt.year,
            dt.month,
            dt.day,
            dt.strftime("%H:%M:%S"),
            carbon
        ])

print(f"Saved {len(history)} hourly readings to:")
print(OUTPUT_CSV)
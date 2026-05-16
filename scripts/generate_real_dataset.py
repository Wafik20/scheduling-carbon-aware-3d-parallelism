#!/usr/bin/env python3
"""
generate_real_dataset.py
------------------------
Fetch carbon-intensity history from Electricity Maps and save to
data/carbon_week.csv (filename kept for downstream-path compatibility).

Uses the /v3/carbon-intensity/past-range endpoint, which supports up to
10 days of hourly data per call. We chunk by 10 days and stitch the
results together.

Date format matches the Electricity Maps docs example exactly:
"YYYY-MM-DD HH:MM" (the requests library URL-encodes the space to '+'
and the colon to '%3A' automatically when building the query string).

Override knobs:

    DAYS_BACK=60                 python scripts/generate_real_dataset.py
    CHUNK_DAYS=5                 python scripts/generate_real_dataset.py
    DATA_CENTER_REGION=eu-west-1 python scripts/generate_real_dataset.py

Runs against whatever EMAPS_API_KEY is in .env at the project root.
"""

import os
import csv
import sys
import time
import requests

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# ============================================
# LOAD ENV
# ============================================
load_dotenv()
EMAPS_API_KEY = os.getenv("EMAPS_API_KEY")
if EMAPS_API_KEY is None:
    raise ValueError("EMAPS_API_KEY not found in .env")

# ============================================
# PROJECT PATHS
# ============================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_BASENAME = os.getenv("OUTPUT_BASENAME", "carbon_week.csv")
OUTPUT_CSV = DATA_DIR / OUTPUT_BASENAME

# ============================================
# CONFIG
# ============================================
DATA_CENTER_REGION    = os.getenv("DATA_CENTER_REGION",   "us-west-1")
DATA_CENTER_PROVIDER  = os.getenv("DATA_CENTER_PROVIDER", "aws")
DAYS_BACK             = int(os.getenv("DAYS_BACK",   "30"))
# past-range hourly granularity is capped at 10 days per call.
CHUNK_DAYS            = int(os.getenv("CHUNK_DAYS", "10"))
GRANULARITY           = os.getenv("GRANULARITY", "hourly")  # or "daily"
SLEEP_BETWEEN_CALLS_S = float(os.getenv("SLEEP_BETWEEN_CALLS_S", "0.3"))

PAST_RANGE_URL = "https://api.electricitymaps.com/v3/carbon-intensity/past-range"


def fetch_chunk(start_dt, end_dt):
    """One call to /past-range covering [start_dt, end_dt].

    Date format matches the docs example: "YYYY-MM-DD HH:MM". The space
    becomes '+' and the colon becomes '%3A' once requests encodes the
    query string, matching the example URL exactly.
    """
    params = {
        "dataCenterRegion":   DATA_CENTER_REGION,
        "dataCenterProvider": DATA_CENTER_PROVIDER,
        "start": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end":   end_dt.strftime("%Y-%m-%d %H:%M"),
        "granularity": GRANULARITY,
    }
    headers = {"auth-token": EMAPS_API_KEY}
    r = requests.get(PAST_RANGE_URL, params=params, headers=headers, timeout=30)
    if r.status_code in (401, 403, 405):
        raise RuntimeError(
            f"past-range returned {r.status_code}: {r.text[:200]}\n"
            "  This endpoint may require a Commercial-tier API plan."
        )
    r.raise_for_status()
    body = r.json()
    return body.get("data") or body.get("history") or []


def main():
    end_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    target_start = end_time - timedelta(days=DAYS_BACK)

    print(f"Region:      {DATA_CENTER_PROVIDER}/{DATA_CENTER_REGION}")
    print(f"Range:       {target_start.isoformat()} → {end_time.isoformat()}")
    print(f"Days back:   {DAYS_BACK}, chunk size: {CHUNK_DAYS} days, "
          f"granularity: {GRANULARITY}")
    print(f"Endpoint:    {PAST_RANGE_URL}")
    print(f"Output:      {OUTPUT_CSV}")
    print()
    print("Fetching...")

    all_entries = []
    seen_datetimes = set()
    chunk_end = end_time
    chunk_n = 0

    while chunk_end > target_start:
        chunk_start = max(target_start, chunk_end - timedelta(days=CHUNK_DAYS))
        chunk_n += 1
        try:
            entries = fetch_chunk(chunk_start, chunk_end)
        except (requests.HTTPError, RuntimeError) as e:
            print(f"  chunk {chunk_n}: ERROR: {e}", file=sys.stderr)
            print("  (stopping; partial dataset will still be saved if any)",
                  file=sys.stderr)
            break

        n_new = 0
        for entry in entries:
            dt_str = entry["datetime"]
            if dt_str in seen_datetimes:
                continue
            seen_datetimes.add(dt_str)
            all_entries.append(entry)
            n_new += 1

        print(f"  chunk {chunk_n:>2}: "
              f"[{chunk_start.strftime('%Y-%m-%d %H:%M')} → "
              f"{chunk_end.strftime('%Y-%m-%d %H:%M')}]  "
              f"+{n_new:>4}  total={len(all_entries):>5}")

        if not entries:
            print("  (empty chunk; stopping)")
            break

        chunk_end = chunk_start
        time.sleep(SLEEP_BETWEEN_CALLS_S)

    if not all_entries:
        raise RuntimeError("No entries fetched. Check API plan and key.")

    all_entries.sort(key=lambda e: e["datetime"])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "month", "day", "time", "carbon"])
        for entry in all_entries:
            dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
            writer.writerow([
                dt.year,
                dt.month,
                dt.day,
                dt.strftime("%H:%M:%S"),
                entry["carbonIntensity"],
            ])

    print()
    print(f"Saved {len(all_entries)} {GRANULARITY} readings spanning")
    print(f"  {all_entries[0]['datetime']}  →  {all_entries[-1]['datetime']}")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
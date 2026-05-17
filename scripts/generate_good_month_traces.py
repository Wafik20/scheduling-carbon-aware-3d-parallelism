#!/usr/bin/env python3
"""
generate_good_month_traces.py
-----------------------------

Fetch hourly carbon-intensity history from Electricity Maps for selected
"good" months and save one continuous CSV per month.

Default months are chosen to stress carbon-aware scheduling on AWS us-west-1:

    March, April, May, September, October, November

Each output file follows the existing simulator format:

    year,month,day,time,carbon

Examples:

    python scripts/generate_good_month_traces.py

    YEAR=2024 MONTHS=4,5 python scripts/generate_good_month_traces.py

    DATA_CENTER_REGION=us-west-1 \
    DATA_CENTER_PROVIDER=aws \
    YEAR=2024 \
    MONTHS=3,4,5,9,10,11 \
    python scripts/generate_good_month_traces.py

Outputs:

    data/carbon_2024_03_hourly.csv
    data/carbon_2024_04_hourly.csv
    data/carbon_2024_05_hourly.csv
    data/carbon_2024_09_hourly.csv
    data/carbon_2024_10_hourly.csv
    data/carbon_2024_11_hourly.csv

Optionally also writes continuous seasonal traces:

    data/carbon_2024_spring_hourly.csv   # Mar-May
    data/carbon_2024_fall_hourly.csv     # Sep-Nov

Runs against EMAPS_API_KEY in .env at the project root.
"""

import os
import csv
import sys
import time
import calendar
import requests

from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


# ============================================
# LOAD ENV
# ============================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

EMAPS_API_KEY = os.getenv("EMAPS_API_KEY")
if EMAPS_API_KEY is None:
    raise ValueError("EMAPS_API_KEY not found in .env")


# ============================================
# PROJECT PATHS
# ============================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


# ============================================
# CONFIG
# ============================================

DATA_CENTER_REGION = os.getenv("DATA_CENTER_REGION", "us-west-1")
DATA_CENTER_PROVIDER = os.getenv("DATA_CENTER_PROVIDER", "aws")

YEAR = int(os.getenv("YEAR", "2024"))

# Good stress-test months for us-west-1.
# Spring usually has strong solar-driven variation.
MONTHS = [
    int(x.strip())
    for x in os.getenv("MONTHS", "3,4,5,9,10,11").split(",")
    if x.strip()
]

# past-range hourly granularity is capped at 10 days per call.
CHUNK_DAYS = int(os.getenv("CHUNK_DAYS", "10"))
GRANULARITY = os.getenv("GRANULARITY", "hourly")
SLEEP_BETWEEN_CALLS_S = float(os.getenv("SLEEP_BETWEEN_CALLS_S", "0.3"))

# Also create continuous season files for ranges with no gaps.
WRITE_SEASONAL = os.getenv("WRITE_SEASONAL", "1") not in ("0", "false", "False")

PAST_RANGE_URL = "https://api.electricitymaps.com/v3/carbon-intensity/past-range"


def month_range_utc(year: int, month: int) -> tuple[datetime, datetime]:
    """Return [start, end) UTC datetimes for a calendar month."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)

    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    return start, end


def fetch_chunk(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """One call to /past-range covering [start_dt, end_dt]."""
    params = {
        "dataCenterRegion": DATA_CENTER_REGION,
        "dataCenterProvider": DATA_CENTER_PROVIDER,
        "start": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end": end_dt.strftime("%Y-%m-%d %H:%M"),
        "granularity": GRANULARITY,
    }
    headers = {"auth-token": EMAPS_API_KEY}

    r = requests.get(PAST_RANGE_URL, params=params, headers=headers, timeout=30)

    if r.status_code in (401, 403, 405):
        raise RuntimeError(
            f"past-range returned {r.status_code}: {r.text[:300]}\n"
            "This endpoint may require a Commercial-tier API plan."
        )

    r.raise_for_status()
    body = r.json()

    return body.get("data") or body.get("history") or []


def fetch_range(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fetch a continuous range by chunking into <= CHUNK_DAYS calls."""
    print(f"Range: {start_dt.isoformat()} → {end_dt.isoformat()}")

    all_entries = []
    seen_datetimes = set()

    chunk_start = start_dt
    chunk_n = 0

    while chunk_start < end_dt:
        chunk_end = min(end_dt, chunk_start + timedelta(days=CHUNK_DAYS))
        chunk_n += 1

        try:
            entries = fetch_chunk(chunk_start, chunk_end)
        except (requests.HTTPError, RuntimeError) as e:
            print(f"  chunk {chunk_n}: ERROR: {e}", file=sys.stderr)
            print("  stopping this range; partial data will be saved if any", file=sys.stderr)
            break

        n_new = 0
        for entry in entries:
            dt_str = entry["datetime"]
            if dt_str in seen_datetimes:
                continue
            seen_datetimes.add(dt_str)
            all_entries.append(entry)
            n_new += 1

        print(
            f"  chunk {chunk_n:>2}: "
            f"[{chunk_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{chunk_end.strftime('%Y-%m-%d %H:%M')}] "
            f"+{n_new:>4} total={len(all_entries):>5}"
        )

        if not entries:
            print("  empty chunk; stopping this range")
            break

        chunk_start = chunk_end
        time.sleep(SLEEP_BETWEEN_CALLS_S)

    all_entries.sort(key=lambda e: e["datetime"])
    return all_entries


def write_entries_to_csv(entries: list[dict], output_csv: Path) -> None:
    """Write Electricity Maps entries in simulator CSV format."""
    if not entries:
        raise RuntimeError(f"No entries to write for {output_csv}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "month", "day", "time", "carbon"])

        for entry in entries:
            dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
            writer.writerow([
                dt.year,
                dt.month,
                dt.day,
                dt.strftime("%H:%M:%S"),
                entry["carbonIntensity"],
            ])

    print(f"Saved {len(entries)} readings to {output_csv}")
    print(f"  {entries[0]['datetime']} → {entries[-1]['datetime']}")


def validate_hourly_continuity(entries: list[dict], label: str) -> None:
    """Warn if the output would not satisfy load_carbon_trace hourly checks."""
    if len(entries) < 2:
        return

    dts = [
        datetime.fromisoformat(e["datetime"].replace("Z", "+00:00"))
        for e in entries
    ]

    bad_gaps = []
    for a, b in zip(dts, dts[1:]):
        gap_s = (b - a).total_seconds()
        if abs(gap_s - 3600.0) > 1e-6:
            bad_gaps.append((a, b, gap_s))
            if len(bad_gaps) >= 5:
                break

    if bad_gaps:
        print(f"[warning] {label} is not perfectly hourly.")
        for a, b, gap_s in bad_gaps:
            print(f"  gap: {a} → {b}, {gap_s / 3600.0:.2f} h")
    else:
        print(f"Validated hourly continuity for {label}")


def generate_month(year: int, month: int) -> None:
    start_dt, end_dt = month_range_utc(year, month)

    output_csv = DATA_DIR / f"carbon_{year}_{month:02d}_hourly.csv"

    print()
    print("=" * 72)
    print(f"Fetching month: {year}-{month:02d}")
    print(f"Region: {DATA_CENTER_PROVIDER}/{DATA_CENTER_REGION}")
    print(f"Output: {output_csv}")
    print("=" * 72)

    entries = fetch_range(start_dt, end_dt)
    validate_hourly_continuity(entries, f"{year}-{month:02d}")
    write_entries_to_csv(entries, output_csv)


def generate_continuous_range(name: str, start_dt: datetime, end_dt: datetime) -> None:
    output_csv = DATA_DIR / f"carbon_{YEAR}_{name}_hourly.csv"

    print()
    print("=" * 72)
    print(f"Fetching continuous range: {name}")
    print(f"Region: {DATA_CENTER_PROVIDER}/{DATA_CENTER_REGION}")
    print(f"Output: {output_csv}")
    print("=" * 72)

    entries = fetch_range(start_dt, end_dt)
    validate_hourly_continuity(entries, name)
    write_entries_to_csv(entries, output_csv)


def main():
    print("Electricity Maps carbon trace generator")
    print(f"Region:      {DATA_CENTER_PROVIDER}/{DATA_CENTER_REGION}")
    print(f"Year:        {YEAR}")
    print(f"Months:      {MONTHS}")
    print(f"Granularity: {GRANULARITY}")
    print(f"Chunk days:  {CHUNK_DAYS}")
    print(f"Endpoint:    {PAST_RANGE_URL}")

    for month in MONTHS:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month: {month}")
        generate_month(YEAR, month)

    if WRITE_SEASONAL:
        # These are continuous ranges, so they are safe for load_carbon_trace.
        spring_start = datetime(YEAR, 3, 1, tzinfo=timezone.utc)
        spring_end = datetime(YEAR, 6, 1, tzinfo=timezone.utc)

        fall_start = datetime(YEAR, 9, 1, tzinfo=timezone.utc)
        fall_end = datetime(YEAR, 12, 1, tzinfo=timezone.utc)

        generate_continuous_range("spring_hourly", spring_start, spring_end)
        generate_continuous_range("fall_hourly", fall_start, fall_end)


if __name__ == "__main__":
    main()
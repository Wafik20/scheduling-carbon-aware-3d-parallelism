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

Runs against EMAPS_API_KEY in .env at the project root. EMAPS_ZONE is required:
obtain the Electricity Maps grid-zone mapping for the intended cloud region
before fetching. Cloud provider/region values only label the output filenames.
All output names include the provider and region; each has a provenance sidecar.
"""

import os
import csv
import sys
import time
import calendar
import math
import json
import hashlib
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
# Resolve the grid zone explicitly; cloud labels are filenames, not API selectors.
EMAPS_ZONE = os.getenv("EMAPS_ZONE")
FETCH_PROVENANCE: list[dict] = []

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
    if not EMAPS_API_KEY:
        raise ValueError("EMAPS_API_KEY not found in .env")
    if not EMAPS_ZONE:
        raise ValueError("Set EMAPS_ZONE using a verified Electricity Maps mapping for the AWS region; AWS labels do not establish a grid zone")
    params = {
        "zone": EMAPS_ZONE,
        "disableCallerLookup": "true",
        "start": start_dt.strftime("%Y-%m-%d %H:%M"),
        "end": end_dt.strftime("%Y-%m-%d %H:%M"),
        "temporalGranularity": GRANULARITY,
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
    if body.get("zone") != EMAPS_ZONE:
        raise ValueError(f"Requested zone {EMAPS_ZONE!r}, received {body.get('zone')!r}")
    FETCH_PROVENANCE.append({
        "endpoint": PAST_RANGE_URL,
        "params": params,
        "response_metadata": {k: v for k, v in body.items() if k not in ("data", "history")},
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    })
    return body.get("data") or body.get("history") or []


def fetch_range(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """Fetch a continuous range by chunking into <= CHUNK_DAYS calls."""
    if not 1 <= CHUNK_DAYS <= 10:
        raise ValueError("CHUNK_DAYS must be between 1 and 10")
    if GRANULARITY != "hourly":
        raise ValueError("Simulator traces require hourly granularity")
    print(f"Range: {start_dt.isoformat()} → {end_dt.isoformat()}")

    all_entries = []
    seen_datetimes = set()

    chunk_start = start_dt
    chunk_n = 0

    while chunk_start < end_dt:
        chunk_end = min(end_dt, chunk_start + timedelta(days=CHUNK_DAYS))
        chunk_n += 1

        entries = fetch_chunk(chunk_start, chunk_end)

        n_new = 0
        for entry in entries:
            dt_str = entry["datetime"]
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if not start_dt <= dt < end_dt:
                continue
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
            raise RuntimeError(f"Empty API response for {chunk_start} to {chunk_end}")

        chunk_start = chunk_end
        time.sleep(SLEEP_BETWEEN_CALLS_S)

    all_entries.sort(key=lambda e: e["datetime"])
    expected = int((end_dt - start_dt).total_seconds() / 3600)
    if len(all_entries) != expected:
        raise ValueError(f"Expected {expected} hourly readings, got {len(all_entries)}")
    for i, entry in enumerate(all_entries):
        dt = datetime.fromisoformat(entry["datetime"].replace("Z", "+00:00"))
        if dt != start_dt + timedelta(hours=i):
            raise ValueError(f"Missing or duplicate hour at {dt}")
        value = entry.get("carbonIntensity")
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid carbon intensity at {dt}")
    return all_entries


def write_entries_to_csv(entries: list[dict], output_csv: Path) -> None:
    """Write Electricity Maps entries in simulator CSV format."""
    if not entries:
        raise RuntimeError(f"No entries to write for {output_csv}")

    temporary = output_csv.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="") as f:
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

    temporary.replace(output_csv)
    output_csv.with_suffix(".provenance.json").write_text(json.dumps({
        "sha256": hashlib.sha256(output_csv.read_bytes()).hexdigest(),
        "cloud_label": {"provider": DATA_CENTER_PROVIDER, "region": DATA_CENTER_REGION},
        "requested_zone": EMAPS_ZONE,
        "requests": FETCH_PROVENANCE,
    }, indent=2) + "\n")
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

    output_csv = DATA_DIR / f"carbon_{year}_{month:02d}_hourly_{DATA_CENTER_PROVIDER}_{DATA_CENTER_REGION}.csv"

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
    output_csv = DATA_DIR / f"carbon_{YEAR}_{name}_hourly_{DATA_CENTER_PROVIDER}_{DATA_CENTER_REGION}.csv"

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

    if sorted(MONTHS) == list(range(1, 13)):
        annual = []
        for month in range(1, 13):
            monthly = DATA_DIR / f"carbon_{YEAR}_{month:02d}_hourly_{DATA_CENTER_PROVIDER}_{DATA_CENTER_REGION}.csv"
            with monthly.open(newline="") as f:
                for row in csv.DictReader(f):
                    dt = datetime.fromisoformat(
                        f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}T{row['time']}+00:00"
                    )
                    annual.append({"datetime": dt.isoformat(), "carbonIntensity": float(row["carbon"])})
        start = datetime(YEAR, 1, 1, tzinfo=timezone.utc)
        expected = 8784 if calendar.isleap(YEAR) else 8760
        if len(annual) != expected:
            raise ValueError(f"Annual trace has {len(annual)} rows; expected {expected}")
        for i, entry in enumerate(annual):
            if datetime.fromisoformat(entry["datetime"]) != start + timedelta(hours=i):
                raise ValueError(f"Annual trace gap or duplicate at row {i}")
            if not math.isfinite(entry["carbonIntensity"]) or entry["carbonIntensity"] < 0:
                raise ValueError(f"Invalid annual carbon value at row {i}")
        output = DATA_DIR / f"carbon_{YEAR}_hourly_{DATA_CENTER_PROVIDER}_{DATA_CENTER_REGION}.csv"
        write_entries_to_csv(annual, output)

    if WRITE_SEASONAL:
        # These are continuous ranges, so they are safe for load_carbon_trace.
        spring_start = datetime(YEAR, 3, 1, tzinfo=timezone.utc)
        spring_end = datetime(YEAR, 6, 1, tzinfo=timezone.utc)

        fall_start = datetime(YEAR, 9, 1, tzinfo=timezone.utc)
        fall_end = datetime(YEAR, 12, 1, tzinfo=timezone.utc)

        generate_continuous_range("spring", spring_start, spring_end)
        generate_continuous_range("fall", fall_start, fall_end)


if __name__ == "__main__":
    main()
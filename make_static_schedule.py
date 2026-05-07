#!/usr/bin/env python3
"""
make_static_schedule.py
-----------------------
Emit a single-phase schedule JSON consumable by train_elastic.py.

train_elastic.py reads `schedule[0]["tp"]`, `schedule[0]["pp"]`, and
`schedule[-1]["end_step"]`. DP is implicit from n_gpus / (tp*pp), so we
include it in the *filename* for downstream plotting only.

Usage:
    python make_static_schedule.py --tp 1 --pp 1 --dp 4 --iters 1500 \
        --out-dir schedules/short
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--pp", type=int, required=True)
    ap.add_argument("--dp", type=int, required=True,
                    help="for filename only — train_elastic infers DP from n_gpus")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--out-dir", default="schedules/short")
    args = ap.parse_args()

    schedule = [{
        "tp": args.tp,
        "pp": args.pp,
        "start_step": 0,
        "end_step": args.iters,
    }]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"tp{args.tp}_pp{args.pp}_dp{args.dp}.json"
    path = out_dir / name
    path.write_text(json.dumps(schedule, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
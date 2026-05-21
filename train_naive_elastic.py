#!/usr/bin/env python3
"""
train_naive_elastic.py
----------------------
Minimal launcher for naive carbon-bucket elastic training.

This client does not profile or read a calibration file. The low / medium /
high strategies are passed by the caller. At launch time it writes those
inputs into a tiny scheduler JSON and runs online_elastic_pretrain.py with
`--scheduler-type naive`.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from carbon_aware_3d.compute_power import PowerMonitor
from train_online_elastic import (
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_MEGATRON_DIR,
    DEFAULT_NUM_GPUS,
    MODEL_CONFIGS,
    build_apptainer_prefix,
    build_online_elastic_cmd,
    run_with_streaming,
)


CARBON_FORECAST_PATH = (
    "/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism/"
    "data/carbon_week.csv"
)

DEFAULT_CONTAINER_BINDS = [
    "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29",
    "/bgfs/stephenlee/wat29/Megatron-LM:/opt/megatron-lm",
]
DEFAULT_WORK_DIR = (
    "/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism/"
    "runs/naive_elastic"
)
DEFAULT_BUCKET_SECONDS = 150.0
DEFAULT_DECIDE_EVERY = 500
DEFAULT_MIN_DWELL_STEPS = 0


def load_carbon_values(path):
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "carbon" not in reader.fieldnames:
            raise ValueError(f"{path}: missing 'carbon' column")
        for row in reader:
            values.append(float(row["carbon"]))
    if not values:
        raise ValueError(f"{path}: empty carbon CSV")
    return values


def bucket_thresholds(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    low_idx = max(0, min(n - 1, n // 3 - 1))
    med_idx = max(0, min(n - 1, (2 * n) // 3 - 1))
    return sorted_values[low_idx], sorted_values[med_idx]


def initial_strategy_for_csv(path, strategies):
    values = load_carbon_values(path)
    low_max, med_max = bucket_thresholds(values)
    first = values[0]
    if first <= low_max:
        return "low", strategies[0]
    if first <= med_max:
        return "medium", strategies[1]
    return "high", strategies[2]


def write_scheduler_config(work_dir, strategies, bucket_seconds,
                           decide_every, min_dwell_steps):
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "naive_scheduler_config.json")
    with open(path, "w") as f:
        json.dump({
            "strategies": strategies,
            "decide_every": decide_every,
            # `seconds_per_forecast_bucket` is no-op for the naive scheduler
            # (which uses progress-based mapping) but is still read by the
            # greedy path; kept here for forward-compat with shared configs.
            "seconds_per_forecast_bucket": bucket_seconds,
            "min_dwell_steps": min_dwell_steps,
        }, f, indent=2)
    return path


def add_strategy_args(parser, name):
    parser.add_argument(f"--{name}-tp", type=int, required=True)
    parser.add_argument(f"--{name}-pp", type=int, required=True)
    parser.add_argument(f"--{name}-dp", type=int, required=True)
    parser.add_argument(f"--{name}-power-w", type=float, default=0.0)
    parser.add_argument(f"--{name}-step-time-s", type=float, default=1.0)


def strategy_from_args(args, name):
    return {
        "tp": getattr(args, f"{name}_tp"),
        "pp": getattr(args, f"{name}_pp"),
        "dp": getattr(args, f"{name}_dp"),
        "power_w": getattr(args, f"{name}_power_w"),
        "step_time_s": getattr(args, f"{name}_step_time_s"),
    }


def validate_strategy(name, strategy, n_gpus):
    expected = strategy["tp"] * strategy["pp"] * strategy["dp"]
    if expected != n_gpus:
        raise ValueError(
            f"{name} strategy has TP*PP*DP={expected}, expected {n_gpus}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Naive low/medium/high carbon bucket elastic launcher"
    )
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--train-iters", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--container-bind", action="append", default=None)
    parser.add_argument("--megatron-dir", default=DEFAULT_MEGATRON_DIR)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--elastic-ckpt-dir", default=None)
    parser.add_argument("--carbon-forecast", default=CARBON_FORECAST_PATH)
    parser.add_argument("--bucket-seconds", type=float,
                        default=DEFAULT_BUCKET_SECONDS,
                        help="No-op for the naive scheduler (kept for "
                             "config compatibility with the greedy path).")
    parser.add_argument("--decide-every", type=int, default=DEFAULT_DECIDE_EVERY)
    parser.add_argument("--min-dwell-steps", type=int,
                        default=DEFAULT_MIN_DWELL_STEPS,
                        help="Min dwell window (steps) for naive bucket "
                             "smoothing. CI is averaged over this window "
                             "before classifying the bucket, so brief "
                             "spikes/dips don't trigger resharding. "
                             "~2x reshard_cost / step_time is a good "
                             "starting point. 0 disables smoothing.")
    add_strategy_args(parser, "low")
    add_strategy_args(parser, "med")
    add_strategy_args(parser, "high")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    cfg = MODEL_CONFIGS[args.model]
    seq_length = args.seq_length or cfg["seq_length"]
    binds = list(args.container_bind or DEFAULT_CONTAINER_BINDS)
    strategies = [
        strategy_from_args(args, "low"),
        strategy_from_args(args, "med"),
        strategy_from_args(args, "high"),
    ]
    for name, strategy in zip(("low", "med", "high"), strategies):
        validate_strategy(name, strategy, args.num_gpus)

    carbon_forecast_path = os.path.abspath(args.carbon_forecast)
    if not os.path.isfile(carbon_forecast_path):
        print(f"[fatal] carbon CSV not found: {carbon_forecast_path}",
              file=sys.stderr)
        sys.exit(2)
    if os.path.isabs(args.work_dir):
        work_bind = f"{args.work_dir}:{args.work_dir}"
        if not any(args.work_dir.startswith(b.split(":", 1)[0]) for b in binds):
            binds.append(work_bind)
    ckpt_dir = args.elastic_ckpt_dir or os.path.join(args.work_dir, "elastic_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isabs(ckpt_dir):
        ckpt_bind = f"{ckpt_dir}:{ckpt_dir}"
        if not any(ckpt_dir.startswith(b.split(":", 1)[0]) for b in binds):
            binds.append(ckpt_bind)

    scheduler_config_path = write_scheduler_config(
        args.work_dir, strategies, args.bucket_seconds,
        args.decide_every, args.min_dwell_steps,
    )
    initial_bucket, initial_strategy = initial_strategy_for_csv(
        carbon_forecast_path, strategies
    )
    first_tp = initial_strategy["tp"]
    first_pp = initial_strategy["pp"]

    print("=" * 60)
    print("  Naive Elastic 3D Parallelism Training")
    print("=" * 60)
    print(f"  Model:             {args.model}")
    print(f"  Carbon forecast:   {carbon_forecast_path}")
    print(f"  Scheduler config:  {scheduler_config_path}")
    print(f"  Train iters:       {args.train_iters}")
    for name, strategy in zip(("low", "med", "high"), strategies):
        print(f"  {name:>6} strategy:   "
              f"TP={strategy['tp']} PP={strategy['pp']} DP={strategy['dp']}")
    print(f"  Initial bucket:    {initial_bucket}")
    print(f"  Initial (TP, PP):  ({first_tp}, {first_pp})")
    print(f"  Decide every:      {args.decide_every} steps")
    print(f"  Min dwell:         {args.min_dwell_steps} steps")
    print(f"  Bucket seconds:    {args.bucket_seconds}s (unused by naive)")
    print(f"  Work dir:          {args.work_dir}")
    print(f"  Checkpoint dir:    {ckpt_dir}")

    naive_args = [
        "--elastic-ckpt-dir", ckpt_dir,
        "--scheduler-type", "naive",
        "--decide-every", str(args.decide_every),
        "--bucket-seconds", str(args.bucket_seconds),
        "--min-dwell-steps", str(args.min_dwell_steps),
    ]
    megatron_cmd = build_online_elastic_cmd(
        cfg, scheduler_config_path, carbon_forecast_path,
        args.num_gpus, args.megatron_dir, args.work_dir,
        first_tp, first_pp, args.train_iters,
        args.micro_batch_size, args.global_batch_size, seq_length,
        extra_args=naive_args + extra,
    )
    inner_cmd = f"cd {args.megatron_dir} && " + " ".join(megatron_cmd)
    full_cmd = build_apptainer_prefix(args.container_image, binds) \
        + ["bash", "-c", inner_cmd]

    logfile = os.path.join(
        args.work_dir, f"naive_elastic_{Path(args.model).stem}.log"
    )
    power_log = os.path.join(
        args.work_dir, f"naive_elastic_{Path(args.model).stem}_power.csv"
    )
    print(f"  Power log:         {power_log}")
    print("=" * 60)
    print()

    if args.dry_run:
        print("Command:\n")
        print("  " + " \\\n    ".join(full_cmd))
        return

    os.makedirs(args.work_dir, exist_ok=True)

    monitor = PowerMonitor(interval=0.5, log_file=power_log)
    monitor.start()
    t0 = time.perf_counter()
    try:
        rc, info = run_with_streaming(
            full_cmd,
            monitor,
            logfile=logfile,
            train_iters=args.train_iters,
        )
    finally:
        monitor.stop()

    wall = time.perf_counter() - t0
    avg_power, readings = monitor.results()

    print(f"\n{'=' * 60}")
    if rc == 0:
        print("  NAIVE ELASTIC TRAINING SUCCEEDED")
    else:
        print(f"  NAIVE ELASTIC TRAINING FAILED (exit {rc})")
        for e in info["errors"][:5]:
            print(f"    {e}")
    print(f"{'=' * 60}")
    print(f"  Wall time:         {wall:.1f}s")
    print(f"  Avg cluster power: {avg_power:.1f}W")
    print(f"  Power samples:     {len(readings)}")
    print(f"  Log:               {logfile}")
    print(f"  Power:             {power_log}")
    print(f"  Phases:            {os.path.join(args.work_dir, 'phases.json')}")
    print(f"  Inner summary:     {os.path.join(args.work_dir, 'elastic_results.json')}")
    print(f"{'=' * 60}\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()

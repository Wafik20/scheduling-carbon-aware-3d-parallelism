#!/usr/bin/env python3
"""
train_profile.py
----------------
Launcher for profile_strategies.py — measures (power_w, step_time_s)
for each (TP, PP, micro_batch_size) case in a profile menu and writes a
calibration JSON whose 'strategies' array can be pasted into
scheduler_config.json.

Mirrors train_online_elastic.py's apptainer + torchrun pattern; reuses
its MODEL_CONFIGS dict, local 4-GPU L40S defaults, and
build_apptainer_prefix helper.

How it works (per case in the menu):

1. RESHARD          destroy old parallel state, initialize new (TP, PP).
                    MPU-only, ~50ms.

2. BUILD            construct model + optimizer (loads checkpoint for
                    non-first configs — this is the dominant cost of the
                    BUILD step in production).

3. VERIFY           sanity-check parallel shapes per rank.

4. WARMUP iters     run train_step until step times stabilize (rolling
                    window of last `--warmup-window` iters within
                    `--warmup-tol` of their mean), bounded by
                    `--warmup-min-seconds` and `--warmup-max-seconds`.
                    Discard timings and power. GPU clocks/thermals
                    reach equilibrium and the first train_step's CUDA
                    kernel JIT / cuDNN autotune is absorbed. Without
                    this, fast configs measure low because GPUs haven't
                    ramped to steady state.

4.5 WARMUP SAVE     (first config only) discard one save_checkpoint call
                    so Megatron's dist-checkpoint subsystem, PyTorch's
                    distributed I/O paths, CUDA pinned-memory buffers,
                    and the filesystem inodes are all warm before the
                    first MEASURED transition. Without this, the first
                    real save is ~20s slower than subsequent ones —
                    one-time setup costs (sharding plan construction,
                    lazy imports, pinned-buffer allocation, fs metadata)
                    that poison the average_reshard_wall_time_s metric.
                    Must run AFTER warmup train_steps because Adam
                    lazily allocates exp_avg/exp_avg_sq on the first
                    optimizer.step(); saving before that fails with
                    KeyError('exp_avg').

5. MEASURE iters    run train_step for `--measure-seconds`, record step
                    time and power. This is what lands in the
                    calibration JSON.

6. SAVE checkpoint  write model state to disk so the NEXT config can
                    load it on its BUILD step (skipped after the last
                    config). The wall time here is the dominant component
                    of reshard_wall_time_s for the next transition.

7. FREE memory      release model from GPU before next reshard.

Steps 4.5 and 4 both exist for the same reason: discarding one-time
setup costs (checkpoint subsystem init, GPU thermal ramp-up, kernel
compile) so measurements reflect steady-state behavior. In production
this is a non-issue — real training scripts checkpoint periodically for
fault tolerance, so the dist-ckpt subsystem is already warm by the time
the scheduler triggers its first switch. The profiler is the only place
that needs to manually warm the subsystem.
"""

import argparse
import json
import os
import subprocess
import sys

from train_online_elastic import (
    DEFAULT_CONTAINER_BINDS,
    DEFAULT_CONTAINER_IMAGE,
    DEFAULT_MEGATRON_DIR,
    DEFAULT_NUM_GPUS,
    MODEL_CONFIGS,
    build_apptainer_prefix,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORK_DIR = "runs/profile"
DEFAULT_MICRO_BATCH_SIZE = 2


# ── Menu peeking ─────────────────────────────────────────────

def read_profile_menu_preview(profile_menu_path: str):
    """Pull first-case metadata. Megatron requires initial
    parallel sizes on the command line; the profiler reshards to each
    subsequent case in-process."""
    with open(profile_menu_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get('strategies', raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{profile_menu_path}: empty or malformed profile menu")
    s0 = raw[0]
    if 'tp' not in s0 or 'pp' not in s0:
        raise ValueError(
            f"{profile_menu_path}: first entry needs 'tp' and 'pp'"
        )

    first_micro_batch_size = None
    if 'micro_batch_size' in s0:
        first_micro_batch_size = int(s0['micro_batch_size'])
        if first_micro_batch_size <= 0:
            raise ValueError(
                f"{profile_menu_path}: first entry micro_batch_size "
                "must be positive"
            )

    micro_batch_sizes = sorted({
        int(s['micro_batch_size'])
        for s in raw
        if isinstance(s, dict) and 'micro_batch_size' in s
    })

    return int(s0['tp']), int(s0['pp']), first_micro_batch_size, micro_batch_sizes


# ── Command building ─────────────────────────────────────────

def build_profile_cmd(cfg, profile_menu_path, output_path,
                      n_gpus, megatron_dir, work_dir,
                      first_tp, first_pp,
                      warmup_min_seconds, warmup_max_seconds,
                      warmup_window, warmup_tol,
                      measure_seconds,
                      micro_batch, global_batch, seq_length,
                      extra_args=None):
    """Build torchrun + profile_strategies.py command. Mirrors
    build_online_elastic_cmd but swaps the inner script + arg set."""
    script = os.path.join(megatron_dir, "profile_strategies.py")

    # Megatron's LR scheduler reads --train-iters; set high enough that
    # all profiler iterations across all configs stay within bounds.
    # Profiler controls its own iteration loops, so this is just a ceiling.
    train_iters = 100000

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        script,
        "--profile-menu",         str(profile_menu_path),
        "--warmup-min-seconds",   str(warmup_min_seconds),
        "--warmup-max-seconds",   str(warmup_max_seconds),
        "--warmup-window",        str(warmup_window),
        "--warmup-tol",           str(warmup_tol),
        "--measure-seconds",      str(measure_seconds),
        "--output",               str(output_path),
        "--elastic-work-dir",     str(work_dir),
        "--tensor-model-parallel-size",   str(first_tp),
        "--pipeline-model-parallel-size", str(first_pp),
        "--num-layers",          str(cfg["num_layers"]),
        "--hidden-size",         str(cfg["hidden_size"]),
        "--num-attention-heads", str(cfg["num_attention_heads"]),
        "--seq-length",          str(seq_length),
        "--max-position-embeddings", str(seq_length),
        "--micro-batch-size",    str(micro_batch),
        "--global-batch-size",   str(global_batch),
        "--train-iters",         str(train_iters),
        "--lr",       "0.00015",
        "--min-lr",   "0.00001",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--clip-grad",     "1.0",
        "--weight-decay",  "0.1",
        "--mock-data",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size",     str(cfg["vocab_size"]),
        "--bf16",
        "--log-interval",   "100",   # quieter than online run
        "--eval-iters",     "0",
        "--eval-interval",  "1000",
    ]

    if cfg["ffn_hidden_size"] is not None:
        cmd += ["--ffn-hidden-size", str(cfg["ffn_hidden_size"])]
    if cfg["num_query_groups"] is not None:
        cmd += ["--num-query-groups", str(cfg["num_query_groups"])]

    cmd += cfg["extra_args"]
    if extra_args:
        cmd += extra_args
    return cmd


# ── Main ─────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Profile each (TP, PP, MBS) case; emit calibration JSON."
    )
    ap.add_argument('--model', required=True,
                    choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--profile-menu', required=True,
                    help='JSON listing {"tp": int, "pp": int, '
                         '"micro_batch_size": int} per case')
    ap.add_argument('--output', default=None,
                    help='Where to write calibration JSON. Defaults to '
                         'runs/calibrations/calibration_<model>_<num-gpus>gpu.json')
    ap.add_argument('--num-gpus', type=int, default=DEFAULT_NUM_GPUS)
    # Convergence-based warmup parameters
    ap.add_argument('--warmup-min-seconds', type=float, default=10.0,
                    help='Minimum warmup before convergence check fires.')
    ap.add_argument('--warmup-max-seconds', type=float, default=60.0,
                    help='Hard cap on warmup duration if step times '
                         'never stabilize.')
    ap.add_argument('--warmup-window', type=int, default=10,
                    help='Rolling-window size for step-time convergence.')
    ap.add_argument('--warmup-tol', type=float, default=0.02,
                    help='Fractional tolerance for step-time convergence '
                         '(default 0.02 = ±2%%).')
    ap.add_argument('--measure-seconds', type=float, default=60.0)
    ap.add_argument('--micro-batch-size',  type=int, default=None,
                    help='Default MBS for old-format menu entries that omit '
                         'micro_batch_size. If omitted, uses the first menu '
                         'MBS when present, otherwise 2.')
    ap.add_argument('--global-batch-size', type=int, default=16)
    ap.add_argument('--seq-length', type=int, default=None,
                    help='Override default seq_length for the model')
    ap.add_argument('--megatron-dir', default=DEFAULT_MEGATRON_DIR,
                    help='Megatron-LM path inside the container')
    ap.add_argument('--container-image',
                    default=DEFAULT_CONTAINER_IMAGE, required=not DEFAULT_CONTAINER_IMAGE)
    ap.add_argument('--container-bind', action='append', default=None)
    ap.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the Apptainer/torchrun command without running it.')
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(
            PROJECT_DIR,
            "runs", "calibrations",
            f"calibration_{args.model}_{args.num_gpus}gpu.json",
        )

    # Validate inputs early
    if not os.path.isfile(args.profile_menu):
        sys.exit(f"[fatal] --profile-menu not found: {args.profile_menu}")
    if not os.path.isfile(args.container_image):
        sys.exit(f"[fatal] --container-image not found: {args.container_image}")
    if args.warmup_min_seconds > args.warmup_max_seconds:
        sys.exit(f"[fatal] --warmup-min-seconds ({args.warmup_min_seconds}) "
                 f"> --warmup-max-seconds ({args.warmup_max_seconds})")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    cfg = MODEL_CONFIGS[args.model]
    seq_length = args.seq_length or cfg["seq_length"]
    first_tp, first_pp, first_mbs, menu_mbs_values = read_profile_menu_preview(
        args.profile_menu
    )
    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else first_mbs or DEFAULT_MICRO_BATCH_SIZE
    )

    binds = list(args.container_bind or DEFAULT_CONTAINER_BINDS)

    inner = build_profile_cmd(
        cfg=cfg,
        profile_menu_path=args.profile_menu,
        output_path=args.output,
        n_gpus=args.num_gpus,
        megatron_dir=args.megatron_dir,
        work_dir=args.work_dir,
        first_tp=first_tp, first_pp=first_pp,
        warmup_min_seconds=args.warmup_min_seconds,
        warmup_max_seconds=args.warmup_max_seconds,
        warmup_window=args.warmup_window,
        warmup_tol=args.warmup_tol,
        measure_seconds=args.measure_seconds,
        micro_batch=micro_batch_size,
        global_batch=args.global_batch_size,
        seq_length=seq_length,
    )
    full_cmd = build_apptainer_prefix(args.container_image, binds) + inner

    print('=' * 70)
    print('  PROFILER LAUNCHER')
    print('=' * 70)
    print(f"  Model:           {args.model}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Profile menu:    {args.profile_menu}")
    print(f"  Initial TP/PP:   {first_tp}/{first_pp}")
    print(f"  Initial MBS:     {first_mbs if first_mbs is not None else micro_batch_size}")
    print(f"  Default MBS:     {micro_batch_size}")
    if menu_mbs_values:
        print(f"  Menu MBS values: {menu_mbs_values}")
    else:
        print("  Menu MBS values: old format; using Default MBS for every case")
    print(f"  Global batch:    {args.global_batch_size}")
    print(f"  Seq length:      {seq_length}")
    print(f"  Container:       {args.container_image}")
    print(f"  Megatron dir:    {args.megatron_dir}")
    print(f"  Work dir:        {args.work_dir}")
    print(f"  Warmup min/max:  {args.warmup_min_seconds}s / "
          f"{args.warmup_max_seconds}s")
    print(f"  Convergence:     window={args.warmup_window} iters, "
          f"tol=±{args.warmup_tol*100:.1f}%")
    print(f"  Measure secs:    {args.measure_seconds}")
    print(f"  Output:          {args.output}")
    print('=' * 70)
    print('Full cmd:', ' '.join(full_cmd), flush=True)
    print('=' * 70, flush=True)

    if args.dry_run:
        print("Dry run requested; not launching profiler.")
        return

    rc = subprocess.call(full_cmd)
    print(f"\nProfiler exited with rc={rc}")
    if rc == 0:
        print(f"Calibration written to: {args.output}")
        print("Drop the 'strategies' array into scheduler_config.json.")
    sys.exit(rc)


if __name__ == '__main__':
    main()

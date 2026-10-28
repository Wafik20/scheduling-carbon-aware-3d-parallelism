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

Measurement, checkpoint transitions, and wraparound are implemented by
profile_strategies.py in the mounted Megatron fork. This file only builds and
launches its command; see docs/profile_rank_order.md for placement experiments.
"""

import argparse
import json
import os
from pathlib import Path
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


def configure_slurm_ranks(environ, n_gpus):
    """Use one Slurm task per GPU; LOCAL_RANK is a visible CUDA ordinal.

    Preserve SLURM_LOCALID for topology records. A singleton task mask maps to
    CUDA zero even when its token is a nonzero job-relative ID or a UUID.
    """
    rank = int(environ["SLURM_PROCID"])
    local_rank = int(environ["SLURM_LOCALID"])
    world_size = int(environ["SLURM_NTASKS"])
    if world_size != n_gpus or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Slurm ranks must match --num-gpus (total world size)")
    if not environ.get("MASTER_ADDR") or not environ.get("MASTER_PORT"):
        raise ValueError("--slurm requires MASTER_ADDR and MASTER_PORT")
    visible = [device.strip() for device in environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if device.strip()]
    cuda_rank = 0 if len(visible) == 1 else local_rank
    environ.update(RANK=str(rank), LOCAL_RANK=str(cuda_rank), WORLD_SIZE=str(world_size),
                   PROFILE_SLURM_LAUNCH="1")
    return rank


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
    ap.add_argument('--megatron-dir', default=None,
                    help='Megatron-LM path inside the container')
    ap.add_argument('--in-container', action='store_true',
                    help='Run torchrun directly inside scripts/run_container.sh.')
    ap.add_argument('--slurm', action='store_true',
                    help='One Slurm task per GPU; requires --in-container. '
                         '--num-gpus is the total world size.')
    ap.add_argument('--container-image', default=DEFAULT_CONTAINER_IMAGE)
    ap.add_argument('--container-bind', action='append', default=None)
    ap.add_argument('--work-dir', default=DEFAULT_WORK_DIR)
    ap.add_argument('--dry-run', action='store_true',
                    help='Print the Apptainer/torchrun command without running it.')
    args = ap.parse_args()
    if args.slurm and not args.in_container:
        ap.error("--slurm requires --in-container")
    rank = None
    if args.slurm:
        try:
            rank = configure_slurm_ranks(os.environ, args.num_gpus)
        except (KeyError, ValueError) as exc:
            ap.error(str(exc))
    if args.megatron_dir is None:
        args.megatron_dir = (
            '/opt/Megatron-Bridge/3rdparty/Megatron-LM'
            if args.in_container else DEFAULT_MEGATRON_DIR
        )

    if args.output is None:
        args.output = os.path.join(
            PROJECT_DIR,
            "runs", "calibrations",
            f"calibration_{args.model}_{args.num_gpus}gpu.json",
        )

    # Validate inputs early
    if not os.path.isfile(args.profile_menu):
        sys.exit(f"[fatal] --profile-menu not found: {args.profile_menu}")
    if not args.in_container and (
        not args.container_image or not os.path.isfile(args.container_image)
    ):
        sys.exit(f"[fatal] --container-image not found: {args.container_image}")
    if args.warmup_min_seconds > args.warmup_max_seconds:
        sys.exit(f"[fatal] --warmup-min-seconds ({args.warmup_min_seconds}) "
                 f"> --warmup-max-seconds ({args.warmup_max_seconds})")

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    cfg = MODEL_CONFIGS[args.model]
    seq_length = args.seq_length or cfg["seq_length"]
    first_tp, first_pp, first_mbs, _ = read_profile_menu_preview(
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
    if args.slurm:
        inner = [sys.executable, *inner[2:]]
    if args.in_container:
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
        os.environ.setdefault('PYTHONUNBUFFERED', '1')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        full_cmd = inner
    else:
        full_cmd = build_apptainer_prefix(args.container_image, binds) + inner

    if rank in (None, 0) or args.dry_run:
        print(f"Profiling {args.model}: {args.num_gpus} GPUs, menu={args.profile_menu}, "
              f"output={args.output}")
        print('Full cmd:', ' '.join(full_cmd), flush=True)

    if args.dry_run:
        print("Dry run requested; not launching profiler.")
        return

    if args.slurm:
        from scripts.nccl_builtin_ib_smoke import configure, verify_log
        from scripts.memlock_smoke import check_memlock
        check_memlock()
        nccl_log = Path(args.work_dir).resolve() / f"nccl-rank{rank}.log"
        nccl_log.unlink(missing_ok=True)
        configure(nccl_log)
    rc = subprocess.call(full_cmd)
    if args.slurm and rc == 0:
        verify_log(nccl_log)
    print(f"\nProfiler exited with rc={rc}")
    if rc == 0:
        print(f"Calibration written to: {args.output}")
        print("Drop the 'strategies' array into scheduler_config.json.")
    sys.exit(rc)


if __name__ == '__main__':
    main()

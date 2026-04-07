#!/usr/bin/env python3
"""
train.py
--------
Unified Megatron training script with power monitoring.

Two modes:
  1. Single config:  python train.py --model gpt2 --tp 1 --pp 1
  2. Schedule:        python train.py --model gpt2 --schedule schedule.json

Schedule mode runs multiple phases with different (TP, PP) configs,
checkpointing between phases with automatic resharding.

Example schedule.json:
[
    {"phase": 1, "start_step": 0,   "end_step": 100, "tp": 2, "pp": 2},
    {"phase": 2, "start_step": 100, "end_step": 200, "tp": 1, "pp": 1},
    {"phase": 3, "start_step": 200, "end_step": 300, "tp": 4, "pp": 1}
]

DP is inferred: DP = N / (TP * PP)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from compute_power import PowerMonitor

# ── Cluster defaults ─────────────────────────────────────────
DEFAULT_NUM_GPUS = 4
DEFAULT_CONTAINER_IMAGE = "/u/data3/wat29/nemo.sif"
DEFAULT_CONTAINER_BINDS = [
    "/u/data3/wat29:/u/data3/wat29",
    "/u/data3/wat29/Megatron-LM:/opt/megatron-lm",
]
DEFAULT_MEGATRON_DIR = "/opt/megatron-lm"
DEFAULT_WORK_DIR = "/u/data3/wat29/carbon_training"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = PROJECT_DIR

_ITER_RE = re.compile(r"iteration\s+(\d+)/")

# ── Model configs ────────────────────────────────────────────
MODEL_CONFIGS = {
    "gpt2": {
        "num_layers": 12, "hidden_size": 768, "ffn_hidden_size": None,
        "num_attention_heads": 12, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50257, "extra_args": [],
    },
    "gpt2_medium": {
        "num_layers": 24, "hidden_size": 1024, "ffn_hidden_size": None,
        "num_attention_heads": 16, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50257, "extra_args": [],
    },
    "gpt2_large": {
        "num_layers": 36, "hidden_size": 1280, "ffn_hidden_size": None,
        "num_attention_heads": 20, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50257, "extra_args": [],
    },
    "llama_1b": {
        "num_layers": 16, "hidden_size": 2048, "ffn_hidden_size": 8192,
        "num_attention_heads": 32, "num_query_groups": 8,
        "seq_length": 256, "vocab_size": 128256,
        "extra_args": [
            "--group-query-attention",
            "--position-embedding-type", "rope",
            "--rotary-base", "500000", "--rotary-percent", "1.0",
            "--swiglu", "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
            "--untie-embeddings-and-output-weights",
            "--use-distributed-optimizer",
            "--transformer-impl", "local",
            "--no-persist-layer-norm",
            "--no-gradient-accumulation-fusion",
        ],
    },
    "llama_3b": {
        "num_layers": 28, "hidden_size": 3072, "ffn_hidden_size": 8192,
        "num_attention_heads": 24, "num_query_groups": 8,
        "seq_length": 256, "vocab_size": 128256,
        "extra_args": [
            "--group-query-attention",
            "--position-embedding-type", "rope",
            "--rotary-base", "500000", "--rotary-percent", "1.0",
            "--swiglu", "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
            "--untie-embeddings-and-output-weights",
            "--use-distributed-optimizer",
            "--transformer-impl", "local",
            "--no-persist-layer-norm",
            "--no-gradient-accumulation-fusion",
        ],
    },
}


# ── Schedule loading ─────────────────────────────────────────

def load_schedule(path, n_gpus, cfg):
    """Load and validate a schedule JSON. Returns list of phase dicts."""
    with open(path) as f:
        raw = json.load(f)

    phases = []
    for i, entry in enumerate(raw):
        tp, pp = entry["tp"], entry["pp"]
        assert n_gpus % (tp * pp) == 0, \
            f"Phase {i+1}: TP={tp}*PP={pp} does not divide N={n_gpus}"
        dp = n_gpus // (tp * pp)
        assert cfg["num_layers"] % pp == 0, \
            f"Phase {i+1}: {cfg['num_layers']} layers not divisible by PP={pp}"
        assert cfg["num_attention_heads"] % tp == 0, \
            f"Phase {i+1}: {cfg['num_attention_heads']} heads not divisible by TP={tp}"
        if cfg["num_query_groups"] is not None:
            assert cfg["num_query_groups"] % tp == 0, \
                f"Phase {i+1}: {cfg['num_query_groups']} query groups not divisible by TP={tp}"

        phases.append({
            "phase": entry.get("phase", i + 1),
            "start_step": entry["start_step"],
            "end_step": entry["end_step"],
            "tp": tp, "pp": pp, "dp": dp,
        })

    # Validate continuity
    for i in range(1, len(phases)):
        assert phases[i]["start_step"] == phases[i-1]["end_step"], \
            f"Gap between phase {phases[i-1]['phase']} and {phases[i]['phase']}"

    return phases


# ── Command building ─────────────────────────────────────────

def build_apptainer_prefix(container_image, container_binds):
    prefix = ["apptainer", "exec", "--nv"]
    for bind in container_binds:
        prefix += ["--bind", bind]
    prefix += [
        "--env", "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "--env", "CUDA_HOME=/usr/local/cuda",
        container_image,
    ]
    return prefix


def build_megatron_cmd(cfg, tp, pp, n_gpus, megatron_dir,
                       train_iters, micro_batch, global_batch,
                       seq_length, model_name,
                       save_dir=None, load_dir=None,
                       save_interval=None, is_schedule=False,
                       extra_args=None):
    """Build torchrun + pretrain_gpt.py command (no apptainer prefix)."""
    pretrain_script = os.path.join(megatron_dir, "pretrain_gpt.py")

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        pretrain_script,
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
        "--num-layers", str(cfg["num_layers"]),
        "--hidden-size", str(cfg["hidden_size"]),
        "--num-attention-heads", str(cfg["num_attention_heads"]),
        "--seq-length", str(seq_length),
        "--max-position-embeddings", str(seq_length),
        "--micro-batch-size", str(micro_batch),
        "--global-batch-size", str(global_batch),
        "--train-iters", str(train_iters),
        "--lr", "0.00015",
        "--min-lr", "0.00001",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--clip-grad", "1.0",
        "--weight-decay", "0.1",
        "--mock-data",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(cfg["vocab_size"]),
        "--split", "99,1,0",
        "--log-interval", "10",
        "--eval-iters", "0",
        "--eval-interval", "1000",
        "--bf16",
    ]

    if cfg["ffn_hidden_size"] is not None:
        cmd += ["--ffn-hidden-size", str(cfg["ffn_hidden_size"])]
    if cfg["num_query_groups"] is not None:
        cmd += ["--num-query-groups", str(cfg["num_query_groups"])]

    # Activation recomputation for large models
    dp = n_gpus // (tp * pp)
    if model_name in ("llama_1b", "llama_3b"):
        cmd += [
            "--recompute-granularity", "full",
            "--recompute-method", "uniform",
            "--recompute-num-layers", str(cfg["num_layers"] // pp),
        ]

    cmd += cfg["extra_args"]

    # Schedule mode: checkpoint resharding support
    if is_schedule:
        cmd += [
            "--ckpt-format", "torch_dist",
            "--use-distributed-optimizer",
            "--dist-ckpt-optim-fully-reshardable",
            "--initial-loss-scale", "1024",
        ]
        if save_dir:
            cmd += ["--save", str(save_dir)]
        if save_interval:
            cmd += ["--save-interval", str(save_interval)]
        if load_dir:
            cmd += [
                "--load", str(load_dir),
                "--auto-detect-ckpt-format",
                "--override-opt_param-scheduler",
            ]

    if extra_args:
        cmd += extra_args

    return cmd


# ── Phase execution with streaming ───────────────────────────

def run_with_streaming(cmd, monitor, logfile=None):
    """
    Run a command, streaming stdout to terminal + optional logfile.
    Parses Megatron iteration lines to feed PowerMonitor.
    Returns (returncode, info_dict).
    """
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    info = {"errors": [], "iter_ms": [], "losses": []}

    lf = open(logfile, "w") if logfile else None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for line in proc.stdout:
            sys.stdout.write(line)
            if lf:
                lf.write(line)
                lf.flush()

            # Feed step to power monitor
            m = _ITER_RE.search(line)
            if m:
                monitor.update_step(int(m.group(1)))

            # Parse metrics
            if "after megatron is initialized" in line:
                m2 = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
                if m2:
                    info["init_time"] = datetime.strptime(m2.group(1), fmt)

            if "successfully loaded checkpoint" in line:
                m2 = re.search(r"at iteration (\d+)", line)
                if m2:
                    info["resumed_iter"] = int(m2.group(1))

            if "before the start of training step" in line:
                m2 = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
                if m2:
                    info["train_start"] = datetime.strptime(m2.group(1), fmt)

            if "after training is done" in line:
                m2 = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
                if m2:
                    info["train_end"] = datetime.strptime(m2.group(1), fmt)

            if "elapsed time per iteration" in line:
                m2 = re.search(r"elapsed time per iteration \(ms\): ([\d.]+)", line)
                if m2:
                    info["iter_ms"].append(float(m2.group(1)))

            if "lm loss:" in line:
                m2 = re.search(r"lm loss: ([\d.E+\-]+)", line)
                if m2:
                    info["losses"].append(float(m2.group(1)))

            for pat in ["RuntimeError:", "AssertionError:", "KeyError:", "TypeError:"]:
                if pat in line and "error_injection" not in line and "import" not in line:
                    info["errors"].append(line.strip())

        proc.wait()
    finally:
        if lf:
            lf.close()

    # Derived timing
    if "init_time" in info and "train_start" in info:
        info["reshard_s"] = (info["train_start"] - info["init_time"]).total_seconds()
    if "train_start" in info and "train_end" in info:
        info["train_s"] = (info["train_end"] - info["train_start"]).total_seconds()
    if info["iter_ms"]:
        info["avg_iter_ms"] = sum(info["iter_ms"]) / len(info["iter_ms"])

    return proc.returncode, info


# ── Single-config mode ───────────────────────────────────────

def run_single(args, cfg, container_binds, extra_megatron_args):
    """Original behavior: one config, one run."""
    TP, PP = args.tp, args.pp
    DP = args.dp if args.dp is not None else args.num_gpus // (TP * PP)
    assert TP * PP * DP == args.num_gpus
    seq_length = args.seq_length or cfg["seq_length"]

    log_dir = os.path.join(PROJECT_DIR, "power_logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    power_log = os.path.join(log_dir, f"tp{TP}_pp{PP}_dp{DP}_power.csv")

    tag = f"{args.model} TP={TP} PP={PP} DP={DP}"
    print(f"\n{'='*50}")
    print(f"  Training: {tag}")
    print(f"  GPUs:     {args.num_gpus}")
    print(f"  Image:    {args.container_image}")
    print(f"  Log:      {power_log}")
    print(f"{'='*50}\n")

    megatron_cmd = build_megatron_cmd(
        cfg, TP, PP, args.num_gpus, args.megatron_dir,
        args.train_iters, args.micro_batch_size, args.global_batch_size,
        seq_length, args.model, extra_args=extra_megatron_args,
    )
    cmd = build_apptainer_prefix(args.container_image, container_binds) + megatron_cmd

    if args.dry_run:
        print("Command:\n")
        print("  " + " \\\n    ".join(cmd))
        return

    monitor = PowerMonitor(interval=0.5, log_file=power_log)
    monitor.start()

    try:
        rc, info = run_with_streaming(cmd, monitor)
        if rc != 0:
            print(f"Warning: training exited with code {rc}", file=sys.stderr)
    finally:
        monitor.stop()

    avg_power, readings = monitor.results()
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"  Avg cluster power : {avg_power:.2f} W")
    print(f"  Samples collected : {len(readings)}")
    print(f"  Log saved to      : {power_log}")
    print(f"{'='*50}\n")


# ── Schedule mode ────────────────────────────────────────────

def run_schedule(args, cfg, container_binds, extra_megatron_args):
    """Multi-phase: different (TP, PP) per phase with checkpoint resharding."""

    # If we're on the host, re-launch ourselves inside the container
    if not args.inside_container:
        relaunch_in_container(args, container_binds)
        return

    # ── We are inside the container. Run phases directly. ────
    phases = load_schedule(args.schedule, args.num_gpus, cfg)
    seq_length = args.seq_length or cfg["seq_length"]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    schedule_name = Path(args.schedule).stem
    power_log = os.path.join(RESULTS_DIR, "power_logs", args.model,
                             f"schedule_{schedule_name}_power.csv")
    os.makedirs(os.path.dirname(power_log), exist_ok=True)

    print("=" * 60)
    print("Carbon-Aware 3D Parallelism Training (inside container)")
    print(f"Model: {args.model} | GPUs: {args.num_gpus} | Phases: {len(phases)}")
    print(f"Steps: {phases[0]['start_step']} → {phases[-1]['end_step']}")
    print("=" * 60)
    for p in phases:
        print(f"  Phase {p['phase']}: steps {p['start_step']}→{p['end_step']} "
              f"(DP={p['dp']}, TP={p['tp']}, PP={p['pp']})")
    print()

    if args.dry_run:
        print("Dry run — exiting.")
        return

    monitor = PowerMonitor(interval=0.5, log_file=power_log)
    monitor.start()

    run_log = []
    prev_ckpt_dir = None

    try:
        for i, phase in enumerate(phases):
            tp, pp, dp = phase["tp"], phase["pp"], phase["dp"]
            phase_name = f"phase{phase['phase']}_dp{dp}_tp{tp}_pp{pp}"

            print(f"\n{'='*60}")
            print(f"PHASE {phase['phase']}: steps {phase['start_step']}→{phase['end_step']} "
                  f"(DP={dp}, TP={tp}, PP={pp})")
            print(f"{'='*60}")

            save_dir = work_dir / phase_name
            save_dir.mkdir(parents=True, exist_ok=True)
            logfile = work_dir / f"{phase_name}.log"

            is_reshard = (prev_ckpt_dir is not None and i > 0 and
                          (phase["tp"] != phases[i-1]["tp"] or
                           phase["pp"] != phases[i-1]["pp"]))

            if is_reshard:
                print(f"  Config change → will reshard checkpoint")
            elif prev_ckpt_dir:
                print(f"  Same config → resuming")
            else:
                print(f"  Starting from scratch")

            phase_steps = phase["end_step"] - phase["start_step"]
            megatron_cmd = build_megatron_cmd(
                cfg, tp, pp, args.num_gpus, args.megatron_dir,
                train_iters=phase["end_step"],
                micro_batch=args.micro_batch_size,
                global_batch=args.global_batch_size,
                seq_length=seq_length,
                model_name=args.model,
                save_dir=save_dir,
                load_dir=prev_ckpt_dir,
                save_interval=phase_steps,
                is_schedule=True,
                extra_args=extra_megatron_args,
            )
            # No apptainer prefix — we're already inside the container
            cmd = megatron_cmd

            t0 = time.perf_counter()
            rc, info = run_with_streaming(cmd, monitor, logfile=str(logfile))
            wall = time.perf_counter() - t0

            if rc != 0:
                print(f"\n  FAILED (exit {rc})")
                for e in info["errors"][:3]:
                    print(f"    {e}")
                print(f"  Log: {logfile}")
                print(f"\nAborting schedule.")
                break

            reshard_s = info.get("reshard_s", 0)
            train_s = info.get("train_s", 0)
            avg_ms = info.get("avg_iter_ms", 0)
            final_loss = info["losses"][-1] if info.get("losses") else None

            print(f"\n  Timing:  wall={wall:.1f}s | reshard={reshard_s:.1f}s | "
                  f"train={train_s:.1f}s | iter={avg_ms:.0f}ms")
            if final_loss:
                print(f"  Loss:    {final_loss:.4f}")

            run_log.append({
                "phase": phase["phase"], "config": phase_name,
                "dp": dp, "tp": tp, "pp": pp,
                "start_step": phase["start_step"], "end_step": phase["end_step"],
                "wall_s": round(wall, 2), "reshard_s": round(reshard_s, 2),
                "train_s": round(train_s, 2),
                "avg_iter_ms": round(avg_ms, 2),
                "final_loss": round(final_loss, 6) if final_loss else None,
                "is_reshard": is_reshard,
            })

            prev_ckpt_dir = save_dir

    finally:
        monitor.stop()

    avg_power, readings = monitor.results()

    # Save per-phase results
    results_file = os.path.join(RESULTS_DIR, f"schedule_{schedule_name}_results.csv")
    if run_log:
        fields = list(run_log[0].keys())
        with open(results_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(run_log)

    # Summary
    if run_log:
        total_wall = sum(r["wall_s"] for r in run_log)
        total_reshard = sum(r["reshard_s"] for r in run_log)
        total_train = sum(r["train_s"] for r in run_log)

        print(f"\n{'='*60}")
        print("SCHEDULE COMPLETE")
        print(f"{'='*60}")
        print(f"  Total wall:      {total_wall:.1f}s")
        print(f"  Total training:  {total_train:.1f}s")
        print(f"  Total reshard:   {total_reshard:.1f}s")
        print(f"  Reshard %:       {total_reshard/total_wall*100:.1f}%")
        print(f"  Avg power:       {avg_power:.1f} W")
        print(f"  Power samples:   {len(readings)}")

        print(f"\n  {'Phase':<6} {'Config':<28} {'Train':>8} {'Iter':>8} {'Loss':>10}")
        print("  " + "-" * 64)
        for r in run_log:
            loss = f"{r['final_loss']:.4f}" if r['final_loss'] else "N/A"
            print(f"  {r['phase']:<6} {r['config']:<28} "
                  f"{r['train_s']:>7.1f}s {r['avg_iter_ms']:>7.0f}ms {loss:>10}")

        print(f"\n  Results: {results_file}")
        print(f"  Power:   {power_log}")


def relaunch_in_container(args, container_binds):
    """Re-launch this script inside the container with --inside-container."""
    script_path = os.path.abspath(__file__)

    apptainer_cmd = ["apptainer", "exec", "--nv"]
    for bind in container_binds:
        apptainer_cmd += ["--bind", bind]
    # Bind the script's directory and AFS
    apptainer_cmd += ["--bind", f"{PROJECT_DIR}:{PROJECT_DIR}"]
    apptainer_cmd += ["--bind", "/afs:/afs"]
    apptainer_cmd += [
        "--env", "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "--env", "CUDA_HOME=/usr/local/cuda",
        args.container_image,
    ]

    # Rebuild the inner command with --inside-container
    inner_cmd = [
        "python", script_path,
        "--model", args.model,
        "--schedule", args.schedule,
        "--num-gpus", str(args.num_gpus),
        "--megatron-dir", args.megatron_dir,
        "--work-dir", args.work_dir,
        "--inside-container",
    ]
    if args.micro_batch_size != 2:
        inner_cmd += ["--micro-batch-size", str(args.micro_batch_size)]
    if args.global_batch_size != 16:
        inner_cmd += ["--global-batch-size", str(args.global_batch_size)]
    if args.seq_length:
        inner_cmd += ["--seq-length", str(args.seq_length)]
    if args.dry_run:
        inner_cmd += ["--dry-run"]

    full_cmd = apptainer_cmd + inner_cmd

    print(f"Launching single container for entire schedule...")
    print(f"  Container: {args.container_image}")
    print()

    if args.dry_run:
        print("Command:\n")
        print("  " + " \\\n    ".join(full_cmd))
        return

    result = subprocess.run(full_cmd)
    sys.exit(result.returncode)


# ── Argument parsing ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified Megatron training with power monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))

    # Mode: single config OR schedule
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--schedule", type=str, default=None,
                      help="Path to schedule JSON (multi-phase mode)")

    # Single-config parallelism (ignored if --schedule is used)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=None)

    # Training
    parser.add_argument("--train-iters", type=int, default=200)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=None)

    # Cluster / container
    parser.add_argument("--num-gpus", type=int,
                        default=int(os.environ.get("NUM_GPUS", DEFAULT_NUM_GPUS)))
    parser.add_argument("--container-image", type=str,
                        default=os.environ.get("CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE))
    parser.add_argument("--container-bind", type=str, action="append", default=None)
    parser.add_argument("--megatron-dir", type=str,
                        default=os.environ.get("MEGATRON_DIR", DEFAULT_MEGATRON_DIR))
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inside-container", action="store_true",
                        help="Internal flag: indicates we are already inside the container")

    args, extra = parser.parse_known_args()

    cfg = MODEL_CONFIGS[args.model]

    # Resolve container binds
    if args.container_bind:
        binds = args.container_bind
    elif os.environ.get("CONTAINER_BINDS"):
        binds = os.environ["CONTAINER_BINDS"].split(",")
    else:
        binds = DEFAULT_CONTAINER_BINDS

    if args.schedule:
        run_schedule(args, cfg, binds, extra)
    else:
        run_single(args, cfg, binds, extra)


if __name__ == "__main__":
    main()
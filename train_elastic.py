#!/usr/bin/env python3
"""
train_elastic.py
----------------
Launcher for elastic parallelism training with power monitoring.

Calls elastic_pretrain.py (in Megatron repo) via torchrun inside
the Apptainer container, wrapping with PowerMonitor for energy tracking.

Usage:
    python train_elastic.py --model gpt2 --schedule elastic_schedule.json
    python train_elastic.py --model gpt2 --schedule elastic_schedule.json --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
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
DEFAULT_WORK_DIR = "/u/data3/wat29/carbon_training/elastic"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = PROJECT_DIR

# ── Regex patterns for extracting the current training step ──
#
# elastic_pretrain.py prints a per-step line with the GLOBAL iteration, e.g.:
#     "    iter    510/1000 | loss: 10.85 | grad_norm: 0.0595"
# The number on the left is already the global step, so we do NOT add any
# phase offset — doing so would give 1010 instead of 510 at phase 2 start.
#
# The \b before `iter` avoids false matches on words like
# `iterations_to_skip`, `grpo_iterations`, etc. in the argument dump.
_ITER_RE   = re.compile(r"\biter\s+(\d+)\s*/\s*\d+")
_STEP_RE   = re.compile(r"\[STEP\]\s+(\d+)")  # optional explicit heartbeat
_CKPT_RE   = re.compile(r"saving checkpoint at iteration\s+(\d+)", re.IGNORECASE)
_RESUME_RE = re.compile(r"Resuming from iteration:\s+(\d+)", re.IGNORECASE)
_STEP_PATTERNS = (_STEP_RE, _ITER_RE, _CKPT_RE, _RESUME_RE)


# ── Model configs (must match elastic_pretrain.py expectations) ──
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
}


# ── Command building ─────────────────────────────────────────

def build_elastic_cmd(cfg, schedule_path, n_gpus, megatron_dir, work_dir,
                      first_tp, first_pp, total_iters,
                      micro_batch, global_batch, seq_length, extra_args=None):
    """Build the torchrun + elastic_pretrain.py command."""
    script = os.path.join(megatron_dir, "elastic_pretrain.py")

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        script,
        "--elastic-schedule", str(schedule_path),
        "--elastic-work-dir", str(work_dir),
        "--tensor-model-parallel-size", str(first_tp),
        "--pipeline-model-parallel-size", str(first_pp),
        "--num-layers", str(cfg["num_layers"]),
        "--hidden-size", str(cfg["hidden_size"]),
        "--num-attention-heads", str(cfg["num_attention_heads"]),
        "--seq-length", str(seq_length),
        "--max-position-embeddings", str(seq_length),
        "--micro-batch-size", str(micro_batch),
        "--global-batch-size", str(global_batch),
        "--train-iters", str(total_iters),
        "--lr", "0.00015",
        "--min-lr", "0.00001",
        "--lr-decay-style", "cosine",
        "--lr-warmup-fraction", "0.01",
        "--clip-grad", "1.0",
        "--weight-decay", "0.1",
        "--mock-data",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(cfg["vocab_size"]),
        "--bf16",
        "--log-interval", "10",
        "--eval-iters", "0",
        "--eval-interval", "1000",
    ]

    if cfg["ffn_hidden_size"] is not None:
        cmd += ["--ffn-hidden-size", str(cfg["ffn_hidden_size"])]
    if cfg["num_query_groups"] is not None:
        cmd += ["--num-query-groups", str(cfg["num_query_groups"])]

    cmd += cfg["extra_args"]

    if extra_args:
        cmd += extra_args

    return cmd


def build_apptainer_prefix(container_image, container_binds):
    prefix = ["apptainer", "exec", "--nv"]
    for bind in container_binds:
        prefix += ["--bind", bind]
    prefix += [
        "--env", "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "--env", "CUDA_HOME=/usr/local/cuda",
        "--env", "PYTHONUNBUFFERED=1",
        container_image,
    ]
    return prefix


# ── Step tracking ────────────────────────────────────────────

def _extract_step(line):
    """Return the global step from a log line, or None.

    Tries patterns in priority order. First match wins. The monotonic
    clamp in run_with_streaming prevents the final checkpoint-save line
    (which echoes 'saving checkpoint at iteration 1000') from rewinding
    the counter below already-reported higher training steps.
    """
    for pat in _STEP_PATTERNS:
        m = pat.search(line)
        if m:
            return int(m.group(1))
    return None


# ── Streaming execution ──────────────────────────────────────

def run_with_streaming(cmd, monitor, logfile=None):
    """Run command, streaming output and feeding PowerMonitor."""
    info = {"errors": [], "iter_ms": [], "losses": []}
    max_step_seen = 0  # monotonic: never rewind

    lf = open(logfile, "w") if logfile else None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        for line in iter(proc.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            if lf:
                lf.write(line)
                lf.flush()

            step = _extract_step(line)
            if step is not None:
                # Clamp to monotonic. Checkpoint-save lines at the end of a
                # phase echo the starting step of the phase we just finished,
                # which would otherwise make the power CSV appear to rewind.
                if step >= max_step_seen:
                    max_step_seen = step
                    try:
                        monitor.update_step(step)
                    except Exception as e:
                        print(f"[train_elastic] update_step failed: {e}",
                              file=sys.stderr, flush=True)

            if "elapsed time per iteration" in line:
                m2 = re.search(r"elapsed time per iteration \(ms\): ([\d.]+)", line)
                if m2:
                    info["iter_ms"].append(float(m2.group(1)))

            m_loss = re.search(r"\|\s*loss:\s*([\d.eE+\-]+)", line)
            if m_loss:
                info["losses"].append(float(m_loss.group(1)))
            elif "lm loss:" in line:
                m2 = re.search(r"lm loss: ([\d.E+\-]+)", line)
                if m2:
                    info["losses"].append(float(m2.group(1)))

            for pat in ["RuntimeError:", "AssertionError:", "KeyError:",
                        "TypeError:", "ModuleNotFoundError:"]:
                if pat in line and "error_injection" not in line and "import" not in line:
                    info["errors"].append(line.strip())

        proc.wait()
    finally:
        if lf:
            lf.close()

    if info["iter_ms"]:
        info["avg_iter_ms"] = sum(info["iter_ms"]) / len(info["iter_ms"])

    return proc.returncode, info


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Elastic parallelism training with power monitoring",
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--schedule", required=True, help="Path to schedule JSON")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--container-image", type=str, default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--container-bind", type=str, action="append", default=None)
    parser.add_argument("--megatron-dir", type=str, default=DEFAULT_MEGATRON_DIR)
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dry-run", action="store_true")

    args, extra = parser.parse_known_args()
    cfg = MODEL_CONFIGS[args.model]
    seq_length = args.seq_length or cfg["seq_length"]

    binds = args.container_bind or DEFAULT_CONTAINER_BINDS

    schedule_path = os.path.abspath(args.schedule)
    with open(schedule_path) as f:
        schedule = json.load(f)

    first_tp = schedule[0]["tp"]
    first_pp = schedule[0]["pp"]
    total_iters = schedule[-1]["end_step"]

    schedule_name = Path(args.schedule).stem
    log_dir = os.path.join(PROJECT_DIR, "power_logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    power_log = os.path.join(log_dir, f"elastic_{schedule_name}_power.csv")

    print("=" * 60)
    print("  Elastic 3D Parallelism Training")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Schedule:   {schedule_path}")
    print(f"  GPUs:       {args.num_gpus}")
    print(f"  Phases:     {len(schedule)}")
    print(f"  Steps:      {schedule[0]['start_step']} -> {schedule[-1]['end_step']}")
    print(f"  Container:  {args.container_image}")
    print(f"  Work dir:   {args.work_dir}")
    print(f"  Power log:  {power_log}")
    print("=" * 60)
    print()

    megatron_cmd = build_elastic_cmd(
        cfg, schedule_path, args.num_gpus, args.megatron_dir, args.work_dir,
        first_tp, first_pp, total_iters,
        args.micro_batch_size, args.global_batch_size, seq_length,
        extra_args=extra,
    )
    inner_cmd = f"cd {args.megatron_dir} && " + " ".join(megatron_cmd)
    full_cmd = build_apptainer_prefix(args.container_image, binds) + ["bash", "-c", inner_cmd]

    if args.dry_run:
        print("Command:\n")
        print("  " + " \\\n    ".join(full_cmd))
        return

    monitor = PowerMonitor(interval=0.5, log_file=power_log)
    monitor.start()

    logfile = os.path.join(args.work_dir, f"elastic_{schedule_name}.log")
    os.makedirs(args.work_dir, exist_ok=True)

    t0 = time.perf_counter()
    try:
        rc, info = run_with_streaming(full_cmd, monitor, logfile=logfile)
    finally:
        monitor.stop()

    wall = time.perf_counter() - t0
    avg_power, readings = monitor.results()

    print(f"\n{'=' * 60}")
    if rc == 0:
        print("  ELASTIC TRAINING SUCCEEDED")
    else:
        print(f"  ELASTIC TRAINING FAILED (exit {rc})")
        for e in info["errors"][:5]:
            print(f"    {e}")
    print(f"{'=' * 60}")
    print(f"  Wall time:        {wall:.1f}s")
    print(f"  Avg cluster power: {avg_power:.1f}W")
    print(f"  Power samples:    {len(readings)}")
    print(f"  Log:              {logfile}")
    print(f"  Power:            {power_log}")
    print(f"{'=' * 60}\n")

    sys.exit(rc)


if __name__ == "__main__":
    main()
"""
train_gpt2.py
-------------
Runs Megatron GPT-2 training with power monitoring across
different 3D parallelism configurations. Uses mock data
and random init for power benchmarking.

Usage:
    apptainer run --nv \
        --bind /u/data3/wat29:/u/data3/wat29 \
        --bind /u/data3/wat29/Megatron-LM:/opt/megatron-lm \
        /u/data3/wat29/nemo.sif \
        python train_gpt2.py --tp 1 --pp 1 --dp 4

Assumes 4 GPUs total (TP × PP × DP must equal 4).
"""

import argparse
import os
import subprocess
import sys

from compute_power import PowerMonitor

# ── Environment setup ─────────────────────────────────────────────────
os.environ.setdefault("CUDA_HOME", "/usr")
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # Required for TP

# ── Constants ─────────────────────────────────────────────────────────
NUM_GPUS = 4
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT = "/opt/megatron-lm/pretrain_gpt.py"

# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GPT-2 training with power monitoring")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
parser.add_argument("--dp", type=int, default=4, help="Data parallelism degree")
parser.add_argument("--train-iters", type=int, default=50, help="Training iterations")
parser.add_argument("--micro-batch-size", type=int, default=2, help="Micro batch size")
parser.add_argument("--global-batch-size", type=int, default=16, help="Global batch size")
args = parser.parse_args()

TP, PP, DP = args.tp, args.pp, args.dp
assert TP * PP * DP == NUM_GPUS, (
    f"TP({TP}) × PP({PP}) × DP({DP}) = {TP*PP*DP}, expected {NUM_GPUS}"
)

# ── Paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_DIR, "power_logs", "gpt2")
LOG_FILE = os.path.join(RESULTS_DIR, f"tp{TP}_pp{PP}_dp{DP}_power.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Build Megatron command ────────────────────────────────────────────
cmd = [
    "torchrun",
    f"--nproc_per_node={NUM_GPUS}",
    PRETRAIN_SCRIPT,
    # Parallelism
    "--tensor-model-parallel-size", str(TP),
    "--pipeline-model-parallel-size", str(PP),
    # GPT-2 Small (124M)
    "--num-layers", "12",
    "--hidden-size", "768",
    "--num-attention-heads", "12",
    "--seq-length", "1024",
    "--max-position-embeddings", "1024",
    # Training
    "--micro-batch-size", str(args.micro_batch_size),
    "--global-batch-size", str(args.global_batch_size),
    "--train-iters", str(args.train_iters),
    "--lr", "0.0001",
    "--min-lr", "0.00001",
    "--lr-decay-style", "cosine",
    "--lr-warmup-fraction", "0.01",
    "--clip-grad", "1.0",
    "--weight-decay", "0.1",
    # Mock data (no real dataset needed for power benchmarking)
    "--mock-data",
    "--tokenizer-type", "NullTokenizer",
    "--vocab-size", "50257",
    "--split", "99,1,0",
    # Logging
    "--log-interval", "10",
    "--eval-iters", "5",
    "--eval-interval", "50",
    # Precision
    "--bf16",
]

# ── Run with power monitoring ─────────────────────────────────────────
tag = f"gpt2 TP={TP} PP={PP} DP={DP}"
print(f"\n{'='*50}")
print(f"  Training: {tag}")
print(f"  Log:      {LOG_FILE}")
print(f"{'='*50}\n")

monitor = PowerMonitor(interval=0.5, log_file=LOG_FILE)
monitor.start()

try:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Warning: training exited with code {result.returncode}", file=sys.stderr)
finally:
    monitor.stop()

# ── Report ────────────────────────────────────────────────────────────
avg_power, readings = monitor.results()

print(f"\n{'='*50}")
print(f"  {tag}")
print(f"  Avg cluster power : {avg_power:.2f} W")
print(f"  Samples collected : {len(readings)}")
print(f"  Log saved to      : {LOG_FILE}")
print(f"{'='*50}\n")
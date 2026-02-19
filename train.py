"""
train.py
--------
Runs a MegatronLM GPT-2 training job with power monitoring.
Usage: python train.py --tp 1 --pp 1 --dp 4
"""

import argparse
import subprocess
import sys
import os

# Point MegatronLM fused kernels to system nvcc
os.environ.setdefault("CUDA_HOME", "/usr")
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
# Required for Tensor Parallelism
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
from compute_power import PowerMonitor

# ── Argument parsing ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
parser.add_argument("--dp", type=int, default=4, help="Data parallelism degree")
args = parser.parse_args()

TP, PP, DP = args.tp, args.pp, args.dp
assert TP * PP * DP == 4, f"TP x PP x DP must equal 4, got {TP*PP*DP}"

# ── Paths ─────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MEGATRON    = os.path.join(BASE, "Megatron-LM")
DATA_PATH   = os.path.join(BASE, "gpt2_wikitext103_text_document")
VOCAB_FILE  = os.path.join(BASE, "gpt2-vocab.json")
MERGE_FILE  = os.path.join(BASE, "gpt2-merges.txt")
LOG_FILE    = os.path.join(BASE, f"tp{TP}_pp{PP}_dp{DP}_power.csv")

# ── Power monitor ─────────────────────────────────────────────────────
monitor = PowerMonitor(interval=0.5, log_file=LOG_FILE)
monitor.start()

# ── Launch MegatronLM training ────────────────────────────────────────
cmd = [
    "torchrun",
    f"--nproc_per_node={TP * PP * DP}",
    f"{MEGATRON}/pretrain_gpt.py",
    # Use local transformer implementation (no Transformer Engine required)
    "--transformer-impl=local",
    "--no-persist-layer-norm",
    "--no-gradient-accumulation-fusion",
    # Parallelism
    f"--tensor-model-parallel-size={TP}",
    f"--pipeline-model-parallel-size={PP}",
    # GPT-2 Small model
    "--num-layers=12",
    "--hidden-size=768",
    "--num-attention-heads=12",
    "--seq-length=1024",
    "--max-position-embeddings=1024",
    # Training
    "--micro-batch-size=2", # Changing this will yield better efficiency
    "--global-batch-size=16",
    "--train-iters=50",
    "--lr=0.0001",
    "--min-lr=0.00001",
    "--lr-decay-style=cosine",
    "--lr-warmup-fraction=0.01",
    "--clip-grad=1.0",
    "--weight-decay=0.1",
    # Data
    f"--data-path={DATA_PATH}",
    f"--vocab-file={VOCAB_FILE}",
    f"--merge-file={MERGE_FILE}",
    "--split=949,50,1",
    # Logging
    "--log-interval=10",
    "--eval-iters=5",
    "--eval-interval=50",
    # Precision
    "--bf16",
]

print(f"\n{'='*50}")
print(f"  Training: TP={TP} PP={PP} DP={DP}")
print(f"{'='*50}\n")

result = subprocess.run(cmd, cwd=MEGATRON)

# ── Results ───────────────────────────────────────────────────────────
monitor.stop()
avg_power, readings = monitor.results()

print(f"\n{'='*50}")
print(f"  TP={TP} PP={PP} DP={DP}")
print(f"  Avg cluster power : {avg_power:.2f} W")
print(f"  Samples collected : {len(readings)}")
print(f"  Log saved to      : {LOG_FILE}")
print(f"{'='*50}\n")
"""
train_llama_3b.py
-----------------
Runs Megatron-LM LLaMA-3.2-3B training with power monitoring.
Usage: python train_llama_3b.py --tp 1 --pp 1 --dp 4
"""

import argparse
import subprocess
import os
import signal
import torch
from compute_power import PowerMonitor

# ── PyTorch 2.6 checkpoint loading fix ──────────────────────
# Allow SIGTERM object inside Megatron checkpoints
torch.serialization.add_safe_globals([signal.SIGTERM])

# ── Environment ─────────────────────────────────────────────
os.environ.setdefault("CUDA_HOME", "/usr")
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# ── Argument parsing ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--pp", type=int, default=1)
parser.add_argument("--dp", type=int, default=4)
parser.add_argument("--model", type=str, default="llama3b")
args = parser.parse_args()

TP, PP, DP, MODEL = args.tp, args.pp, args.dp, args.model
assert TP * PP * DP == 4, "TP x PP x DP must equal 4"

# ── Paths ───────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MEGATRON    = os.path.join(BASE, "Megatron-LM")
DATA_DIR    = os.path.join(BASE, "data", MODEL)
DATA_PATH   = os.path.join(BASE, "gpt2_wikitext103_text_document")
LOAD_PATH   = "/u/wat29/llama-3.2-3B-megatron"
LOG_FILE    = os.path.join(DATA_DIR, f"tp{TP}_pp{PP}_dp{DP}_power.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Power monitor ───────────────────────────────────────────
monitor = PowerMonitor(interval=0.5, log_file=LOG_FILE)
monitor.start()

# ── Megatron command ────────────────────────────────────────
cmd = [
    "torchrun",
    f"--nproc_per_node={TP * PP * DP}",
    f"{MEGATRON}/pretrain_gpt.py",

    # Transformer backend (identical to 1B benchmark)
    "--transformer-impl=local",
    "--no-persist-layer-norm",
    "--no-gradient-accumulation-fusion",

    # Parallelism
    f"--tensor-model-parallel-size={TP}",
    f"--pipeline-model-parallel-size={PP}",

    # Load converted 3B checkpoint
    f"--load={LOAD_PATH}",
    "--use-checkpoint-args",

    # Training (identical hyperparameters to 1B)
    "--micro-batch-size=1",
    "--global-batch-size=16",
    "--train-iters=50",
    "--lr=0.0001",
    "--min-lr=0.00001",
    "--lr-decay-style=cosine",
    "--lr-warmup-fraction=0.01",
    "--clip-grad=1.0",
    "--weight-decay=0.1",

    # Data (same dataset for benchmark consistency)
    f"--data-path={DATA_PATH}",
    "--split=949,50,1",

    # Logging
    "--log-interval=10",
    "--eval-iters=5",
    "--eval-interval=50",

    # Precision
    "--bf16",
]

print("\n" + "="*50)
print(f"  Training: model={MODEL} TP={TP} PP={PP} DP={DP}")
print("="*50 + "\n")

print("CMD:", " ".join(cmd))
subprocess.run(cmd, cwd=MEGATRON)

# ── Results ─────────────────────────────────────────────────
monitor.stop()
avg_power, readings = monitor.results()

print("\n" + "="*50)
print(f"  model={MODEL} TP={TP} PP={PP} DP={DP}")
print(f"  Avg cluster power : {avg_power:.2f} W")
print(f"  Samples collected : {len(readings)}")
print(f"  Log saved to      : {LOG_FILE}")
print("="*50 + "\n")
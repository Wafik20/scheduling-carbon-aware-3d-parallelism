"""
train_llama.py
--------------
Runs Megatron-LM LLaMA-3.2-1B training with power monitoring.
Usage: python train_llama.py --tp 1 --pp 1 --dp 4
"""

import argparse
import subprocess
import os
from compute_power import PowerMonitor

# ── Environment ─────────────────────────────────────────────
os.environ.setdefault("CUDA_HOME", "/usr")
os.environ["PATH"] = "/usr/bin:" + os.environ.get("PATH", "")
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# ── Argument parsing ────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--pp", type=int, default=1)
parser.add_argument("--dp", type=int, default=4)
parser.add_argument("--model", type=str, default="llama1b")
args = parser.parse_args()

TP, PP, DP, MODEL = args.tp, args.pp, args.dp, args.model
assert TP * PP * DP == 4, f"TP x PP x DP must equal 4"

# ── Paths ───────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
MEGATRON    = os.path.join(BASE, "Megatron-LM")
DATA_DIR    = os.path.join(BASE, "data", MODEL)
DATA_PATH   = os.path.join(BASE, "gpt2_wikitext103_text_document")
LOAD_PATH   = "/u/wat29/llama-3.2-1B-megatron"
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

    # Transformer backend
    "--transformer-impl=local",
    "--no-persist-layer-norm",
    "--no-gradient-accumulation-fusion",

    # Parallelism
    f"--tensor-model-parallel-size={TP}",
    f"--pipeline-model-parallel-size={PP}",

    # Load converted checkpoint
    f"--load={LOAD_PATH}",
    "--use-checkpoint-args",

    # LLaMA-3.2-1B Architecture
    "--num-layers=16",
    "--hidden-size=2048",
    "--num-attention-heads=32",
    "--ffn-hidden-size=8192",
    "--seq-length=4096",
    "--max-position-embeddings=4096",
    "--position-embedding-type=rope",
    "--normalization=RMSNorm",
    "--swiglu",
    "--untie-embeddings-and-output-weights",

    # Training
    "--micro-batch-size=1",
    "--global-batch-size=16",
    "--train-iters=50",
    "--lr=0.0001",
    "--min-lr=0.00001",
    "--lr-decay-style=cosine",
    "--lr-warmup-fraction=0.01",
    "--clip-grad=1.0",
    "--weight-decay=0.1",

    # Data (reuse WikiText for benchmarking)
    f"--data-path={DATA_PATH}",
    "--split=949,50,1",

    # Logging
    "--log-interval=10",
    "--eval-iters=5",
    "--eval-interval=50",

    # Precision
    "--bf16",
]

print(f"\n{'='*50}")
print(f"  Training: model={MODEL} TP={TP} PP={PP} DP={DP}")
print(f"{'='*50}\n")

subprocess.run(cmd, cwd=MEGATRON)

# ── Results ─────────────────────────────────────────────────
monitor.stop()
avg_power, readings = monitor.results()

print(f"\n{'='*50}")
print(f"  model={MODEL} TP={TP} PP={PP} DP={DP}")
print(f"  Avg cluster power : {avg_power:.2f} W")
print(f"  Samples collected : {len(readings)}")
print(f"  Log saved to      : {LOG_FILE}")
print(f"{'='*50}\n")
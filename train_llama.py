"""
train_llama.py
--------------
Runs Megatron LLaMA 3.2 (1B or 3B) training with power monitoring
across different 3D parallelism configurations. Uses mock data
and random init for power benchmarking.

Usage:
    apptainer run --nv \
        --bind /u/data3/wat29:/u/data3/wat29 \
        --bind /u/data3/wat29/Megatron-LM:/opt/megatron-lm \
        /u/data3/wat29/nemo.sif \
        python train_llama.py --model 1b --tp 1 --pp 1 --dp 4
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

# LLaMA 3.2 model configs
MODEL_CONFIGS = {
    "1b": {
        "num_layers": 16,
        "hidden_size": 2048,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "num_query_groups": 8,
    },
    "3b": {
        "num_layers": 28,
        "hidden_size": 3072,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 24,
        "num_query_groups": 8,
    },
}

# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LLaMA 3.2 training with power monitoring")
parser.add_argument("--model", type=str, default="1b", choices=["1b", "3b"],
                    help="Model size: 1b or 3b")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
parser.add_argument("--dp", type=int, default=4, help="Data parallelism degree")
parser.add_argument("--train-iters", type=int, default=50, help="Training iterations")
parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size")
parser.add_argument("--global-batch-size", type=int, default=8, help="Global batch size")
parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
args = parser.parse_args()

TP, PP, DP = args.tp, args.pp, args.dp
assert TP * PP * DP == NUM_GPUS, (
    f"TP({TP}) × PP({PP}) × DP({DP}) = {TP*PP*DP}, expected {NUM_GPUS}"
)

cfg = MODEL_CONFIGS[args.model]

# Validate parallelism is compatible with model architecture
assert cfg["num_layers"] % PP == 0, (
    f"{cfg['num_layers']} layers not divisible by PP={PP}"
)
assert cfg["num_attention_heads"] % TP == 0, (
    f"{cfg['num_attention_heads']} attention heads not divisible by TP={TP}"
)
assert cfg["num_query_groups"] % TP == 0, (
    f"{cfg['num_query_groups']} query groups not divisible by TP={TP}"
)

# ── Paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_DIR, "power_logs", f"llama_{args.model}")
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
    # LLaMA architecture
    "--num-layers", str(cfg["num_layers"]),
    "--hidden-size", str(cfg["hidden_size"]),
    "--ffn-hidden-size", str(cfg["ffn_hidden_size"]),
    "--num-attention-heads", str(cfg["num_attention_heads"]),
    "--group-query-attention",
    "--num-query-groups", str(cfg["num_query_groups"]),
    "--seq-length", str(args.seq_length),
    "--max-position-embeddings", str(args.seq_length),
    "--position-embedding-type", "rope",
    "--rotary-base", "500000",
    "--rotary-percent", "1.0",
    "--swiglu",
    "--normalization", "RMSNorm",
    "--disable-bias-linear",
    "--attention-dropout", "0.0",
    "--hidden-dropout", "0.0",
    "--untie-embeddings-and-output-weights",
    # Training
    "--micro-batch-size", str(args.micro_batch_size),
    "--global-batch-size", str(args.global_batch_size),
    "--train-iters", str(args.train_iters),
    "--lr", "0.00015",
    "--min-lr", "0.00001",
    "--lr-decay-style", "cosine",
    "--lr-warmup-fraction", "0.01",
    "--clip-grad", "1.0",
    "--weight-decay", "0.1",
    # Mock data (no real dataset needed for power benchmarking)
    "--mock-data",
    "--tokenizer-type", "NullTokenizer",
    "--vocab-size", "128256",
    "--split", "99,1,0",
    # Logging
    "--log-interval", "10",
    "--eval-iters", "5",
    "--eval-interval", "50",
    # Precision
    "--bf16",
    # Memory optimizations
    "--use-distributed-optimizer",
    "--transformer-impl", "local",
    "--no-persist-layer-norm",
    "--no-gradient-accumulation-fusion",
    "--recompute-granularity", "full",
    "--recompute-method", "uniform",
    "--recompute-num-layers", str(cfg["num_layers"] // PP),
]

# ── Run with power monitoring ─────────────────────────────────────────
tag = f"llama-{args.model} TP={TP} PP={PP} DP={DP}"
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
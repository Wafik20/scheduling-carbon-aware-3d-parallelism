"""
train.py
--------
Unified Megatron training script with power monitoring for:
  - GPT-2 Small (124M)
  - LLaMA 3.2 1B
  - LLaMA 3.2 3B

Uses mock data and random init for power benchmarking.
Automatically wraps the torchrun command inside the nemo.sif
Apptainer container, so this script is run from the **host**.

Usage:
    python train.py --model gpt2  --tp 1 --pp 1 --dp 4
    python train.py --model 1b    --tp 2 --pp 1 --dp 2
    python train.py --model 3b    --tp 4 --pp 2 --dp 1
"""

import argparse
import os
import subprocess
import sys

from compute_power import PowerMonitor

# ── Constants ─────────────────────────────────────────────────────────
NUM_GPUS = 4
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_SCRIPT = "/opt/megatron-lm/pretrain_gpt.py"

# Apptainer / container
CONTAINER_IMAGE = "/u/data3/wat29/nemo.sif"
CONTAINER_BINDS = [
    "/u/data3/wat29:/u/data3/wat29",
    "/u/data3/wat29/Megatron-LM:/opt/megatron-lm",
]

# ── Model configs ─────────────────────────────────────────────────────
# Each entry contains architecture params + any model-specific Megatron flags.
# "extra_args" are appended verbatim to the command.
MODEL_CONFIGS = {
    "gpt2": {
        "num_layers": 12,
        "hidden_size": 768,
        "ffn_hidden_size": None,          # default (4×hidden) for GPT-2
        "num_attention_heads": 12,
        "num_query_groups": None,         # standard MHA
        "seq_length": 1024,
        "vocab_size": 50257,
        "extra_args": [],
    },
    "1b": {
        "num_layers": 16,
        "hidden_size": 2048,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "seq_length": 256,
        "vocab_size": 128256,
        "extra_args": [
            "--group-query-attention",
            "--position-embedding-type", "rope",
            "--rotary-base", "500000",
            "--rotary-percent", "1.0",
            "--swiglu",
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--attention-dropout", "0.0",
            "--hidden-dropout", "0.0",
            "--untie-embeddings-and-output-weights",
            "--use-distributed-optimizer",
            "--transformer-impl", "local",
            "--no-persist-layer-norm",
            "--no-gradient-accumulation-fusion",
        ],
    },
    "3b": {
        "num_layers": 28,
        "hidden_size": 3072,
        "ffn_hidden_size": 8192,
        "num_attention_heads": 24,
        "num_query_groups": 8,
        "seq_length": 256,
        "vocab_size": 128256,
        "extra_args": [
            "--group-query-attention",
            "--position-embedding-type", "rope",
            "--rotary-base", "500000",
            "--rotary-percent", "1.0",
            "--swiglu",
            "--normalization", "RMSNorm",
            "--disable-bias-linear",
            "--attention-dropout", "0.0",
            "--hidden-dropout", "0.0",
            "--untie-embeddings-and-output-weights",
            "--use-distributed-optimizer",
            "--transformer-impl", "local",
            "--no-persist-layer-norm",
            "--no-gradient-accumulation-fusion",
        ],
    },
}

# ── Argument parsing ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Unified Megatron training with power monitoring")
parser.add_argument("--model", type=str, required=True,
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model to train: gpt2, 1b, or 3b")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
parser.add_argument("--dp", type=int, default=4, help="Data parallelism degree")
parser.add_argument("--train-iters", type=int, default=200,
                    help="Number of training iterations (main control for run duration)")
parser.add_argument("--micro-batch-size", type=int, default=2,
                    help="Micro batch size per GPU per step")
parser.add_argument("--global-batch-size", type=int, default=16,
                    help="Global batch size across all GPUs")
parser.add_argument("--seq-length", type=int, default=None,
                    help="Override default sequence length for the model")
parser.add_argument("--dry-run", action="store_true",
                    help="Print the full command without running")

args, extra_megatron_args = parser.parse_known_args()

TP, PP, DP = args.tp, args.pp, args.dp
assert TP * PP * DP == NUM_GPUS, (
    f"TP({TP}) × PP({PP}) × DP({DP}) = {TP*PP*DP}, expected {NUM_GPUS}"
)

cfg = MODEL_CONFIGS[args.model]
seq_length = args.seq_length or cfg["seq_length"]

# ── Validate parallelism vs. architecture ─────────────────────────────
assert cfg["num_layers"] % PP == 0, (
    f"{cfg['num_layers']} layers not divisible by PP={PP}"
)
assert cfg["num_attention_heads"] % TP == 0, (
    f"{cfg['num_attention_heads']} attention heads not divisible by TP={TP}"
)
if cfg["num_query_groups"] is not None:
    assert cfg["num_query_groups"] % TP == 0, (
        f"{cfg['num_query_groups']} query groups not divisible by TP={TP}"
    )

# ── Paths ─────────────────────────────────────────────────────────────
LOG_SUBDIRS = {"gpt2": "gpt2", "1b": "llama_1b", "3b": "llama_3b"}
RESULTS_DIR = os.path.join(PROJECT_DIR, "power_logs", LOG_SUBDIRS[args.model])
LOG_FILE = os.path.join(RESULTS_DIR, f"tp{TP}_pp{PP}_dp{DP}_power.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Build Megatron command ────────────────────────────────────────────
# Wrap torchrun inside the Apptainer container
apptainer_prefix = ["apptainer", "exec", "--nv"]
for bind in CONTAINER_BINDS:
    apptainer_prefix += ["--bind", bind]
apptainer_prefix += [
    "--env", "CUDA_DEVICE_MAX_CONNECTIONS=1",
    "--env", "CUDA_HOME=/usr/local/cuda",
    CONTAINER_IMAGE,
]

cmd = apptainer_prefix + [
    "torchrun",
    f"--nproc_per_node={NUM_GPUS}",
    PRETRAIN_SCRIPT,
    # Parallelism
    "--tensor-model-parallel-size", str(TP),
    "--pipeline-model-parallel-size", str(PP),
    # Architecture (common)
    "--num-layers", str(cfg["num_layers"]),
    "--hidden-size", str(cfg["hidden_size"]),
    "--num-attention-heads", str(cfg["num_attention_heads"]),
    "--seq-length", str(seq_length),
    "--max-position-embeddings", str(seq_length),
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
    # Mock data
    "--mock-data",
    "--tokenizer-type", "NullTokenizer",
    "--vocab-size", str(cfg["vocab_size"]),
    "--split", "99,1,0",
    # Logging
    "--log-interval", "10",
    "--eval-iters", "5",
    "--eval-interval", "50",
    # Precision
    "--bf16",
]

# Optional architecture fields
if cfg["ffn_hidden_size"] is not None:
    cmd += ["--ffn-hidden-size", str(cfg["ffn_hidden_size"])]

if cfg["num_query_groups"] is not None:
    cmd += ["--num-query-groups", str(cfg["num_query_groups"])]

# LLaMA models use activation recomputation
if args.model in ("1b", "3b"):
    cmd += [
        "--recompute-granularity", "full",
        "--recompute-method", "uniform",
        "--recompute-num-layers", str(cfg["num_layers"] // PP),
    ]

# Model-specific extra flags
cmd += cfg["extra_args"]

# Any extra flags the user passed through
cmd += extra_megatron_args

# ── Run with power monitoring ─────────────────────────────────────────
tag = f"{args.model} TP={TP} PP={PP} DP={DP}"
print(f"\n{'='*50}")
print(f"  Training: {tag}")
print(f"  Log:      {LOG_FILE}")
print(f"{'='*50}\n")

if args.dry_run:
    print("Command:\n")
    print("  " + " \\\n    ".join(cmd))
    print()
    sys.exit(0)

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
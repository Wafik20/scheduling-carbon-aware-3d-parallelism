"""
train.py
--------
Unified Megatron training script with power monitoring for:
  - GPT-2 Small  (124M)
  - GPT-2 Medium (355M)
  - GPT-2 Large  (774M)
  - LLaMA 3.2 1B
  - LLaMA 3.2 3B

Uses mock data and random init for power benchmarking.
Automatically wraps the torchrun command inside an Apptainer
container, so this script is run from the **host**.

All cluster-specific paths can be overridden via CLI flags or
environment variables, making it portable across setups.

Usage:
    # Local (tintin)
    python train.py --model gpt2      --tp 1 --pp 1 --dp 4

    # Pitt CRC
    python train.py --model gpt2 --tp 1 --pp 1 --dp 4 \
        --num-gpus 4 \
        --container-image /bgfs/stephenlee/wat29/nemo.sif \
        --container-bind /bgfs/stephenlee/wat29/Megatron-LM:/opt/megatron-lm \
        --container-bind /ihome/stephenlee/wat29:/ihome/stephenlee/wat29

Environment variables (override defaults):
    NUM_GPUS            Number of GPUs (default: 4)
    CONTAINER_IMAGE     Path to .sif file
    CONTAINER_BINDS     Comma-separated bind mounts (host:container,...)
    MEGATRON_DIR        Megatron-LM dir inside container (default: /opt/megatron-lm)
"""

import argparse
import os
import re
import subprocess
import sys

from compute_power import PowerMonitor

# ── Cluster defaults (overridable via env vars or CLI) ────────────────
DEFAULT_NUM_GPUS = 4
DEFAULT_CONTAINER_IMAGE = "/u/data3/wat29/nemo.sif"
DEFAULT_CONTAINER_BINDS = [
    "/u/data3/wat29:/u/data3/wat29",
    "/u/data3/wat29/Megatron-LM:/opt/megatron-lm",
]
DEFAULT_MEGATRON_DIR = "/opt/megatron-lm"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Regex to capture Megatron's iteration log lines ───────────────────
# Megatron prints lines like:  " iteration       10/     200 | ..."
_ITER_RE = re.compile(r"iteration\s+(\d+)/")

# ── Model configs ─────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gpt2": {
        "num_layers": 12,
        "hidden_size": 768,
        "ffn_hidden_size": None,
        "num_attention_heads": 12,
        "num_query_groups": None,
        "seq_length": 1024,
        "vocab_size": 50257,
        "extra_args": [],
    },
    "gpt2_medium": {
        "num_layers": 24,
        "hidden_size": 1024,
        "ffn_hidden_size": None,
        "num_attention_heads": 16,
        "num_query_groups": None,
        "seq_length": 1024,
        "vocab_size": 50257,
        "extra_args": [],
    },
    "gpt2_large": {
        "num_layers": 36,
        "hidden_size": 1280,
        "ffn_hidden_size": None,
        "num_attention_heads": 20,
        "num_query_groups": None,
        "seq_length": 1024,
        "vocab_size": 50257,
        "extra_args": [],
    },
    "llama_1b": {
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
    "llama_3b": {
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
parser = argparse.ArgumentParser(
    description="Unified Megatron training with power monitoring",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

# Model & parallelism
parser.add_argument("--model", type=str, required=True,
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model to train: gpt2, gpt2_medium, gpt2_large, llama_1b, or llama_3b")
parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism degree")
parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism degree")
parser.add_argument("--dp", type=int, default=None,
                    help="Data parallelism degree (default: num_gpus / (tp*pp))")

# Training
parser.add_argument("--train-iters", type=int, default=200,
                    help="Number of training iterations (main control for run duration)")
parser.add_argument("--micro-batch-size", type=int, default=2,
                    help="Micro batch size per GPU per step")
parser.add_argument("--global-batch-size", type=int, default=16,
                    help="Global batch size across all GPUs")
parser.add_argument("--seq-length", type=int, default=None,
                    help="Override default sequence length for the model")

# Cluster / container setup
parser.add_argument("--num-gpus", type=int,
                    default=int(os.environ.get("NUM_GPUS", DEFAULT_NUM_GPUS)),
                    help="Number of GPUs (default: $NUM_GPUS or 4)")
parser.add_argument("--container-image", type=str,
                    default=os.environ.get("CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE),
                    help="Path to Apptainer .sif image")
parser.add_argument("--container-bind", type=str, action="append", default=None,
                    help="Bind mount (host:container). Repeatable. "
                         "Overrides defaults when used.")
parser.add_argument("--megatron-dir", type=str,
                    default=os.environ.get("MEGATRON_DIR", DEFAULT_MEGATRON_DIR),
                    help="Megatron-LM directory inside the container")

parser.add_argument("--dry-run", action="store_true",
                    help="Print the full command without running")

args, extra_megatron_args = parser.parse_known_args()

# ── Resolve cluster config ────────────────────────────────────────────
NUM_GPUS = args.num_gpus
CONTAINER_IMAGE = args.container_image
MEGATRON_DIR = args.megatron_dir
PRETRAIN_SCRIPT = os.path.join(MEGATRON_DIR, "pretrain_gpt.py")

if args.container_bind:
    CONTAINER_BINDS = args.container_bind
elif os.environ.get("CONTAINER_BINDS"):
    CONTAINER_BINDS = os.environ["CONTAINER_BINDS"].split(",")
else:
    CONTAINER_BINDS = DEFAULT_CONTAINER_BINDS

# ── Resolve parallelism ──────────────────────────────────────────────
TP, PP = args.tp, args.pp
DP = args.dp if args.dp is not None else NUM_GPUS // (TP * PP)

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
RESULTS_DIR = os.path.join(PROJECT_DIR, "power_logs", args.model)
LOG_FILE = os.path.join(RESULTS_DIR, f"tp{TP}_pp{PP}_dp{DP}_power.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Build Megatron command ────────────────────────────────────────────
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
if args.model in ("llama_1b", "llama_3b"):
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
print(f"  GPUs:     {NUM_GPUS}")
print(f"  Image:    {CONTAINER_IMAGE}")
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
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        bufsize=1,                   # line-buffered
    )

    for line in proc.stdout:
        sys.stdout.write(line)       # echo everything to the terminal
        m = _ITER_RE.search(line)
        if m:
            monitor.update_step(int(m.group(1)))

    proc.wait()
    if proc.returncode != 0:
        print(f"Warning: training exited with code {proc.returncode}",
              file=sys.stderr)
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
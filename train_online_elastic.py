#!/usr/bin/env python3
"""
train_online_elastic.py
-----------------------
Launcher for ONLINE elastic parallelism training with power monitoring.

Sibling of train_elastic.py. Same model menu, same container, same
streaming power monitor; the only difference is that the inner script
(online_elastic_pretrain.py) decides phases at runtime from a strategies
config + a CI-forecast CSV, instead of consuming a fixed schedule JSON.

Inputs:
    --scheduler-config <path.json>   Calibration JSON from profile_strategies.py
                                     (provides strategies + average_reshard_wall_time_s)
    --carbon-forecast  <path.csv>    Hourly CI forecast (gCO2/kWh)
    --train-iters      <int>         Total training iterations

Policy parameters (required by inner script — provide via CLI here OR
embed in scheduler-config JSON; CLI takes precedence):

    --switch-power-w     <float>    Power draw during reshard (W)
    --deadline-seconds   <float>    Total training deadline (s)
    --lookahead-steps    <int>      Greedy lookahead horizon (steps)
    --decide-every       <int>      Decision interval (steps)

Optional overrides:

    --switch-time-seconds <float>   Override JSON's average_reshard_wall_time_s
    --bucket-seconds      <float>   Wall seconds per CI forecast row (default 3600)

Usage:
    python train_online_elastic.py --model gpt2_medium \\
        --scheduler-config calibrations/calibration_gpt2_medium_4gpu.json \\
        --carbon-forecast data/carbon_synthetic_dataset.csv \\
        --train-iters 100000 \\
        --switch-power-w 200 \\
        --deadline-seconds 14400 \\
        --lookahead-steps 8000 \\
        --decide-every 1000

    # Dry run:
    python train_online_elastic.py --model gpt2 \\
        --scheduler-config scheduler_config.json \\
        --carbon-forecast data/carbon_synthetic_dataset.csv \\
        --train-iters 5000 --dry-run
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
# Distinct from train_elastic.py's work dir so offline-planned and
# online-decided runs don't overwrite each other's phases.json /
# elastic_results.json / checkpoints.
DEFAULT_WORK_DIR = "/u/data3/wat29/carbon_training/online_elastic"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = PROJECT_DIR

# ── Regex patterns for extracting the current training step ──
#
# online_elastic_pretrain.py prints the same per-step format as
# elastic_pretrain.py — "iter N/M | loss: ... | ..." — so the existing
# patterns work unchanged. Same monotonic-clamp story applies.
_ITER_RE   = re.compile(r"\biter\s+(\d+)\s*/\s*\d+")
_STEP_RE   = re.compile(r"\[STEP\]\s+(\d+)")
_CKPT_RE   = re.compile(r"saving checkpoint at iteration\s+(\d+)", re.IGNORECASE)
_RESUME_RE = re.compile(r"Resuming from iteration:\s+(\d+)", re.IGNORECASE)
_STEP_PATTERNS = (_STEP_RE, _ITER_RE, _CKPT_RE, _RESUME_RE)


# ── Shared LLaMA extra_args ──────────────────────────────────
_LLAMA_EXTRA_ARGS = [
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
]

# ── Pythia family extra_args ─────────────────────────────────
_PYTHIA_EXTRA_ARGS = [
    "--position-embedding-type", "rope",
    "--rotary-base", "10000", "--rotary-percent", "0.25",
    "--untie-embeddings-and-output-weights",
    "--use-distributed-optimizer",
    "--transformer-impl", "local",
    "--no-persist-layer-norm",
    "--no-gradient-accumulation-fusion",
    "--attention-softmax-in-fp32",
]


# ── Model configs (kept in sync with train_elastic.py) ───────
#
# Divisibility constraints are enforced by online_elastic_pretrain.py for
# every (tp, pp) in the scheduler config, not just the first one. So all
# strategies in the user's scheduler_config.json must satisfy:
#   num_layers % pp == 0
#   num_attention_heads % tp == 0
#   num_query_groups % tp == 0   (GQA models only)
#   n_gpus % (tp * pp) == 0
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
        "extra_args": _LLAMA_EXTRA_ARGS,
    },
    "llama_2b": {
        "num_layers": 28, "hidden_size": 2560, "ffn_hidden_size": 6912,
        "num_attention_heads": 20, "num_query_groups": 4,
        "seq_length": 256, "vocab_size": 128256,
        "extra_args": _LLAMA_EXTRA_ARGS,
    },
    "llama_3b": {
        "num_layers": 28, "hidden_size": 3072, "ffn_hidden_size": 8192,
        "num_attention_heads": 24, "num_query_groups": 8,
        "seq_length": 256, "vocab_size": 128256,
        "extra_args": _LLAMA_EXTRA_ARGS,
    },
    "pythia_410m": {
        "num_layers": 24, "hidden_size": 1024, "ffn_hidden_size": 4096,
        "num_attention_heads": 16, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50304,
        "extra_args": _PYTHIA_EXTRA_ARGS,
    },
    "pythia_1_4b": {
        "num_layers": 24, "hidden_size": 2048, "ffn_hidden_size": 8192,
        "num_attention_heads": 16, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50304,
        "extra_args": _PYTHIA_EXTRA_ARGS,
    },
    "pythia_2_8b": {
        "num_layers": 32, "hidden_size": 2560, "ffn_hidden_size": 10240,
        "num_attention_heads": 32, "num_query_groups": None,
        "seq_length": 1024, "vocab_size": 50304,
        "extra_args": _PYTHIA_EXTRA_ARGS,
    },
}


# ── Scheduler config peeking ─────────────────────────────────

def read_initial_tp_pp(scheduler_config_path: str):
    """Pull the (tp, pp) of strategies[0] out of scheduler_config.json.

    Megatron needs the initial parallel sizes on the torchrun command line,
    and online_elastic_pretrain.py validates that they match exactly one
    entry in the strategies list. Convention: strategies[0] is the initial
    state (matches GreedyScheduler's default initial_strategy_idx=0).
    Reorder the JSON if you want a different starting config.
    """
    with open(scheduler_config_path) as f:
        cfg = json.load(f)
    if 'strategies' not in cfg or not cfg['strategies']:
        raise ValueError(
            f"{scheduler_config_path}: missing or empty 'strategies' list"
        )
    s0 = cfg['strategies'][0]
    if 'tp' not in s0 or 'pp' not in s0:
        raise ValueError(
            f"{scheduler_config_path}: strategies[0] missing 'tp' or 'pp'"
        )
    return int(s0['tp']), int(s0['pp'])


# ── Command building ─────────────────────────────────────────

def build_online_elastic_cmd(cfg, scheduler_config_path, carbon_forecast_path,
                             n_gpus, megatron_dir, work_dir,
                             first_tp, first_pp, total_iters,
                             micro_batch, global_batch, seq_length,
                             extra_args=None):
    """Build the torchrun + online_elastic_pretrain.py command."""
    script = os.path.join(megatron_dir, "online_elastic_pretrain.py")

    cmd = [
        "torchrun", f"--nproc_per_node={n_gpus}",
        script,
        "--scheduler-config", str(scheduler_config_path),
        "--carbon-forecast", str(carbon_forecast_path),
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
        # Reduces memory fragmentation. Particularly important for the
        # online path because it always uses fully-reshardable optimizer
        # checkpoints (any strategy can transition to any other), and the
        # allgather temp buffer can push peak usage over the L40 ceiling
        # if memory is fragmented.
        "--env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        container_image,
    ]
    return prefix


# ── Step tracking ────────────────────────────────────────────

def _extract_step(line):
    """Return the global step from a log line, or None. Identical
    behavior to train_elastic.py — the inner script's output format
    is the same."""
    for pat in _STEP_PATTERNS:
        m = pat.search(line)
        if m:
            return int(m.group(1))
    return None


# ── Streaming execution ──────────────────────────────────────

def run_with_streaming(cmd, monitor, logfile=None):
    """Run command, streaming output and feeding PowerMonitor.

    Note: online_elastic_pretrain.py runs its OWN NVML sampler internally
    (so the scheduler can observe live power). That sampler and this
    launcher's PowerMonitor are independent — both read NVML, neither
    blocks the other. The inner sampler informs decisions; the launcher
    monitor builds an external power log for post-hoc analysis.
    """
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
                if step >= max_step_seen:
                    max_step_seen = step
                    try:
                        monitor.update_step(step)
                    except Exception as e:
                        print(f"[train_online_elastic] update_step failed: {e}",
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
        description="Online (greedy carbon-aware) elastic parallelism training "
                    "with power monitoring",
    )
    parser.add_argument("--model", required=True,
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--scheduler-config", required=True,
                        help="Path to scheduler/calibration JSON. "
                             "strategies[0] is treated as the initial config.")
    parser.add_argument("--carbon-forecast", required=True,
                        help="Path to CI forecast CSV (year,month,day,time,carbon)")
    parser.add_argument("--train-iters", type=int, required=True,
                        help="Total training iterations. With no static "
                             "schedule, this must be specified directly.")
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS)
    parser.add_argument("--container-image", type=str, default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument("--container-bind", type=str, action="append", default=None)
    parser.add_argument("--megatron-dir", type=str, default=DEFAULT_MEGATRON_DIR)
    parser.add_argument("--work-dir", type=str, default=DEFAULT_WORK_DIR)
    parser.add_argument("--dry-run", action="store_true")

    # ── Policy parameters (forwarded to inner script) ────────
    # Required by inner script — supply via CLI here OR embed in the
    # scheduler-config JSON. CLI takes precedence when both are present.
    # Policy lives at the launcher level so the same calibration JSON
    # can drive multiple comparison runs (greedy / fast-only / slow-only)
    # by varying only the CLI flags.
    parser.add_argument("--switch-power-w", type=float, default=None,
                        help="Power draw during reshard window (W).")
    parser.add_argument("--deadline-seconds", type=float, default=None,
                        help="Total training deadline (s).")
    parser.add_argument("--lookahead-steps", type=int, default=None,
                        help="Greedy lookahead horizon (steps).")
    parser.add_argument("--decide-every", type=int, default=None,
                        help="Decision interval (steps).")
    parser.add_argument("--switch-time-seconds", type=float, default=None,
                        help="Override calibration's average_reshard_wall_time_s.")
    parser.add_argument("--bucket-seconds", type=float, default=None,
                        help="Wall seconds per CI forecast row (default 3600).")

    args, extra = parser.parse_known_args()
    cfg = MODEL_CONFIGS[args.model]
    seq_length = args.seq_length or cfg["seq_length"]
    binds = args.container_bind or DEFAULT_CONTAINER_BINDS

    scheduler_config_path = os.path.abspath(args.scheduler_config)
    carbon_forecast_path = os.path.abspath(args.carbon_forecast)

    # Both files must exist before we even start the container — much
    # cheaper to fail here than after Megatron init inside torchrun.
    for label, path in (("scheduler-config", scheduler_config_path),
                        ("carbon-forecast", carbon_forecast_path)):
        if not os.path.isfile(path):
            print(f"[fatal] --{label} not found: {path}", file=sys.stderr)
            sys.exit(2)

    # Read initial (tp, pp) from strategies[0] in the JSON. The inner
    # script will re-validate that this matches exactly one entry.
    first_tp, first_pp = read_initial_tp_pp(scheduler_config_path)

    # Build the policy-arg pass-through list. Only forward values that
    # were actually specified on the CLI; otherwise the inner script's
    # resolve() falls back to JSON. This lets the user mix sources.
    policy_forwards = []
    if args.switch_power_w is not None:
        policy_forwards += ["--switch-power-w", str(args.switch_power_w)]
    if args.deadline_seconds is not None:
        policy_forwards += ["--deadline-seconds", str(args.deadline_seconds)]
    if args.lookahead_steps is not None:
        policy_forwards += ["--lookahead-steps", str(args.lookahead_steps)]
    if args.decide_every is not None:
        policy_forwards += ["--decide-every", str(args.decide_every)]
    if args.switch_time_seconds is not None:
        policy_forwards += ["--switch-time-seconds", str(args.switch_time_seconds)]
    if args.bucket_seconds is not None:
        policy_forwards += ["--bucket-seconds", str(args.bucket_seconds)]

    # Naming for the launcher's log + power CSV — keyed off the scheduler
    # config name so multiple configs against the same model don't collide.
    config_name = Path(args.scheduler_config).stem
    log_dir = os.path.join(PROJECT_DIR, "power_logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    power_log = os.path.join(
        log_dir, f"online_elastic_{config_name}_power.csv"
    )

    print("=" * 60)
    print("  Online Elastic 3D Parallelism Training (greedy carbon-aware)")
    print("=" * 60)
    print(f"  Model:             {args.model}")
    print(f"  Scheduler config:  {scheduler_config_path}")
    print(f"  Carbon forecast:   {carbon_forecast_path}")
    print(f"  Train iters:       {args.train_iters}")
    print(f"  Initial (TP, PP):  ({first_tp}, {first_pp}) "
          f"← strategies[0]")
    print(f"  GPUs:              {args.num_gpus}")
    print(f"  Container:         {args.container_image}")
    print(f"  Work dir:          {args.work_dir}")
    print(f"  Power log:         {power_log}")

    # Show resolved policy values where possible. CLI values override JSON;
    # values not set on CLI will be picked up from JSON inside the inner script.
    policy_lines = []
    if args.switch_power_w is not None:
        policy_lines.append(f"  switch_power_w:    {args.switch_power_w} W (CLI)")
    if args.deadline_seconds is not None:
        policy_lines.append(f"  deadline_s:        {args.deadline_seconds:.0f}s "
                            f"({args.deadline_seconds/3600:.2f}h) (CLI)")
    if args.lookahead_steps is not None:
        policy_lines.append(f"  lookahead_steps:   {args.lookahead_steps} (CLI)")
    if args.decide_every is not None:
        policy_lines.append(f"  decide_every:      {args.decide_every} (CLI)")
    if args.switch_time_seconds is not None:
        policy_lines.append(f"  switch_time_s:     {args.switch_time_seconds}s "
                            f"(CLI override)")
    if args.bucket_seconds is not None:
        policy_lines.append(f"  bucket_seconds:    {args.bucket_seconds}s (CLI)")
    if policy_lines:
        print("  Policy (CLI):")
        for line in policy_lines:
            print(line)
    else:
        print("  Policy:            entirely from JSON (no CLI overrides)")

    print("=" * 60)
    print()

    # Combine policy pass-through with any other extras (Megatron args).
    all_extra = policy_forwards + extra

    megatron_cmd = build_online_elastic_cmd(
        cfg, scheduler_config_path, carbon_forecast_path,
        args.num_gpus, args.megatron_dir, args.work_dir,
        first_tp, first_pp, args.train_iters,
        args.micro_batch_size, args.global_batch_size, seq_length,
        extra_args=all_extra,
    )
    inner_cmd = f"cd {args.megatron_dir} && " + " ".join(megatron_cmd)
    full_cmd = build_apptainer_prefix(args.container_image, binds) \
               + ["bash", "-c", inner_cmd]

    if args.dry_run:
        print("Command:\n")
        print("  " + " \\\n    ".join(full_cmd))
        return

    monitor = PowerMonitor(interval=0.5, log_file=power_log)
    monitor.start()

    logfile = os.path.join(args.work_dir, f"online_elastic_{config_name}.log")
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
        print("  ONLINE ELASTIC TRAINING SUCCEEDED")
    else:
        print(f"  ONLINE ELASTIC TRAINING FAILED (exit {rc})")
        for e in info["errors"][:5]:
            print(f"    {e}")
    print(f"{'=' * 60}")
    print(f"  Wall time:         {wall:.1f}s")
    print(f"  Avg cluster power: {avg_power:.1f}W")
    print(f"  Power samples:     {len(readings)}")
    print(f"  Log:               {logfile}")
    print(f"  Power:             {power_log}")
    # The inner script also writes phases.json + elastic_results.json
    # to args.work_dir — pointer here for convenience.
    print(f"  Phases:            {os.path.join(args.work_dir, 'phases.json')}")
    print(f"  Inner summary:     {os.path.join(args.work_dir, 'elastic_results.json')}")
    print(f"{'=' * 60}\n")

    sys.exit(rc)


if __name__ == "__main__":
    main()
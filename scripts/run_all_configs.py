"""
run_all_configs.py
------------------
Enumerates every valid (TP, PP, DP) 3D-parallelism configuration for a
given world size and model, then runs train.py for each one sequentially
with power monitoring.

Cluster-specific settings (container path, bind mounts, GPU count) are
passed through to train.py and can also be set via environment variables.

Usage:
    python scripts/run_all_configs.py --model gpt2 --num-gpus 4 --dry-run
    python scripts/run_all_configs.py --model gpt2 --num-gpus 4 \
        --container-image /path/to/nemo.sif \
        --container-bind /path/to/Megatron-LM:/opt/megatron-lm
"""

import argparse
import itertools
import subprocess
import sys
import os
from datetime import datetime

# ── Defaults (same as train.py) ──────────────────────────────────────
DEFAULT_NUM_GPUS = 4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from train import MODEL_CONFIGS

MODEL_CONSTRAINTS = MODEL_CONFIGS

def divisors(n):
    """Return sorted list of divisors of n."""
    return sorted(d for d in range(1, n + 1) if n % d == 0)


def valid_configs(world_size, model):
    """Yield every (TP, PP, DP) triple that satisfies:
      - TP * PP * DP == world_size
      - num_layers % PP == 0
      - num_attention_heads % TP == 0
      - num_query_groups % TP == 0  (if GQA is used)
    """
    c = MODEL_CONSTRAINTS[model]
    for tp, pp in itertools.product(divisors(world_size), repeat=2):
        dp, rem = divmod(world_size, tp * pp)
        if rem != 0 or dp < 1:
            continue
        if c["num_layers"] % pp != 0:
            continue
        if c["num_attention_heads"] % tp != 0:
            continue
        if c["num_query_groups"] is not None and c["num_query_groups"] % tp != 0:
            continue
        yield tp, pp, dp


def main():
    parser = argparse.ArgumentParser(
        description="Run train.py across all valid parallelism configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model & training
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONSTRAINTS.keys()) + ["all"],
                        help="Model to benchmark: gpt2, gpt2_medium, gpt2_large, "
                             "llama_1b, llama_3b, or 'all'")
    parser.add_argument("--train-iters", type=int, default=200,
                        help="Number of training iterations per config")
    parser.add_argument("--micro-batch-size", type=int, default=2,
                        help="Micro batch size per GPU per step")
    parser.add_argument("--global-batch-size", type=int, default=16,
                        help="Global batch size across all GPUs")
    parser.add_argument("--seq-length", type=int, default=None,
                        help="Override default sequence length for the model")
    parser.add_argument("--warmup-iters", type=int, default=5,
                        help="Short probe run to check if config fits in memory "
                             "(0 to disable)")

    # Cluster setup (passed through to train.py)
    parser.add_argument("--num-gpus", type=int,
                        default=int(os.environ.get("NUM_GPUS", DEFAULT_NUM_GPUS)),
                        help="Number of GPUs (default: $NUM_GPUS or 4)")
    parser.add_argument("--container-image", type=str, default=None,
                        help="Path to Apptainer .sif image (passed to train.py)")
    parser.add_argument("--container-bind", type=str, action="append", default=None,
                        help="Bind mount host:container (repeatable, passed to train.py)")
    parser.add_argument("--megatron-dir", type=str, default=None,
                        help="Megatron-LM directory inside the container")

    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    args, extra = parser.parse_known_args()

    num_gpus = args.num_gpus
    models = list(MODEL_CONSTRAINTS.keys()) if args.model == "all" else [args.model]

    # Build plan: list of (model, configs)
    plan = []
    for model in models:
        cfgs = sorted(cfg for cfg in valid_configs(num_gpus, model)
                      if args.global_batch_size % (args.micro_batch_size * cfg[2]) == 0)
        if cfgs:
            plan.append((model, cfgs))

    total_runs = sum(len(cfgs) for _, cfgs in plan)
    if total_runs == 0:
        print(f"No valid configs for model={args.model}, num_gpus={num_gpus}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Models      : {', '.join(m for m, _ in plan)}")
    print(f"  GPUs        : {num_gpus}")
    print(f"  Total runs  : {total_runs}")
    print(f"  Train iters : {args.train_iters}")
    print(f"  Warmup probe: {args.warmup_iters} iters" if args.warmup_iters
          else "  Warmup probe: disabled")
    print(f"{'='*60}")
    for model, cfgs in plan:
        print(f"  [{model}]")
        for tp, pp, dp in cfgs:
            print(f"    TP={tp}  PP={pp}  DP={dp}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo_root, "train.py")
    results = []
    run_idx = 0

    # Cluster flags to forward to train.py
    cluster_args = ["--num-gpus", str(num_gpus)]
    if args.container_image:
        cluster_args += ["--container-image", args.container_image]
    if args.container_bind:
        for bind in args.container_bind:
            cluster_args += ["--container-bind", bind]
    if args.megatron_dir:
        cluster_args += ["--megatron-dir", args.megatron_dir]

    for model, cfgs in plan:
        for tp, pp, dp in cfgs:
            run_idx += 1
            tag = f"{model} TP={tp} PP={pp} DP={dp}"
            print(f"\n>>> [{run_idx}/{total_runs}] Starting {tag}  "
                  f"({datetime.now():%H:%M:%S})")

            base_cmd = [
                sys.executable, script,
                "--model", model,
                "--tp", str(tp),
                "--pp", str(pp),
                "--dp", str(dp),
                "--micro-batch-size", str(args.micro_batch_size),
                "--global-batch-size", str(args.global_batch_size),
            ] + cluster_args

            if args.seq_length is not None:
                base_cmd += ["--seq-length", str(args.seq_length)]

            # ── Warmup probe ──────────────────────────────────────────
            if args.warmup_iters > 0:
                probe_cmd = base_cmd + [
                    "--train-iters", str(args.warmup_iters),
                ] + extra

                print(f"  Probing with {args.warmup_iters} iters...")
                try:
                    ret = subprocess.run(probe_cmd)
                    if ret.returncode != 0:
                        status = "SKIP(probe failed)"
                        print(f"  ⚠ {tag} failed probe — does not fit in memory, "
                              "skipping...", file=sys.stderr)
                        results.append((model, tp, pp, dp, status))
                        print(f"<<< [{run_idx}/{total_runs}] {tag} → {status}")
                        continue
                except Exception as e:
                    status = f"SKIP(probe {type(e).__name__})"
                    print(f"  ⚠ {tag} probe error: {e}, skipping...",
                          file=sys.stderr)
                    results.append((model, tp, pp, dp, status))
                    print(f"<<< [{run_idx}/{total_runs}] {tag} → {status}")
                    continue
                print(f"  Probe passed — starting full run...")

            # ── Full training run ─────────────────────────────────────
            full_cmd = base_cmd + [
                "--train-iters", str(args.train_iters),
            ] + extra

            try:
                ret = subprocess.run(full_cmd)
                if ret.returncode == 0:
                    status = "OK"
                else:
                    status = f"FAIL(exit {ret.returncode})"
                    print(f"  ⚠ {tag} failed with exit code {ret.returncode}, "
                          "skipping to next config...", file=sys.stderr)
            except Exception as e:
                status = f"FAIL({type(e).__name__})"
                print(f"  ⚠ {tag} unexpected error: {e}, "
                      "skipping to next config...", file=sys.stderr)

            results.append((model, tp, pp, dp, status))
            print(f"<<< [{run_idx}/{total_runs}] {tag} → {status}")

    # ── Summary ───────────────────────────────────────────────────────
    passed = sum(1 for *_, s in results if s == "OK")
    failed = len(results) - passed

    print(f"\n{'='*60}")
    print(f"  Summary — {num_gpus} GPUs  ({passed} passed, {failed} failed)")
    print(f"{'='*60}")
    current_model = None
    for model, tp, pp, dp, status in results:
        if model != current_model:
            current_model = model
            print(f"  [{model}]")
        marker = "✓" if status == "OK" else "✗"
        print(f"    {marker} TP={tp}  PP={pp}  DP={dp}  →  {status}")
    print(f"{'='*60}\n")

    if any(s != "OK" for *_, s in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

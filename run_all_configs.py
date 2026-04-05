"""
run_all_configs.py
------------------
Enumerates every valid (TP, PP, DP) 3D-parallelism configuration for a
given world size and model, then runs train.py for each one sequentially
with power monitoring.

Usage:
    python run_all_configs.py --model gpt2      --world-size 4
    python run_all_configs.py --model llama_1b   --world-size 4
    python run_all_configs.py --model llama_3b   --world-size 8 --train-iters 100
    python run_all_configs.py --model all        --world-size 4
"""

import argparse
import itertools
import subprocess
import sys
import os
from datetime import datetime

# Architecture constraints (must stay in sync with train.py MODEL_CONFIGS)
MODEL_CONSTRAINTS = {
    "gpt2": {
        "num_layers": 12,
        "num_attention_heads": 12,
        "num_query_groups": None,   # standard MHA — no GQA constraint
    },
    "llama_1b": {
        "num_layers": 16,
        "num_attention_heads": 32,
        "num_query_groups": 8,
    },
    "llama_3b": {
        "num_layers": 28,
        "num_attention_heads": 24,
        "num_query_groups": 8,
    },
}


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
        description="Run train.py across all valid parallelism configs"
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONSTRAINTS.keys()) + ["all"],
                        help="Model to benchmark: gpt2, llama_1b, llama_3b, or 'all'")
    parser.add_argument("--world-size", type=int, default=4,
                        help="Total number of GPUs")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    args, extra = parser.parse_known_args()

    models = list(MODEL_CONSTRAINTS.keys()) if args.model == "all" else [args.model]

    # Build plan: list of (model, configs)
    plan = []
    for model in models:
        cfgs = sorted(valid_configs(args.world_size, model))
        if cfgs:
            plan.append((model, cfgs))

    total_runs = sum(len(cfgs) for _, cfgs in plan)
    if total_runs == 0:
        print(f"No valid configs for model={args.model}, world_size={args.world_size}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Models      : {', '.join(m for m, _ in plan)}")
    print(f"  World size  : {args.world_size} GPUs")
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

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    results = []
    run_idx = 0

    for model, cfgs in plan:
        for tp, pp, dp in cfgs:
            run_idx += 1
            tag = f"{model} TP={tp} PP={pp} DP={dp}"
            print(f"\n>>> [{run_idx}/{total_runs}] Starting {tag}  "
                  f"({datetime.now():%H:%M:%S})")

            cmd = [
                sys.executable, script,
                "--model", model,
                "--tp", str(tp),
                "--pp", str(pp),
                "--dp", str(dp),
                "--micro-batch-size", str(args.micro_batch_size),
                "--global-batch-size", str(args.global_batch_size),
            ]

            if args.seq_length is not None:
                cmd += ["--seq-length", str(args.seq_length)]

            # ── Warmup probe: quick check that the config fits in memory ──
            if args.warmup_iters > 0:
                probe_cmd = cmd + [
                    "--train-iters", str(args.warmup_iters),
                    "--dry-run",  # skip power monitoring for the probe
                ] + extra

                # Remove --dry-run: we actually want to run inside container,
                # but we don't need power logging. Use a minimal eval setup.
                probe_cmd = cmd + [
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

            # ── Full training run ─────────────────────────────────────────
            full_cmd = cmd + [
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
    print(f"  Summary — {args.world_size} GPUs  ({passed} passed, {failed} failed)")
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
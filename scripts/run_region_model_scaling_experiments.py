#!/usr/bin/env python3
"""Run regional dynamic-parallelism sweeps across measured model profiles.

This driver mirrors run_region_model_token_sweep_experiments.py, but expands
the model catalog to:

- GPT-2
- GPT-2 Medium
- GPT-2 Large
- Llama 1B
- Llama 2B
- Llama 3B

The default token grid scales from:

    1B, 2B, ..., 8B

The largest Llama 3B budgets exceed the clean non-wrapping random-offset room in
the May 2024 trace under the evaluator's deadline slack accounting. For that
reason this scaling sweep defaults to per-run offset windows and records the
available random-offset room in the manifest; use --offset-sampling shared and
--trace-window-policy strict for the earlier clean shared-window comparison.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_region_model_token_sweep_experiments as base


REPO_ROOT = base.REPO_ROOT
DEFAULT_PYTHON = base.DEFAULT_PYTHON

DEFAULT_MODELS: tuple[str, ...] = (
    "gpt2",
    "gpt2_medium",
    "gpt2_large",
    "llama1b",
    "llama2b",
    "llama3b",
)

DEFAULT_TOKEN_BUDGETS: tuple[int, ...] = (
    1_000_000_000,
    2_000_000_000,
    3_000_000_000,
    4_000_000_000,
    5_000_000_000,
    6_000_000_000,
    7_000_000_000,
    8_000_000_000,
)

MODEL_ALIASES = {
    "gpt": "gpt2",
    "gpt2": "gpt2",
    "gpt_2": "gpt2",
    "gpt2_medium": "gpt2_medium",
    "gpt_medium": "gpt2_medium",
    "gpt2_large": "gpt2_large",
    "gpt_large": "gpt2_large",
    "llama1b": "llama1b",
    "llama_1b": "llama1b",
    "llama2b": "llama2b",
    "llama_2b": "llama2b",
    "llama3b": "llama3b",
    "llama_3b": "llama3b",
}


def parse_model_slug(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(DEFAULT_MODELS)
        raise argparse.ArgumentTypeError(
            f"unknown model {value!r}; choose one of: {choices}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the May 2024 regional fixed-start dynamic-parallelism "
            "model-scaling sweep."
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing carbon_2024_05_hourly_aws_<region>.csv traces.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "region_may2024_model_scaling_1b_to_8b_25slack_per_run_offsets",
        help="Root output directory for all runs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for runner and per-run evaluator logs. Defaults to <out-root>/logs.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python executable to use for evaluate_carbon_policy.py.",
    )
    parser.add_argument(
        "--gpt2-profile",
        type=Path,
        default=REPO_ROOT / "runs" / "fixed_dp_25slack_gpt2_3p5b" / "strategy_profile.csv",
        help="GPT-2 measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--gpt2-medium-profile",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "exp3_gpt2_medium_may2024_shift_dp"
        / "strategy_profile.csv",
        help="GPT-2 Medium measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--gpt2-large-profile",
        type=Path,
        default=REPO_ROOT
        / "runs"
        / "exp3_gpt2_large_may2024_shift_dp"
        / "strategy_profile.csv",
        help="GPT-2 Large measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--llama1b-profile",
        type=Path,
        default=REPO_ROOT / "runs" / "fixed_dp_25slack_llama1b_3b" / "strategy_profile.csv",
        help="Llama 1B measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--llama2b-profile",
        type=Path,
        default=REPO_ROOT / "runs" / "exp1_llama_2b_may2024" / "strategy_profile.csv",
        help="Llama 2B measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--llama3b-profile",
        type=Path,
        default=REPO_ROOT / "runs" / "exp1_llama_3b_may2024" / "strategy_profile.csv",
        help="Llama 3B measured strategy_profile.csv.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=parse_model_slug,
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
        help=(
            "Subset of models to run. Aliases like gpt-medium and llama_3b "
            "are accepted."
        ),
    )
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        type=base.parse_token_budget,
        default=list(DEFAULT_TOKEN_BUDGETS),
        help="Shared fixed token budgets for every model, e.g. 1b 2b ... 8b.",
    )
    parser.add_argument(
        "--slack-frac",
        type=float,
        default=0.25,
        help="Deadline slack over fastest static wall time. Default 0.25.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=64,
        help="Number of trace-offset draws per run.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="CPU worker processes passed to evaluate_carbon_policy.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by the evaluator.",
    )
    parser.add_argument(
        "--n-detail-rollouts",
        type=int,
        default=0,
        help="Number of per-episode detail plots to save. Default 0 for speed.",
    )
    parser.add_argument(
        "--forecast-horizon-h",
        type=int,
        default=6,
        help="Forecast horizon passed through to the evaluator.",
    )
    parser.add_argument(
        "--smart-lookahead-h",
        type=int,
        default=6,
        help="Greedy lookahead horizon passed through to the evaluator.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(base.REGIONS),
        choices=list(base.REGIONS),
        help="Subset of AWS regions to run.",
    )
    parser.add_argument(
        "--offset-sampling",
        choices=["shared", "per-run"],
        default="per-run",
        help=(
            "Use one shared random-offset window across all runs, or let each "
            "run use its own deadline-dependent offset window."
        ),
    )
    parser.add_argument(
        "--min-shared-offset-window-h",
        type=float,
        default=240.0,
        help=(
            "Minimum acceptable shared random-offset window. Default 240h "
            "keeps at least ten days of May trace start times."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary.csv already exists.",
    )
    parser.add_argument(
        "--trace-window-policy",
        choices=["allow-zero-offset", "strict"],
        default="allow-zero-offset",
        help=(
            "How to handle model/token pairs whose deadline leaves no random "
            "offset room in the non-wrapping May trace. The default keeps the "
            "1B-8B grid and lets the evaluator pin those runs to offset 0; "
            "strict rejects them."
        ),
    )
    parser.add_argument(
        "--allow-short-trace",
        action="store_true",
        help=(
            "Compatibility alias for --trace-window-policy allow-zero-offset."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print completed/partial/missing runs for --out-root and exit.",
    )
    parser.add_argument(
        "--no-stream-evaluator-output",
        action="store_true",
        help="Write evaluator output only to per-run log files.",
    )
    return parser.parse_args()


def build_model_specs(args: argparse.Namespace) -> dict[str, base.ModelSpec]:
    return {
        "gpt2": base.ModelSpec(
            slug="gpt2",
            label="GPT-2",
            evaluator_model_name="gpt2",
            profile_csv=args.gpt2_profile,
            n_gpus=8,
            seq_length=1024,
        ),
        "gpt2_medium": base.ModelSpec(
            slug="gpt2_medium",
            label="GPT-2 Medium",
            evaluator_model_name="gpt2_medium",
            profile_csv=args.gpt2_medium_profile,
            n_gpus=4,
            seq_length=1024,
        ),
        "gpt2_large": base.ModelSpec(
            slug="gpt2_large",
            label="GPT-2 Large",
            evaluator_model_name="gpt2_large",
            profile_csv=args.gpt2_large_profile,
            n_gpus=4,
            seq_length=1024,
        ),
        "llama1b": base.ModelSpec(
            slug="llama1b",
            label="Llama 1B",
            evaluator_model_name="llama_1b",
            profile_csv=args.llama1b_profile,
            n_gpus=4,
            seq_length=256,
        ),
        "llama2b": base.ModelSpec(
            slug="llama2b",
            label="Llama 2B",
            evaluator_model_name="llama_2b",
            profile_csv=args.llama2b_profile,
            n_gpus=4,
            seq_length=256,
        ),
        "llama3b": base.ModelSpec(
            slug="llama3b",
            label="Llama 3B",
            evaluator_model_name="llama_3b",
            profile_csv=args.llama3b_profile,
            n_gpus=4,
            seq_length=256,
        ),
    }


def profile_summary(model: base.ModelSpec) -> dict[str, object]:
    if not model.profile_csv.exists():
        raise FileNotFoundError(model.profile_csv)

    required = {
        "tp",
        "pp",
        "dp",
        "power_w",
        "step_time_s",
        "throughput_tok_s",
        "joules_per_token",
        "reshard_wall_time_s",
        "warmup_wall_s",
        "save_s",
        "warmup_iters",
    }
    with model.profile_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Strategy profile has no rows: {model.profile_csv}")

    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Strategy profile {model.profile_csv} missing columns: {sorted(missing)}"
        )

    gpu_products = sorted({
        int(row["tp"]) * int(row["pp"]) * int(row["dp"])
        for row in rows
    })
    if gpu_products != [model.n_gpus]:
        raise ValueError(
            f"{model.slug} declares n_gpus={model.n_gpus}, but profile "
            f"{model.profile_csv} has tp*pp*dp products {gpu_products}"
        )

    throughputs = [float(row["throughput_tok_s"]) for row in rows]
    powers = [float(row["power_w"]) for row in rows]
    return {
        "model": model.slug,
        "model_label": model.label,
        "evaluator_model_name": model.evaluator_model_name,
        "strategy_profile_csv": base.rel_or_abs(model.profile_csv),
        "n_gpus": model.n_gpus,
        "seq_length": model.seq_length,
        "strategy_count": len(rows),
        "fastest_throughput_tok_s": max(throughputs),
        "slowest_throughput_tok_s": min(throughputs),
        "min_power_w": min(powers),
        "max_power_w": max(powers),
    }


def write_model_catalog(path: Path, models: list[base.ModelSpec]) -> pd.DataFrame:
    catalog = pd.DataFrame([profile_summary(model) for model in models])
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(path, index=False)
    return catalog


def oracle_token_quantum_for_model(tokens: int, profile_csv: Path) -> int:
    """Choose a DP token quantum that can advance on one-hour steps."""
    quantum = base.oracle_token_quantum(tokens)
    profile = pd.read_csv(profile_csv)
    min_hourly_tokens = float(profile["throughput_tok_s"].min()) * 3600.0
    if quantum < min_hourly_tokens:
        return quantum

    safe_quantum = int(max(1_000_000, (min_hourly_tokens // 1_000_000) * 1_000_000))
    return min(quantum, safe_quantum)


def build_run_specs(args: argparse.Namespace) -> list[base.RunSpec]:
    model_specs = build_model_specs(args)
    selected_models = [model_specs[name] for name in dict.fromkeys(args.models)]

    runs: list[base.RunSpec] = []
    for region in args.regions:
        trace_path = base.find_trace(args.trace_dir, region)
        trace_h = base.trace_duration_h(trace_path)

        for model in selected_models:
            for tokens in args.token_budgets:
                deadline_h = base.max_episode_hours(
                    tokens,
                    model.profile_csv,
                    args.slack_frac,
                )
                offset_room_h = base.max_random_offset_h(
                    trace_h=trace_h,
                    max_episode_h=deadline_h,
                    forecast_horizon_h=args.forecast_horizon_h,
                    smart_lookahead_h=args.smart_lookahead_h,
                )
                out_dir = args.out_root / region / model.slug / base.token_slug(tokens)
                runs.append(
                    base.RunSpec(
                        region=region,
                        model=model,
                        tokens=tokens,
                        max_episode_hours=deadline_h,
                        oracle_token_quantum=oracle_token_quantum_for_model(
                            tokens,
                            model.profile_csv,
                        ),
                        trace_path=trace_path,
                        out_dir=out_dir,
                        max_random_offset_h=offset_room_h,
                    )
                )
    return runs


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    args = parse_args()
    if args.status:
        return base.print_status(args.out_root)

    if args.slack_frac <= 0:
        raise ValueError("--slack-frac must be positive")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    model_specs = build_model_specs(args)
    selected_models = [model_specs[name] for name in dict.fromkeys(args.models)]
    catalog_path = args.out_root / "model_catalog.csv"
    model_catalog = write_model_catalog(catalog_path, selected_models)

    runs = build_run_specs(args)
    allow_zero_offset = (
        args.allow_short_trace or args.trace_window_policy == "allow-zero-offset"
    )
    blocked = [run for run in runs if run.max_random_offset_h <= 0.0]
    if blocked and not allow_zero_offset:
        examples = "\n".join(
            "  "
            f"{run.region} {run.model.slug} {run.token_slug}: "
            f"deadline={run.max_episode_hours}h leaves "
            f"{run.max_random_offset_h:.1f}h random-offset room"
            for run in blocked[:12]
        )
        if len(blocked) > 12:
            examples += f"\n  ... and {len(blocked) - 12} more"
        raise ValueError(
            "Some runs are too long for the non-wrapping May trace. "
            "Reduce --token-budgets, use a longer trace, or pass "
            "--trace-window-policy allow-zero-offset if repeated/near-zero "
            "offsets are acceptable.\n"
            f"{examples}"
        )

    shared_offset_window_h: float | None = None
    if args.offset_sampling == "shared":
        shared_offset_window_h = min(run.max_random_offset_h for run in runs)
        if shared_offset_window_h <= 0.0 and args.allow_short_trace:
            shared_offset_window_h = 0.0
        elif shared_offset_window_h <= 0.0:
            raise ValueError(
                "The selected grid leaves no shared random-offset window. "
                "Use --offset-sampling per-run for the 1B-8B scaling sweep, "
                "reduce --token-budgets, use a longer trace, or pass "
                "--allow-short-trace to intentionally pin every shared-offset "
                "run to offset 0."
            )
        elif shared_offset_window_h < args.min_shared_offset_window_h:
            raise ValueError(
                "The selected grid leaves too little shared random-offset "
                "window for a clean paper comparison: "
                f"{shared_offset_window_h:.1f}h < "
                f"{args.min_shared_offset_window_h:.1f}h. "
                "Reduce --token-budgets, lower --min-shared-offset-window-h, "
                "or use --offset-sampling per-run for exploratory sweeps."
            )

    commands = [
        base.build_command(
            python=args.python,
            run=run,
            n_episodes=args.n_episodes,
            num_workers=args.num_workers,
            seed=args.seed,
            n_detail_rollouts=args.n_detail_rollouts,
            forecast_horizon_h=args.forecast_horizon_h,
            smart_lookahead_h=args.smart_lookahead_h,
            max_offset_h=shared_offset_window_h,
        )
        for run in runs
    ]
    manifest_path = args.out_root / "manifest.csv"
    log_dir = args.log_dir if args.log_dir is not None else args.out_root / "logs"
    runner_log_path = log_dir / "runner.log"
    progress_path = args.out_root / "progress.csv"
    status_path = args.out_root / "status.json"

    base.write_manifest(manifest_path, runs, commands, shared_offset_window_h)
    progress_rows = base.build_progress_rows(
        runs=runs,
        commands=commands,
        log_dir=log_dir,
    )
    base.write_progress(progress_path, progress_rows)
    base.write_status_json(
        status_path=status_path,
        out_root=args.out_root,
        progress_path=progress_path,
        runner_log_path=runner_log_path,
        rows=progress_rows,
    )

    runner_log = base.RunnerLog(runner_log_path)
    try:
        runner_log.write("")
        runner_log.write(f"==== model-scaling run started {now_iso()} ====")
        runner_log.write(f"Wrote model catalog to {catalog_path}")
        runner_log.write(model_catalog.round(3).to_string(index=False))
        runner_log.write(f"Wrote manifest to {manifest_path}")
        runner_log.write(f"Progress CSV: {progress_path}")
        runner_log.write(f"Status JSON: {status_path}")
        runner_log.write(f"Log directory: {log_dir}")
        if shared_offset_window_h is not None:
            runner_log.write(
                "Using shared random-offset window: "
                f"0.0h to {shared_offset_window_h:.1f}h"
            )
        if blocked and allow_zero_offset:
            runner_log.write(
                "Allowing "
                f"{len(blocked)} run(s) with no non-wrapping random-offset room; "
                "the evaluator will pin those episodes to offset 0."
            )

        for row_index, (run, command) in enumerate(zip(runs, commands)):
            idx = row_index + 1
            summary_path = run.out_dir / "summary.csv"
            run_log_path = log_dir / base.run_log_name(idx, run)
            if args.skip_existing and summary_path.exists():
                message = f"summary already exists: {summary_path}"
                runner_log.write(
                    f"[{idx:03d}/{len(runs):03d}] skip "
                    f"{run.region} {run.label}: {summary_path}"
                )
                base.update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="skipped_existing",
                    finished_at=now_iso(),
                    returncode=0,
                    message=message,
                )
                continue

            runner_log.write(
                f"[{idx:03d}/{len(runs):03d}] "
                f"{run.region} {run.label} "
                f"deadline={run.max_episode_hours}h "
                f"offset_room={run.max_random_offset_h:.1f}h"
            )
            runner_log.write(base.shell_quote(command))
            if args.dry_run:
                base.update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="dry_run",
                    finished_at=now_iso(),
                    returncode=0,
                    message="dry run only; command was not executed",
                )
                continue

            run.out_dir.mkdir(parents=True, exist_ok=True)
            start_time = time.monotonic()
            started_at = now_iso()

            def on_start(pid: int) -> None:
                base.update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="running",
                    started_at=started_at,
                    pid=pid,
                    message=f"running; evaluator log: {run_log_path}",
                )

            try:
                returncode = base.stream_logged_command(
                    command=command,
                    run_log_path=run_log_path,
                    stream_output=not args.no_stream_evaluator_output,
                    on_start=on_start,
                )
            except KeyboardInterrupt:
                base.update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="interrupted",
                    finished_at=now_iso(),
                    duration_s=base.seconds_since(start_time),
                    returncode="",
                    message="runner interrupted by KeyboardInterrupt",
                )
                raise

            duration_s = base.seconds_since(start_time)
            if returncode != 0:
                base.update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="failed",
                    finished_at=now_iso(),
                    duration_s=duration_s,
                    returncode=returncode,
                    message=f"failed; see evaluator log: {run_log_path}",
                )
                runner_log.write(
                    f"[{idx:03d}/{len(runs):03d}] failed "
                    f"returncode={returncode}; log={run_log_path}"
                )
                raise subprocess.CalledProcessError(returncode, command)

            base.update_progress_row(
                rows=progress_rows,
                row_index=row_index,
                progress_path=progress_path,
                status_path=status_path,
                out_root=args.out_root,
                runner_log_path=runner_log_path,
                status="complete",
                finished_at=now_iso(),
                duration_s=duration_s,
                returncode=returncode,
                message=f"complete; evaluator log: {run_log_path}",
            )
            runner_log.write(
                f"[{idx:03d}/{len(runs):03d}] complete "
                f"{run.region} {run.label} duration_s={duration_s}"
            )

        if not args.dry_run:
            combined = base.summarize_completed_runs(args.out_root)
            if not combined.empty:
                summary_path = args.out_root / "combined_fixed_dp_summary.csv"
                combined.to_csv(summary_path, index=False)
                runner_log.write(f"Wrote combined summary to {summary_path}")
                runner_log.write(combined.round(3).to_string(index=False))
            else:
                runner_log.write("No completed summaries found yet.")
        runner_log.write(f"==== model-scaling run finished {now_iso()} ====")
    finally:
        runner_log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run regional fixed-start dynamic-parallelism sweeps by model and token budget.

This is a driver around evaluate_carbon_policy.py for comparing GPT-2 and
Llama 1B on the same fixed token budgets across the May 2024 AWS regional
carbon traces.

The script intentionally does not pass --include-job-shift. It evaluates only:

- static baselines
- Greedy
- Fixed Start Time Dynamic Parallelism

By default the shared token grid is:

    1B, 2B, 3B, 4B, 5B, 6B, 7B, 8B

For each model/token pair, the deadline is computed as 25% slack over the
fastest measured strategy in that model's strategy profile.

The evaluator supports per-episode multiprocessing via --num-workers. This
driver still passes the measured strategy-profile CSVs, GPU counts, and shared
random-offset cap so the paper run uses the actual readings instead of the
embedded default strategy profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
EVALUATOR = REPO_ROOT / "evaluate_carbon_policy.py"
DP_LABEL = "Fixed Start Time Dynamic Parallelism"
STATIC_POLICY_PREFIX = "static("

PROGRESS_COLUMNS: tuple[str, ...] = (
    "idx",
    "total",
    "region",
    "model",
    "model_label",
    "tokens",
    "token_slug",
    "status",
    "started_at",
    "finished_at",
    "duration_s",
    "pid",
    "returncode",
    "out_dir",
    "summary_path",
    "run_log_path",
    "command",
    "message",
)

REGIONS: tuple[str, ...] = (
    "us-west-1",
    "us-west-2",
    "us-east-2",
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


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    evaluator_model_name: str
    profile_csv: Path
    n_gpus: int
    seq_length: int


@dataclass(frozen=True)
class RunSpec:
    region: str
    model: ModelSpec
    tokens: int
    max_episode_hours: int
    oracle_token_quantum: int
    trace_path: Path
    out_dir: Path
    max_random_offset_h: float

    @property
    def token_slug(self) -> str:
        return token_slug(self.tokens)

    @property
    def label(self) -> str:
        return f"{self.model.label} {self.token_slug}"


def parse_token_budget(value: str) -> int:
    text = value.strip().lower().replace("_", "")
    if not text:
        raise argparse.ArgumentTypeError("empty token budget")

    multiplier = 1
    if text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]

    try:
        amount = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid token budget {value!r}; use values like 3b or 3000000000"
        ) from exc

    tokens = int(round(amount * multiplier))
    if tokens <= 0:
        raise argparse.ArgumentTypeError(
            f"token budget must be positive, got {value!r}"
        )
    return tokens


def token_slug(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    return str(tokens)


def oracle_token_quantum(tokens: int) -> int:
    if tokens <= 3_000_000_000:
        return 5_000_000
    if tokens <= 7_000_000_000:
        return 10_000_000
    if tokens <= 12_000_000_000:
        return 25_000_000
    return 50_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPT-2 and Llama 1B fixed-start dynamic-parallelism "
            "regional token sweeps."
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
        default=REPO_ROOT / "runs" / "region_may2024_model_token_sweep_25slack_shared_offsets",
        help="Root output directory for all runs.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=(
            "Directory for runner and per-run evaluator logs. Defaults to "
            "<out-root>/logs."
        ),
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
        help="GPT-2 measured strategy profile CSV.",
    )
    parser.add_argument(
        "--llama-profile",
        type=Path,
        default=REPO_ROOT / "runs" / "fixed_dp_25slack_llama1b_3b" / "strategy_profile.csv",
        help="Llama 1B measured strategy profile CSV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2", "llama1b"],
        choices=["gpt2", "llama1b"],
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--token-budgets",
        nargs="+",
        type=parse_token_budget,
        default=list(DEFAULT_TOKEN_BUDGETS),
        help="Shared fixed token budgets for every model, e.g. 1b 2b 3b 5b 10b.",
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
        help=(
            "CPU worker processes to pass to evaluate_carbon_policy.py for "
            "per-episode parallelism. Use 1 for serial evaluator behavior."
        ),
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
        default=list(REGIONS),
        choices=REGIONS,
        help="Subset of AWS regions to run.",
    )
    parser.add_argument(
        "--offset-sampling",
        choices=["shared", "per-run"],
        default="shared",
        help=(
            "Use a shared random-offset window across all runs, or let each "
            "run use the evaluator's per-deadline offset window. Default "
            "shared is recommended for paper comparisons."
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
        "--allow-short-trace",
        action="store_true",
        help=(
            "Allow runs whose deadline leaves no room for random offsets in "
            "the May trace. Without this, such runs are rejected."
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
        help=(
            "Print completed/partial/missing runs for --out-root and exit. "
            "This reads manifest.csv and the output directories."
        ),
    )
    parser.add_argument(
        "--no-stream-evaluator-output",
        action="store_true",
        help=(
            "Write evaluator output to per-run log files without also "
            "streaming it to the terminal."
        ),
    )
    return parser.parse_args()


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def shell_quote(args: list[str]) -> str:
    return shlex.join(args)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def seconds_since(start: float) -> str:
    return f"{time.monotonic() - start:.3f}"


def run_log_name(idx: int, run: RunSpec) -> str:
    return f"{idx:03d}_{run.region}_{run.model.slug}_{run.token_slug}.log"


class RunnerLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self._handle.close()

    def write(self, message: str = "", *, echo: bool = True) -> None:
        if echo:
            print(message, flush=True)
        self._handle.write(message + "\n")


def write_progress(progress_path: Path, rows: list[dict[str, object]]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=PROGRESS_COLUMNS).to_csv(progress_path, index=False)


def write_status_json(
    *,
    status_path: Path,
    out_root: Path,
    progress_path: Path,
    runner_log_path: Path,
    rows: list[dict[str, object]],
) -> None:
    done_statuses = {"complete", "skipped_existing", "dry_run"}
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1

    running = [row for row in rows if row["status"] == "running"]
    status = {
        "updated_at": now_iso(),
        "out_root": rel_or_abs(out_root),
        "progress_csv": rel_or_abs(progress_path),
        "runner_log": rel_or_abs(runner_log_path),
        "total": len(rows),
        "done": sum(1 for row in rows if row["status"] in done_statuses),
        "counts": counts,
        "current": running[-1] if running else None,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def update_progress_row(
    *,
    rows: list[dict[str, object]],
    row_index: int,
    progress_path: Path,
    status_path: Path,
    out_root: Path,
    runner_log_path: Path,
    **updates: object,
) -> None:
    rows[row_index].update(updates)
    write_progress(progress_path, rows)
    write_status_json(
        status_path=status_path,
        out_root=out_root,
        progress_path=progress_path,
        runner_log_path=runner_log_path,
        rows=rows,
    )


def build_progress_rows(
    *,
    runs: list[RunSpec],
    commands: list[list[str]],
    log_dir: Path,
) -> list[dict[str, object]]:
    total = len(runs)
    rows: list[dict[str, object]] = []
    for idx, (run, command) in enumerate(zip(runs, commands), start=1):
        run_log_path = log_dir / run_log_name(idx, run)
        rows.append({
            "idx": idx,
            "total": total,
            "region": run.region,
            "model": run.model.slug,
            "model_label": run.model.label,
            "tokens": run.tokens,
            "token_slug": run.token_slug,
            "status": "pending",
            "started_at": "",
            "finished_at": "",
            "duration_s": "",
            "pid": "",
            "returncode": "",
            "out_dir": rel_or_abs(run.out_dir),
            "summary_path": rel_or_abs(run.out_dir / "summary.csv"),
            "run_log_path": rel_or_abs(run_log_path),
            "command": shell_quote(command),
            "message": "",
        })
    return rows


def stream_logged_command(
    *,
    command: list[str],
    run_log_path: Path,
    stream_output: bool,
    on_start: Callable[[int], None],
) -> int:
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8", buffering=1) as run_log:
        run_log.write(f"\n==== started_at={now_iso()} ====\n")
        run_log.write(f"$ {shell_quote(command)}\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        on_start(process.pid)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                run_log.write(line)
                if stream_output:
                    print(line, end="", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            run_log.write("\n==== interrupted; terminating child ====\n")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                run_log.write("\n==== child did not terminate; killing ====\n")
                process.kill()
                process.wait()
            raise
        finally:
            run_log.write(f"\n==== finished_at={now_iso()} returncode={process.returncode} ====\n")


def find_trace(trace_dir: Path, region: str) -> Path:
    expected = trace_dir / f"carbon_2024_05_hourly_aws_{region}.csv"
    if expected.exists():
        return expected

    matches = sorted(trace_dir.rglob(f"*{region}*.csv")) if trace_dir.exists() else []
    if len(matches) == 1:
        return matches[0]

    message = [
        f"Missing trace for {region}: {expected}",
        "",
        "Expected one of:",
        f"  {expected}",
        f"  {trace_dir}/**/*{region}*.csv",
    ]
    if matches:
        message.extend(["", "Multiple candidates matched:"])
        message.extend(f"  {path}" for path in matches[:20])
        if len(matches) > 20:
            message.append(f"  ... and {len(matches) - 20} more")
    raise FileNotFoundError("\n".join(message))


def trace_duration_h(trace_path: Path) -> float:
    with trace_path.open(newline="") as handle:
        row_count = sum(1 for _ in handle) - 1
    if row_count <= 0:
        raise ValueError(f"Trace has no data rows: {trace_path}")
    return float(row_count)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def fastest_throughput(profile_csv: Path) -> float:
    if not profile_csv.exists():
        raise FileNotFoundError(profile_csv)
    with profile_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        throughputs = [
            float(row["throughput_tok_s"])
            for row in reader
            if row.get("throughput_tok_s")
        ]
    if not throughputs:
        raise ValueError(f"No throughput_tok_s values found in {profile_csv}")
    return max(throughputs)


def max_episode_hours(tokens: int, profile_csv: Path, slack_frac: float) -> int:
    fastest_wall_h = tokens / fastest_throughput(profile_csv) / 3600.0
    return int(math.ceil(fastest_wall_h * (1.0 + slack_frac)))


def max_random_offset_h(
    *,
    trace_h: float,
    max_episode_h: float,
    forecast_horizon_h: float,
    smart_lookahead_h: float,
) -> float:
    evaluator_slack_h = (
        max_episode_h * 1.5
        + 1.0
        + max(forecast_horizon_h, smart_lookahead_h)
    )
    return max(0.0, trace_h - evaluator_slack_h)


def build_model_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        "gpt2": ModelSpec(
            slug="gpt2",
            label="GPT-2",
            evaluator_model_name="gpt2",
            profile_csv=args.gpt2_profile,
            n_gpus=8,
            seq_length=1024,
        ),
        "llama1b": ModelSpec(
            slug="llama1b",
            label="Llama 1B",
            evaluator_model_name="llama_1b",
            profile_csv=args.llama_profile,
            n_gpus=4,
            seq_length=256,
        ),
    }


def build_run_specs(args: argparse.Namespace) -> list[RunSpec]:
    model_specs = build_model_specs(args)
    selected_models = [model_specs[name] for name in args.models]

    runs: list[RunSpec] = []
    for region in args.regions:
        trace_path = find_trace(args.trace_dir, region)
        trace_h = trace_duration_h(trace_path)

        for model in selected_models:
            for tokens in args.token_budgets:
                deadline_h = max_episode_hours(tokens, model.profile_csv, args.slack_frac)
                offset_room_h = max_random_offset_h(
                    trace_h=trace_h,
                    max_episode_h=deadline_h,
                    forecast_horizon_h=args.forecast_horizon_h,
                    smart_lookahead_h=args.smart_lookahead_h,
                )
                out_dir = (
                    args.out_root
                    / region
                    / model.slug
                    / token_slug(tokens)
                )
                runs.append(
                    RunSpec(
                        region=region,
                        model=model,
                        tokens=tokens,
                        max_episode_hours=deadline_h,
                        oracle_token_quantum=oracle_token_quantum(tokens),
                        trace_path=trace_path,
                        out_dir=out_dir,
                        max_random_offset_h=offset_room_h,
                    )
                )
    return runs


def build_command(
    *,
    python: Path,
    run: RunSpec,
    n_episodes: int,
    num_workers: int,
    seed: int,
    n_detail_rollouts: int,
    forecast_horizon_h: int,
    smart_lookahead_h: int,
    max_offset_h: float | None,
) -> list[str]:
    command = [
        str(python),
        str(EVALUATOR),
        "--trace",
        rel_or_abs(run.trace_path),
        "--tokens",
        str(run.tokens),
        "--max-episode-hours",
        str(run.max_episode_hours),
        "--oracle-token-quantum",
        str(run.oracle_token_quantum),
        "--oracle-switch-cost-mode",
        "sim",
        "--n-episodes",
        str(n_episodes),
        "--num-workers",
        str(num_workers),
        "--n-detail-rollouts",
        str(n_detail_rollouts),
        "--seed",
        str(seed),
        "--forecast-horizon-h",
        str(forecast_horizon_h),
        "--smart-lookahead-h",
        str(smart_lookahead_h),
        "--model-name",
        run.model.evaluator_model_name,
        "--global-batch-size",
        "16",
        "--n-gpus",
        str(run.model.n_gpus),
        "--seq-length",
        str(run.model.seq_length),
        "--strategy-profile-csv",
        rel_or_abs(run.model.profile_csv),
        "--out",
        rel_or_abs(run.out_dir),
    ]
    if max_offset_h is not None:
        command.extend(["--max-offset-h", f"{max_offset_h:.6f}"])
    return command


def write_manifest(
    path: Path,
    runs: list[RunSpec],
    commands: list[list[str]],
    shared_offset_window_h: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, command in zip(runs, commands):
        rows.append({
            "region": run.region,
            "model": run.model.slug,
            "model_label": run.model.label,
            "tokens": run.tokens,
            "token_slug": run.token_slug,
            "max_episode_hours": run.max_episode_hours,
            "oracle_token_quantum": run.oracle_token_quantum,
            "max_random_offset_h": run.max_random_offset_h,
            "shared_offset_window_h": shared_offset_window_h,
            "trace_path": rel_or_abs(run.trace_path),
            "strategy_profile_csv": rel_or_abs(run.model.profile_csv),
            "n_gpus": run.model.n_gpus,
            "seq_length": run.model.seq_length,
            "out_dir": rel_or_abs(run.out_dir),
            "command": shell_quote(command),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_completed_runs(out_root: Path) -> pd.DataFrame:
    rows = []
    for results_path in sorted(out_root.glob("*/*/*/results.csv")):
        run_dir = results_path.parent

        try:
            region, model, token = run_dir.relative_to(out_root).parts
        except ValueError:
            continue

        results = pd.read_csv(results_path)
        results["completed_bool"] = bool_series(results["completed"])
        results["feasible_bool"] = bool_series(results["feasible"])
        results["has_reshards"] = results["reshards"].astype(float) > 0

        episode_key_cols = ["episode", "offset_h"]
        static_mask = results["policy"].str.startswith(STATIC_POLICY_PREFIX)
        feasible_static = results.loc[static_mask & results["feasible_bool"]].copy()
        if feasible_static.empty:
            continue

        feasible_static_counts = (
            feasible_static.groupby(episode_key_cols, dropna=False)
            .size()
            .rename("feasible_static_candidates_for_episode")
            .reset_index()
        )
        best_static_idx = (
            feasible_static.groupby(episode_key_cols, dropna=False)["carbon_g"]
            .idxmin()
        )
        episode_best_static = (
            feasible_static.loc[
                best_static_idx,
                episode_key_cols + ["policy", "carbon_g"],
            ]
            .rename(columns={
                "policy": "episode_best_feasible_static_policy",
                "carbon_g": "episode_best_feasible_static_carbon_g",
            })
            .merge(
                feasible_static_counts,
                on=episode_key_cols,
                how="left",
                validate="one_to_one",
            )
        )
        if episode_best_static.duplicated(episode_key_cols).any():
            raise AssertionError(
                f"Non-unique episode-best static baseline in {results_path}"
            )

        best_single_static_policy = (
            feasible_static.groupby("policy")["carbon_g"]
            .mean()
            .sort_values()
            .index[0]
        )
        best_single_static = feasible_static.loc[
            feasible_static["policy"] == best_single_static_policy,
            episode_key_cols + ["carbon_g"],
        ].rename(columns={"carbon_g": "best_single_static_policy_carbon_g"})

        dp_ep = results.loc[
            (results["policy"] == DP_LABEL)
            & results["completed_bool"],
            episode_key_cols + ["carbon_g", "feasible_bool", "reshards", "has_reshards"],
        ].rename(columns={"carbon_g": "fixed_dp_carbon_g"})
        feasible_dp = dp_ep.loc[dp_ep["feasible_bool"]].copy()
        paired = feasible_dp.merge(
            episode_best_static,
            on=episode_key_cols,
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            continue
        paired_single_static = paired.merge(
            best_single_static,
            on=episode_key_cols,
            how="inner",
            validate="one_to_one",
        )

        saved_g = (
            paired["episode_best_feasible_static_carbon_g"]
            - paired["fixed_dp_carbon_g"]
        )
        saved_pct = (
            100.0
            * saved_g
            / paired["episode_best_feasible_static_carbon_g"]
        )
        reshards_gt0 = paired.loc[paired["has_reshards"]]
        saved_pct_reshards_gt0 = (
            100.0
            * (
                reshards_gt0["episode_best_feasible_static_carbon_g"]
                - reshards_gt0["fixed_dp_carbon_g"]
            )
            / reshards_gt0["episode_best_feasible_static_carbon_g"]
        )
        old_saved_g = (
            paired_single_static["best_single_static_policy_carbon_g"]
            - paired_single_static["fixed_dp_carbon_g"]
        )
        old_saved_pct = (
            100.0
            * old_saved_g.mean()
            / paired_single_static["best_single_static_policy_carbon_g"].mean()
        )
        rows.append({
            "region": region,
            "model": model,
            "token_budget": token,
            "episode_key_columns": (
                "region,model,token_budget,max_episode_hours,n_gpus,"
                "seq_length,episode,offset_h"
            ),
            "best_single_static_policy": best_single_static_policy,
            "episode_best_static_carbon_mean": (
                paired["episode_best_feasible_static_carbon_g"].mean()
            ),
            "fixed_dp_carbon_mean": paired["fixed_dp_carbon_g"].mean(),
            "savings_vs_episode_best_static_g_mean": saved_g.mean(),
            "savings_vs_episode_best_static_pct_mean": saved_pct.mean(),
            "savings_vs_episode_best_static_pct_reshards_gt0_mean": (
                saved_pct_reshards_gt0.mean()
                if len(saved_pct_reshards_gt0) else float("nan")
            ),
            "savings_vs_best_single_static_policy_g_mean": old_saved_g.mean(),
            "savings_vs_best_single_static_policy_pct": old_saved_pct,
            "paired_feasible_runs": len(paired),
            "fixed_dp_feasible_runs": len(feasible_dp),
            "fixed_dp_rows_dropped_no_feasible_static_baseline": (
                len(feasible_dp) - len(paired)
            ),
            "feasible_static_candidates_mean": (
                paired["feasible_static_candidates_for_episode"].mean()
            ),
            "feasible_static_candidates_min": (
                paired["feasible_static_candidates_for_episode"].min()
            ),
            "reshards_gt0_runs": int(paired["has_reshards"].sum()),
            "reshards_zero_runs": int((paired["reshards"] == 0).sum()),
            "reshards_zero_fraction": float((paired["reshards"] == 0).mean()),
            "fixed_dp_reshards_median": paired["reshards"].median(),
            "run_dir": rel_or_abs(run_dir),
        })

    return pd.DataFrame(rows)


def manifest_output_status(out_root: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    manifest_path = out_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    rows: list[dict[str, object]] = []
    counts = {"complete": 0, "partial": 0, "missing": 0}
    with manifest_path.open(newline="") as handle:
        for idx, row in enumerate(csv.DictReader(handle), start=1):
            out_dir = Path(row["out_dir"])
            if not out_dir.is_absolute():
                out_dir = REPO_ROOT / out_dir

            files = {
                "results.csv": out_dir / "results.csv",
                "aggregate.png": out_dir / "aggregate.png",
                "summary.csv": out_dir / "summary.csv",
            }
            exists = {name: path.exists() for name, path in files.items()}
            if exists["summary.csv"]:
                status = "complete"
            elif out_dir.exists():
                status = "partial"
            else:
                status = "missing"
            counts[status] += 1
            rows.append({
                "idx": idx,
                "region": row["region"],
                "model": row["model"],
                "token_slug": row["token_slug"],
                "status": status,
                "results": exists["results.csv"],
                "aggregate": exists["aggregate.png"],
                "summary": exists["summary.csv"],
                "out_dir": rel_or_abs(out_dir),
            })
    return rows, counts


def print_status(out_root: Path) -> int:
    rows, counts = manifest_output_status(out_root)
    total = len(rows)
    print(f"Output root: {out_root}")
    print(
        f"complete={counts['complete']} "
        f"partial={counts['partial']} "
        f"missing={counts['missing']} "
        f"total={total}"
    )

    print("\nBy group:")
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["region"]), str(row["model"]))
        grouped.setdefault(
            key,
            {"complete": 0, "partial": 0, "missing": 0, "tokens": []},
        )
        grouped[key][str(row["status"])] += 1
        if row["status"] == "complete":
            grouped[key]["tokens"].append(row["token_slug"])

    for (region, model), group in grouped.items():
        done_tokens = ",".join(str(token) for token in group["tokens"]) or "-"
        print(
            f"{region:9s} {model:7s} "
            f"complete={group['complete']} "
            f"partial={group['partial']} "
            f"missing={group['missing']} "
            f"done_tokens={done_tokens}"
        )

    not_complete = [row for row in rows if row["status"] != "complete"]
    if not_complete:
        print("\nNot complete:")
        for row in not_complete:
            print(
                f"{row['idx']:02d} {row['region']:9s} {row['model']:7s} "
                f"{row['token_slug']:>2s} {row['status']:7s} "
                f"results={row['results']} aggregate={row['aggregate']} "
                f"summary={row['summary']} out={row['out_dir']}"
            )
    else:
        combined = out_root / "combined_fixed_dp_summary.csv"
        print("\nAll manifest runs are complete.")
        if combined.exists():
            print(f"Combined summary: {combined}")
        else:
            print("Combined summary has not been written yet.")
    return 0


def main() -> int:
    args = parse_args()
    if args.status:
        return print_status(args.out_root)

    if args.slack_frac <= 0:
        raise ValueError("--slack-frac must be positive")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")

    runs = build_run_specs(args)
    blocked = [run for run in runs if run.max_random_offset_h <= 0.0]
    if blocked and not args.allow_short_trace:
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
            "--allow-short-trace if repeated/near-zero offsets are acceptable.\n"
            f"{examples}"
        )

    shared_offset_window_h: float | None = None
    if args.offset_sampling == "shared":
        shared_offset_window_h = min(run.max_random_offset_h for run in runs)
        if shared_offset_window_h < args.min_shared_offset_window_h:
            raise ValueError(
                "The selected grid leaves too little shared random-offset "
                "window for a clean paper comparison: "
                f"{shared_offset_window_h:.1f}h < "
                f"{args.min_shared_offset_window_h:.1f}h. "
                "Reduce --token-budgets, lower --min-shared-offset-window-h, "
                "or use --offset-sampling per-run for exploratory sweeps."
            )

    commands = [
        build_command(
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
    write_manifest(manifest_path, runs, commands, shared_offset_window_h)
    progress_rows = build_progress_rows(
        runs=runs,
        commands=commands,
        log_dir=log_dir,
    )
    write_progress(progress_path, progress_rows)
    write_status_json(
        status_path=status_path,
        out_root=args.out_root,
        progress_path=progress_path,
        runner_log_path=runner_log_path,
        rows=progress_rows,
    )

    runner_log = RunnerLog(runner_log_path)
    try:
        runner_log.write("")
        runner_log.write(f"==== run started {now_iso()} ====")
        runner_log.write(f"Wrote manifest to {manifest_path}")
        runner_log.write(f"Progress CSV: {progress_path}")
        runner_log.write(f"Status JSON: {status_path}")
        runner_log.write(f"Log directory: {log_dir}")
        if shared_offset_window_h is not None:
            runner_log.write(
                "Using shared random-offset window: "
                f"0.0h to {shared_offset_window_h:.1f}h"
            )

        for row_index, (run, command) in enumerate(zip(runs, commands)):
            idx = row_index + 1
            summary_path = run.out_dir / "summary.csv"
            run_log_path = log_dir / run_log_name(idx, run)
            if args.skip_existing and summary_path.exists():
                message = f"summary already exists: {summary_path}"
                runner_log.write(
                    f"[{idx:03d}/{len(runs):03d}] skip "
                    f"{run.region} {run.label}: {summary_path}"
                )
                update_progress_row(
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
            runner_log.write(shell_quote(command))
            if args.dry_run:
                update_progress_row(
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
                update_progress_row(
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
                returncode = stream_logged_command(
                    command=command,
                    run_log_path=run_log_path,
                    stream_output=not args.no_stream_evaluator_output,
                    on_start=on_start,
                )
            except KeyboardInterrupt:
                update_progress_row(
                    rows=progress_rows,
                    row_index=row_index,
                    progress_path=progress_path,
                    status_path=status_path,
                    out_root=args.out_root,
                    runner_log_path=runner_log_path,
                    status="interrupted",
                    finished_at=now_iso(),
                    duration_s=seconds_since(start_time),
                    returncode="",
                    message="runner interrupted by KeyboardInterrupt",
                )
                raise

            duration_s = seconds_since(start_time)
            if returncode != 0:
                update_progress_row(
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

            update_progress_row(
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
            combined = summarize_completed_runs(args.out_root)
            if not combined.empty:
                summary_path = args.out_root / "combined_fixed_dp_summary.csv"
                combined.to_csv(summary_path, index=False)
                runner_log.write(f"Wrote combined summary to {summary_path}")
                runner_log.write(combined.round(3).to_string(index=False))
            else:
                runner_log.write("No completed summaries found yet.")
        runner_log.write(f"==== run finished {now_iso()} ====")
    finally:
        runner_log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

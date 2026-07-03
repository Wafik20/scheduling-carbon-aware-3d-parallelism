#!/usr/bin/env python3
"""Run the May region comparison experiments.

This is a thin driver around evaluate_carbon_policy.py. It reproduces the
25%-slack fixed-start dynamic-parallelism experiment grid for:

- GPT-2 at 3.5B, 10B, 12B, 15B, and 20B tokens
- Llama 1B at 3B tokens
- AWS us-west-1, us-west-2, and us-east-2 May 2024 traces

By default the script looks for traces in ./data, but pass --trace-dir /data
when running on a machine where the traces are mounted there.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
EVALUATOR = REPO_ROOT / "evaluate_carbon_policy.py"


@dataclass(frozen=True)
class Workload:
    slug: str
    label: str
    tokens: int
    max_episode_hours: int
    oracle_token_quantum: int
    extra_args: tuple[str, ...] = field(default_factory=tuple)


WORKLOADS: tuple[Workload, ...] = (
    Workload(
        slug="gpt2_3p5b",
        label="GPT-2 3.5B tokens",
        tokens=3_500_000_000,
        max_episode_hours=12,
        oracle_token_quantum=5_000_000,
    ),
    Workload(
        slug="gpt2_10b",
        label="GPT-2 10B tokens",
        tokens=10_000_000_000,
        max_episode_hours=32,
        oracle_token_quantum=25_000_000,
    ),
    Workload(
        slug="gpt2_15b",
        label="GPT-2 15B tokens",
        tokens=15_000_000_000,
        max_episode_hours=48,
        oracle_token_quantum=25_000_000,
    ),
    Workload(
        slug="gpt2_12b",
        label="GPT-2 12B tokens",
        tokens=12_000_000_000,
        max_episode_hours=39,
        oracle_token_quantum=25_000_000,
    ),
    Workload(
        slug="gpt2_20b",
        label="GPT-2 20B tokens",
        tokens=20_000_000_000,
        max_episode_hours=64,
        oracle_token_quantum=50_000_000,
    ),
    Workload(
        slug="llama1b_3b",
        label="Llama 1B 3B tokens",
        tokens=3_000_000_000,
        max_episode_hours=108,
        oracle_token_quantum=5_000_000,
        extra_args=(
            "--model-name",
            "llama_1b",
            "--n-gpus",
            "4",
            "--seq-length",
            "256",
            "--strategy-profile-csv",
            "runs/fixed_dp_25slack_llama1b_3b/strategy_profile.csv",
        ),
    ),
)


REGIONS: tuple[str, ...] = (
    "us-west-1",
    "us-west-2",
    "us-east-2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-DP May region experiment grid.",
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
        default=REPO_ROOT / "runs" / "region_may2024_fixed_dp_25slack",
        help="Root output directory for all runs.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python executable to use for evaluate_carbon_policy.py.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=32,
        help="Number of trace-offset draws per run.",
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
        "--regions",
        nargs="+",
        default=list(REGIONS),
        choices=REGIONS,
        help="Subset of regions to run.",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=[w.slug for w in WORKLOADS],
        choices=[w.slug for w in WORKLOADS],
        help="Subset of workloads to run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose summary.csv already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_command(
    *,
    python: Path,
    trace_path: Path,
    out_dir: Path,
    workload: Workload,
    n_episodes: int,
    seed: int,
    n_detail_rollouts: int,
) -> list[str]:
    return [
        str(python),
        str(EVALUATOR),
        "--trace",
        rel_or_abs(trace_path),
        "--tokens",
        str(workload.tokens),
        "--max-episode-hours",
        str(workload.max_episode_hours),
        "--oracle-token-quantum",
        str(workload.oracle_token_quantum),
        "--oracle-switch-cost-mode",
        "sim",
        "--n-episodes",
        str(n_episodes),
        "--n-detail-rollouts",
        str(n_detail_rollouts),
        "--seed",
        str(seed),
        "--out",
        rel_or_abs(out_dir),
        *workload.extra_args,
    ]


def shell_quote(args: list[str]) -> str:
    return shlex.join(args)


def find_trace(trace_dir: Path, region: str) -> Path:
    expected = trace_dir / f"carbon_2024_05_hourly_aws_{region}.csv"
    if expected.exists():
        return expected

    if trace_dir.exists():
        matches = sorted(trace_dir.rglob(f"*{region}*.csv"))
    else:
        matches = []

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
        message.extend(["", "Multiple candidates matched; please choose the trace dir/file layout explicitly:"])
        message.extend(f"  {path}" for path in matches[:20])
        if len(matches) > 20:
            message.append(f"  ... and {len(matches) - 20} more")
    else:
        repo_trace = REPO_ROOT / "data" / expected.name
        if repo_trace.exists():
            message.extend([
                "",
                "A matching repo-local trace exists. Try:",
                "  python3 scripts/run_region_fixed_dp_experiments.py --trace-dir data --dry-run",
            ])
        else:
            message.extend([
                "",
                "Check available files with:",
                f"  find {trace_dir} -name '*{region}*.csv' -print",
            ])
    raise FileNotFoundError("\n".join(message))


def main() -> int:
    args = parse_args()
    selected = set(args.workloads)
    workloads = [w for w in WORKLOADS if w.slug in selected]

    commands: list[tuple[str, str, Path, list[str]]] = []
    for region in args.regions:
        trace_path = find_trace(args.trace_dir, region)

        for workload in workloads:
            out_dir = args.out_root / region / workload.slug
            command = build_command(
                python=args.python,
                trace_path=trace_path,
                out_dir=out_dir,
                workload=workload,
                n_episodes=args.n_episodes,
                seed=args.seed,
                n_detail_rollouts=args.n_detail_rollouts,
            )
            commands.append((region, workload.label, out_dir, command))

    for idx, (region, label, out_dir, command) in enumerate(commands, start=1):
        summary_path = out_dir / "summary.csv"
        if args.skip_existing and summary_path.exists():
            print(f"[{idx:02d}/{len(commands):02d}] skip {region} {label}: {summary_path}")
            continue

        print(f"[{idx:02d}/{len(commands):02d}] {region} {label}")
        print(shell_quote(command))
        if args.dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

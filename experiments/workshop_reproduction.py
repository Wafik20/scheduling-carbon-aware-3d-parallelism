"""Audited workshop reproduction; no evaluations run without --execute.

Original fixed-start study: static, Greedy and full-future DP, no shift or RL.
The default Chinchilla joint-mode extension uses an adaptive quantum rule.
Workshop fixed-start mode preserves exact historical budgets and deadlines,
not recalibrated 25% slack, with updated eight-GPU measurements. Neither mode
is an exact hardware replication or a globally optimal oracle. Check quantum sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("us-east-2", "us-west-1", "us-west-2")
FIELDS = ("tp", "pp", "dp", "micro_batch_size", "num_microbatches", "power_w",
          "step_time_s", "throughput_tok_s", "joules_per_token", "warmup_wall_s",
          "warmup_iters", "save_s", "reshard_wall_time_s")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_row(row):
    """Account for CSV rounding (step 4 decimals, power 1, J/token 6)."""
    positive = ("power_w", "step_time_s", "throughput_tok_s", "joules_per_token")
    for field in positive + ("warmup_wall_s", "warmup_iters", "save_s"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0 or (field in positive and value == 0):
            raise ValueError(f"Invalid {field}: {value}")
    tokens = int(row["global_batch_size"]) * int(row["seq_length"])
    if int(row["tp"]) * int(row["pp"]) * int(row["dp"]) != 8:
        raise ValueError("Not an eight-GPU strategy")
    step, rate = float(row["step_time_s"]), float(row["throughput_tok_s"])
    if abs(step - tokens / rate) > 0.000051 + tokens * 0.0051 / rate**2:
        raise ValueError("Step/throughput mismatch")
    power, efficiency = float(row["power_w"]), float(row["joules_per_token"])
    if abs(efficiency - power / rate) > 0.00000051 + 0.051 / rate + power * 0.0051 / rate**2:
        raise ValueError("Power/energy mismatch")
    if int(row["tokens_per_iter"]) != tokens:
        raise ValueError("Token batch mismatch")


def audit_traces(data, study="chinchilla"):
    monthly = study == "workshop-fixed-start"
    period = "2024_05" if monthly else "2024"
    hours = 744 if monthly else 8784
    start = datetime(2024, 5, 1) if monthly else datetime(2024, 1, 1)
    end = datetime(2024, 5, 31, 23) if monthly else datetime(2024, 12, 31, 23)
    traces, warnings, series = {}, [], {}
    for region in REGIONS:
        path = (Path(data) / f"carbon_{period}_hourly_aws_{region}.csv").resolve()
        rows = list(csv.DictReader(path.open()))
        dates = [datetime.fromisoformat(f"{int(r['year']):04}-{int(r['month']):02}-{int(r['day']):02}T{r['time']}") for r in rows]
        carbon = [float(r["carbon"]) for r in rows]
        if len(rows) != hours or dates[0] != start or dates[-1] != end:
            raise ValueError(f"Incomplete hourly trace period: {path}")
        if any((b-a).total_seconds() != 3600 for a, b in zip(dates, dates[1:])) or any(not math.isfinite(c) or c < 0 for c in carbon):
            raise ValueError(f"Invalid hourly trace: {path}")
        for other, values in series.items():
            if carbon == values:
                warnings.append(f"BLOCKED: {region} and {other} have identical carbon series; provenance unresolved")
        series[region] = carbon
        traces[region] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), hours=len(rows))
    return traces, warnings


def static_seconds(row, tokens, tokens_per_iter):
    warm_tokens = float(row["warmup_iters"]) * tokens_per_iter
    warm_s = float(row["warmup_wall_s"])
    if warm_tokens > 0 and tokens <= warm_tokens:
        return float(row.get("save_s", 0)) + warm_s * tokens / warm_tokens
    return float(row.get("save_s", 0)) + warm_s + (tokens - warm_tokens) / float(row["throughput_tok_s"])


WORKSHOP_GRID = (
    ("gpt2", 3_500_000_000, 12, 5_000_000),
    ("gpt2", 10_000_000_000, 32, 25_000_000),
    ("gpt2", 12_000_000_000, 39, 25_000_000),
    ("gpt2", 15_000_000_000, 48, 25_000_000),
    ("gpt2", 20_000_000_000, 64, 50_000_000),
    ("llama_1b", 3_000_000_000, 108, 5_000_000),
)


def fixed_start_workloads(base, selected):
    """Keep historical workloads fixed while substituting audited measurements."""
    by_model = {w["model"]: w for w in base}
    workloads = []
    for model, tokens, deadline, quantum in WORKSHOP_GRID:
        workload = dict(by_model[model])
        rows = [r for r in selected if r["model"] == model]
        durations = [static_seconds(r, tokens, workload["global_batch_size"] * workload["seq_length"]) for r in rows]
        workload.update(slug=f"{model}_{tokens // 1_000_000}m", tokens=tokens,
                        deadline_h=deadline, token_quantum=quantum,
                        fastest_static_h=min(durations) / 3600,
                        feasible_static_strategies=sum(s <= deadline * 3600 for s in durations),
                        quantum_warning="Original fixed-grid quantum; check discretization sensitivity")
        workloads.append(workload)
    return workloads


def prepare(args):
    study = getattr(args, "study", "chinchilla")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sweep = Path(args.sweep).resolve()
    manifest = read_json(sweep / "manifest.json")
    cases = {c["id"]: c for c in manifest["cases"] if "tp-pp-dp" in c["equivalent_orders"] and (study != "workshop-fixed-start" or c["model"] in {"gpt2", "llama_1b"})}
    selected, rejected = {}, []
    for row in csv.DictReader((sweep / "results.csv").open()):
        if row["id"] not in cases:
            continue
        attempt = sweep / "cases" / row["id"] / "attempts" / f"{int(row['attempt']):04}"
        status_path = attempt / "status.json"
        status = read_json(status_path) if status_path.exists() else {}
        if status.get("status") != "completed" or status.get("returncode") != 0:
            rejected.append(dict(id=row["id"], attempt=row["attempt"], reason="not successful"))
            continue
        validate_row(row)
        key = (row["model"], *(int(row[k]) for k in ("tp", "pp", "dp", "micro_batch_size")))
        if key not in selected or int(row["attempt"]) > int(selected[key]["attempt"]):
            selected[key] = row
    write_csv(out / "selected_measurements.csv", list(selected.values()))
    profiles = out / "profiles"
    profiles.mkdir()
    workloads = []
    for model in sorted({c["model"] for c in cases.values()}):
        rows = [r for key, r in sorted(selected.items()) if key[0] == model]
        if not rows:
            raise ValueError(f"No successful strategies for {model}")
        counts = set()
        parameter_logs = []
        for r in rows:
            if int(r["tp"]) == int(r["pp"]) == 1:
                log = sweep / "cases" / r["id"] / "attempts" / f"{int(r['attempt']):04}" / "launcher.log"
                counts.update(int(v) for v in re.findall(r"number of parameters on .*?:\s*(\d+)", log.read_text()))
                parameter_logs.append(str(log))
        if len(counts) != 1:
            raise ValueError(f"Ambiguous full model parameter count for {model}: {counts}")
        params = counts.pop()
        tokens = 20 * params
        batches = {(int(r["global_batch_size"]), int(r["seq_length"])) for r in rows}
        if len(batches) != 1:
            raise ValueError(f"Inconsistent batches: {model}")
        batch, seq = batches.pop()
        fastest = min(static_seconds(r, tokens, batch * seq) for r in rows)
        profile = profiles / f"{model}.csv"
        normalized = []
        for row in rows:
            record = {field: row[field] for field in FIELDS}
            record["reshard_wall_time_s"] = row["reshard_wall_time_s"]
            record["average_reshard_wall_time_s"] = row["average_reshard_wall_time_s"]
            record.update(source_id=row["id"], source_attempt=row["attempt"])
            normalized.append(record)
        write_csv(profile, normalized)
        tier = 5_000_000 if tokens <= 3e9 else 10_000_000 if tokens <= 7e9 else 25_000_000 if tokens <= 12e9 else 50_000_000
        quantum = max(1_000_000, min(tier, math.floor(min(float(r["throughput_tok_s"]) * 3600 for r in rows) / 1_000_000) * 1_000_000))
        workloads.append(dict(token_quantum=quantum, model=model, parameters=params, tokens=tokens, deadline_h=math.ceil(1.25 * fastest / 3600), fastest_static_h=fastest / 3600, global_batch_size=batch, seq_length=seq, strategies=len(rows), profile=str(profile), parameter_logs=parameter_logs, min_steady_hourly_tokens=min(float(r["throughput_tok_s"]) * 3600 for r in rows), quantum_warning="q50m can round slow-strategy progress to zero" if min(float(r["throughput_tok_s"]) * 3600 for r in rows) <= 50_000_000 else "Warmup-hour progress may be smaller; test q25m/q5m sensitivity"))
    if study == "workshop-fixed-start":
        workloads = fixed_start_workloads(workloads, list(selected.values()))
    data = args.data or ROOT / "data" / ("recovered_workshop_may2024" if study == "workshop-fixed-start" else "")
    traces, warnings = audit_traces(data, study)
    for workload in workloads:
        if workload.get("feasible_static_strategies") == 0:
            warnings.append(f"BLOCKED: {workload['slug']} has no deadline-feasible measured static strategy; original deadline retained")
    shift_hours = 0 if study == "workshop-fixed-start" else 12
    max_offset = min(t["hours"] for t in traces.values()) - (1.5 * max(w["deadline_h"] for w in workloads) + shift_hours + 1 + 6)
    if max_offset < 0:
        raise ValueError("Traces too short for common safe offsets")
    chosen = {r["id"] for r in selected.values()}
    missing = [dict(id=c["id"], status=read_json(sweep / "cases" / c["id"] / "status.json").get("status")) for c in cases.values() if c["id"] not in chosen and (sweep / "cases" / c["id"] / "status.json").exists()]
    missing += [dict(id=c["id"], status="missing") for c in cases.values() if c["id"] not in chosen and not (sweep / "cases" / c["id"] / "status.json").exists()]
    write_json(out / "workloads.json", workloads)
    write_json(out / "audit.json", dict(study=study, sweep=str(sweep), source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [sweep / "manifest.json", sweep / "results.csv", *profiles.glob("*.csv")]}, traces=traces, warnings=warnings, blocked=bool(warnings), max_offset_h=max_offset, missing_or_failed=missing, rejected_attempts=rejected, selected_counts={w["model"]: w["strategies"] for w in workloads}, assumptions=(["Original six fixed-start token/deadline/quantum grid with updated eight-GPU measured hardware; not exact replication", "No job shifting; 32 paired episodes, seed 0"] if study == "workshop-fixed-start" else ["20 tokens per measured parameter", "deadline ceil(1.25 * fastest warmup-inclusive static hours)"]) + [ "warmup tokens count toward training", "discretized oracle; not a globally optimal oracle", "workers capped at16 using24GiB divided by24*(token bins+1)*(deadline hours+1)*strategies; estimate not peak-memory guarantee or practical runtime promise", "8 GPUs, global batch and sequence length from measured rows", "blank reshard_wall_time_s retained as missing; evaluator uses SimConfig.switch_time_s=0 fallback, not measured zero switch cost"]))
    print(f"Prepared {out}; {len(warnings)} trace provenance blockers; max offset {max_offset}h")


def commands(prepared, output, quantum=None, no_job_shift=False):
    audit = read_json(prepared / "audit.json")
    for workload in read_json(prepared / "workloads.json"):
        for region, trace in audit["traces"].items():
            dest = output / workload.get("slug", workload["model"]) / region
            effective_quantum = quantum or workload["token_quantum"]
            workers = min(16, max(1, int(24 * 1024**3 / (24 * (math.ceil(workload["tokens"] / effective_quantum) + 1) * (workload["deadline_h"] + 1) * workload["strategies"]))))
            cmd = [sys.executable, "-u", str(ROOT / "evaluate_carbon_policy.py"), "--trace", trace["path"], "--strategy-profile", workload["profile"], "--model-name", workload["model"], "--tokens", str(workload["tokens"]), "--model-num-params", str(workload["parameters"]), "--global-batch-size", str(workload["global_batch_size"]), "--seq-length", str(workload["seq_length"]), "--max-episode-hours", str(workload["deadline_h"]), "--max-offset-h", str(audit["max_offset_h"]), "--n-gpus", "8", "--n-episodes", "32", "--num-workers", str(workers), "--oracle-token-quantum", str(effective_quantum), "--include-job-shift", "--shift-window-h", "12", "--seed", "0", "--n-detail-rollouts", "0", "--out", str(dest)]
            if no_job_shift or audit.get("study") == "workshop-fixed-start":
                cmd.remove("--include-job-shift")
            yield dest, cmd


def run(args):
    prepared, out = Path(args.prepared).resolve(), Path(args.out).resolve()
    audit = read_json(prepared / "audit.json")
    for source, digest in audit["source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Audited source changed: {source}")
    if args.quantum is not None and (not math.isfinite(args.quantum) or args.quantum <= 0):
        raise ValueError("Token quantum must be positive")
    current, warnings = audit_traces(Path(next(iter(audit["traces"].values()))["path"]).parent, audit.get("study", "chinchilla"))
    if current != audit["traces"]:
        raise ValueError("Traces changed since prepare; prepare a new immutable audit")
    if not args.execute:
        for _, cmd in commands(prepared, out, args.quantum, args.no_job_shift):
            print(shlex.join(cmd))
        print("\n".join(warnings), file=sys.stderr)
        return
    if warnings or audit["blocked"]:
        raise ValueError("Trace provenance blocked: " + "; ".join(warnings))
    out.mkdir(parents=True, exist_ok=False)
    for dest, cmd in commands(prepared, out, args.quantum, args.no_job_shift):
        dest.mkdir(parents=True)
        status = dict(command=cmd, status="started", started_at=datetime.now(timezone.utc).isoformat())
        write_json(dest / "status.json", status)
        with (dest / "evaluation.log").open("w") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            status["pid"] = process.pid
            write_json(dest / "status.json", status)
            code = process.wait()
        status.update(status="completed" if code == 0 else "failed", returncode=code, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(dest / "status.json", status)
        if code:
            raise RuntimeError(f"Evaluation failed: {dest}")


def paired_rows(df):
    """Match energy and time to the saved best-CARBON static policy."""
    def truth(values):
        return values.astype(str).str.lower().isin(["true", "1"])
    valid = df[truth(df["feasible"]) & truth(df["completed"])].copy()
    baseline_name = "episode_best_feasible_static_policy"
    policies = valid[~valid.policy.str.startswith("static(") & valid[baseline_name].notna()].copy()
    baseline = valid[valid.policy.str.startswith("static(")][["episode", "offset_h", "policy", "carbon_g", "energy_kj", "completion_delay_h"]].rename(columns={"policy": baseline_name, "carbon_g": "baseline_carbon_g", "energy_kj": "baseline_energy_kj", "completion_delay_h": "baseline_completion_delay_h"})
    paired = policies.merge(baseline, on=["episode", "offset_h", baseline_name], validate="many_to_one")
    for metric in ("carbon_g", "energy_kj", "completion_delay_h"):
        paired[f"{metric}_savings_pct"] = 100 * (paired[f"baseline_{metric}"] - paired[metric]) / paired[f"baseline_{metric}"]
    return paired


def summarize(args):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames, coverage = [], []
    for path in sorted(Path(args.results).glob("*/*/results.csv")):
        df = pd.read_csv(path)
        paired = paired_rows(df)
        paired["model"], paired["region"] = path.parent.parent.name, path.parent.name
        frames.append(paired)
        for policy, group in df.groupby("policy"):
            coverage.append(dict(model=path.parent.parent.name, region=path.parent.name, policy=policy, total=len(group), feasible=int(group.feasible.astype(str).str.lower().isin(["true", "1"]).sum()), paired=int((paired.policy == policy).sum())))
    if not frames:
        raise ValueError("No evaluator results found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "coverage.csv", coverage)
    pairs = pd.concat(frames, ignore_index=True)
    pairs.to_csv(out / "paired_results.csv", index=False)
    if pairs.empty:
        (out / "README.txt").write_text("No feasible paired comparisons; no savings figures generated.\n")
        return
    metrics = ["carbon_g", "energy_kj", "completion_delay_h"]
    columns = metrics + [f"baseline_{m}" for m in metrics] + [f"{m}_savings_pct" for m in metrics]
    summary = pairs.groupby(["model", "region", "policy"])[columns].mean()
    summary["paired_n"] = pairs.groupby(["model", "region", "policy"]).size()
    if "reshards" in pairs:
        pairs["reshards_zero"] = pairs["reshards"].eq(0)
        summary["reshards_zero_fraction"] = pairs.groupby(["model", "region", "policy"])["reshards_zero"].mean()
    summary.to_csv(out / "paired_summary.csv")
    for policy, group in summary.reset_index().groupby("policy"):
        for metric in metrics:
            matrix = group.pivot(index="model", columns="region", values=f"{metric}_savings_pct")
            counts = group.pivot(index="model", columns="region", values="paired_n").reindex_like(matrix)
            fig, ax = plt.subplots(figsize=(8, 6))
            image = ax.imshow(matrix, cmap="RdBu", vmin=-100, vmax=100)
            ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20)
            ax.set_yticks(range(len(matrix.index)), matrix.index)
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    if pd.notna(matrix.iloc[i, j]):
                        ax.text(j, i, f"{matrix.iloc[i, j]:.1f}%\nN={int(counts.iloc[i, j])}", ha="center", va="center")
            ax.set_title(f"{policy}: {metric} savings")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(out / f"{re.sub(r'[^a-zA-Z0-9]+', '_', policy)}_{metric}_heatmap.png")
            plt.close(fig)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        pairs.boxplot(column=f"{metric}_savings_pct", by="policy", ax=ax, rot=20)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Savings (%) vs same-episode best-carbon static")
        fig.suptitle("Secondary pooled view (models/regions mixed); discretized DP")
        fig.tight_layout()
        fig.savefig(out / f"{metric}_savings.png")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sweep", default=str(ROOT / "runs/placement_sweep_8gpu"))
    p.add_argument("--study", choices=("chinchilla", "workshop-fixed-start"), default="chinchilla")
    p.add_argument("--data", default=None, help="Trace directory; default depends on study")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--no-job-shift", action="store_true")
    p.add_argument("--token-quantum", dest="quantum", type=float, default=None)
    p = sub.add_parser("summarize")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()

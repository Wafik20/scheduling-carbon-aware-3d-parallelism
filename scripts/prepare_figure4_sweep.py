"""Prepare the original 1--8B Figure 4 sweep; evaluation requires --execute.

New eight-GPU measurements, historical May traces and original token-sweep
settings. Not a replication of the original hardware. Existing partial or
mismatched runs are refused, never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.workshop_reproduction import audit_traces, static_seconds

MODELS = ("gpt2", "llama_1b")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def build_manifest(source, data, out):
    source, data, out = map(lambda p: Path(p).resolve(), (source, data, out))
    audit = read_json(source / "audit.json")
    if audit["blocked"]:
        raise ValueError("Source profile audit is blocked")
    for path, expected in audit["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Audited source changed: {path}")
    traces, warnings = audit_traces(data, "workshop-fixed-start")
    if warnings:
        raise ValueError("; ".join(warnings))
    base = {w["model"]: w for w in read_json(source / "workloads.json")}
    fingerprints = {str(source / name): digest(source / name)
                    for name in ("audit.json", "workloads.json")}
    fingerprints.update(audit["source_sha256"])
    for code in (ROOT / "evaluate_carbon_policy.py", ROOT / "src/carbon_aware_3d/carbon_aware_env.py"):
        fingerprints[str(code)] = digest(code)
    workloads = []
    for model in MODELS:
        info = base[model]
        profile = Path(info["profile"]).resolve()
        rows = list(csv.DictReader(profile.open()))
        if not rows or info["global_batch_size"] != 16 or info["seq_length"] != (1024 if model == "gpt2" else 256):
            raise ValueError(f"Unexpected profile batch/sequence: {model}")
        rates = [float(r["throughput_tok_s"]) for r in rows]
        if any(not math.isfinite(r) or r <= 0 for r in rates):
            raise ValueError(f"Invalid throughput: {model}")
        fingerprints[str(profile)] = digest(profile)
        for billion in range(1, 9):
            tokens = billion * 1_000_000_000
            deadline = math.ceil(1.25 * tokens / max(rates) / 3600)
            quantum = 5_000_000 if billion <= 3 else 10_000_000 if billion <= 7 else 25_000_000
            feasible = sum(static_seconds(r, tokens, 16 * info["seq_length"]) <= deadline * 3600 for r in rows)
            if not feasible:
                raise ValueError(f"No warmup-inclusive feasible static strategy: {model}/{billion}B")
            workloads.append(dict(model=model, slug=f"{model}_{billion}b", tokens=tokens,
                deadline_h=deadline, token_quantum=quantum, profile=str(profile),
                parameters=info["parameters"], seq_length=info["seq_length"],
                global_batch_size=16, feasible_static_strategies=feasible,
                quantum_warning=("Original quantum may round slow-strategy hourly progress to zero; retained for historical comparability"
                                 if quantum >= min(rates) * 3600 else "Warmup may reduce hourly progress; check quantum sensitivity")))
    cap = min(t["hours"] for t in traces.values()) - (1.5 * max(w["deadline_h"] for w in workloads) + 1 + 6)
    if cap <= 0:
        raise ValueError("No shared non-wrapping trace arrival window")
    for trace in traces.values():
        fingerprints[trace["path"]] = trace["sha256"]
    runs = []
    for w in workloads:
        for region, trace in traces.items():
            dest = out / "results" / w["slug"] / region
            cmd = [sys.executable, "-u", str(ROOT / "evaluate_carbon_policy.py"),
                "--trace", trace["path"], "--strategy-profile", w["profile"],
                "--model-name", w["model"], "--tokens", str(w["tokens"]),
                "--model-num-params", str(w["parameters"]), "--global-batch-size", "16",
                "--seq-length", str(w["seq_length"]), "--max-episode-hours", str(w["deadline_h"]),
                "--max-offset-h", str(cap), "--n-gpus", "8", "--n-episodes", "64",
                "--num-workers", "16", "--oracle-token-quantum", str(w["token_quantum"]),
                "--oracle-switch-cost-mode", "sim", "--forecast-horizon-h", "6",
                "--smart-lookahead-h", "6", "--seed", "0", "--n-detail-rollouts", "0",
                "--out", str(dest)]
            runs.append(dict(model=w["model"], tokens=w["tokens"], region=region,
                output=str(dest), command=cmd, profile_sha256=digest(w["profile"]),
                trace_sha256=trace["sha256"]))
    return dict(study="figure4-original-token-sweep", workloads=workloads, runs=runs,
        max_offset_h=cap, source_sha256=fingerprints, traces=traces,
        assumptions=["Deadline ceil(1.25 * tokens / fastest steady throughput / 3600); warmup feasibility checked separately",
        "64 paired episodes, seed 0, hourly decisions, six-hour forecast/lookahead; no job shifting or RL",
        "16 CPU workers; native math libraries limited to one thread",
        "New eight-GPU profiles; global batch 16, GPT2 sequence 1024, Llama sequence 256",
        "Missing reshard time falls back to zero, not a measured zero; target warmup retained",
        "GPU-only measured energy; regional simulation, not AWS hardware measurements",
        "Discretized full-future oracle; historical tier quanta, not a globally optimal oracle"])


def result_complete(run):
    dest = Path(run["output"])
    if not dest.exists():
        return False
    status_path = dest / "status.json"
    if not status_path.exists():
        raise ValueError(f"Partial run has no status; use a new output root: {dest}")
    status = read_json(status_path)
    if status.get("command") != run["command"] or status.get("profile_sha256") != run["profile_sha256"] or status.get("trace_sha256") != run["trace_sha256"]:
        raise ValueError(f"Mismatched run: {dest}")
    if status.get("status") != "completed" or status.get("returncode") != 0:
        raise ValueError(f"Partial/failed run refused; use a new output root: {dest}")
    for name in ("results.csv", "summary.csv"):
        path = dest / name
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Missing completed output: {path}")
    rows = list(csv.DictReader((dest / "results.csv").open()))
    policies = {}
    for row in rows:
        policies.setdefault(row["policy"], []).append(row)
    required = ["Greedy", "Fixed Start Time Dynamic Parallelism"]
    required += [p for p in policies if p.startswith("static(")]
    if len(required) < 3 or any(p not in policies or len(policies[p]) != 64 or len({r["episode"] for r in policies[p]}) != 64 for p in required):
        raise ValueError(f"Insufficient completed episodes: {dest}")
    return True


def prepare(source, data, out):
    out = Path(out).resolve()
    manifest = build_manifest(source, data, out)
    if out.exists():
        if not (out / "manifest.json").exists() or read_json(out / "manifest.json") != manifest:
            raise ValueError(f"Existing snapshot differs; use a new output root: {out}")
    else:
        out.mkdir(parents=True)
        write_json(out / "manifest.json", manifest)
        write_json(out / "workloads.json", manifest["workloads"])
        (out / "commands.txt").write_text("\n".join(shlex.join(r["command"]) for r in manifest["runs"]) + "\n")
    return manifest


def run(manifest, execute=False):
    # Preflight all existing runs before starting any work.
    pending = [r for r in manifest["runs"] if not result_complete(r)]
    for r in pending:
        print(shlex.join(r["command"]), flush=True)
    print(f"{len(manifest['runs']) - len(pending)} completed; {len(pending)} pending; execute={execute}", flush=True)
    if not execute:
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MPLBACKEND="Agg")
    for r in pending:
        dest = Path(r["output"])
        dest.mkdir(parents=True, exist_ok=False)
        status = dict(r, status="started", started_at=datetime.now(timezone.utc).isoformat())
        write_json(dest / "status.json", status)
        with (dest / "evaluation.log").open("w") as log:
            process = subprocess.Popen(r["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["pid"] = process.pid
            write_json(dest / "status.json", status)
            code = process.wait()
        status.update(status="completed" if code == 0 else "failed", returncode=code,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(dest / "status.json", status)
        if code:
            raise RuntimeError(f"Evaluation failed: {dest}")
        result_complete(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/figure4_full_sweep")
    parser.add_argument("--source-prepared", type=Path, default=ROOT / "runs/workshop_fixed_start_may2024_ready")
    parser.add_argument("--data", type=Path, default=ROOT / "data/recovered_workshop_may2024")
    parser.add_argument("--execute", action="store_true", help="Run pending evaluations (default only prepares and prints)")
    args = parser.parse_args()
    manifest = prepare(args.source_prepared, args.data, args.out)
    run(manifest, args.execute)


if __name__ == "__main__":
    main()

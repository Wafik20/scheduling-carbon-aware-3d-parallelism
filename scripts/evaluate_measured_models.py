"""CPU-only, fixed-start comparison of all measured placement configurations.

Joint optimization selects TP/PP/DP/MBS/placement, not a delayed start. The
configuration oracle is discretized and replayed, not a continuous optimum.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_carbon_policy import (
    load_strategy_profile, strategy_label, static_policy_label,
    default_config, load_carbon_trace, rollout_static,
    plot_comparison,
)
from experiments.workshop_reproduction import validate_row
from carbon_aware_3d.carbon_aware_env import rollout_oracle_zero_overhead

ALGORITHM_VERSION = "zero-overhead-rolling-dp-v1"
DEFAULT_TRACE = ROOT / "data/recovered_workshop_may2024/carbon_2024_05_hourly_aws_us-west-1.csv"
WORKER_BUDGET_BYTES = 22 * 1024**3
ARTIFACTS = ("summary.json", "results.csv", "schedules.csv", "comparison.png")

MODELS = {"gpt2", "gpt2_medium", "gpt2_large", "llama_1b", "llama_2b",
          "llama_3b", "pythia_410m", "pythia_1_4b", "pythia_2_8b"}
TUPLE = ["tp", "pp", "dp", "micro_batch_size", "placement_id"]
ASSUMPTIONS = [
    "All measured configurations retained in source order; no placement/order filtering.",
    "20 training tokens per canonical TP=PP=1 logged model parameter unless overridden.",
    "Deadline ceil(1.25 * fastest zero-overhead fixed completion hours) unless overridden.",
    "Joint means TP/PP/DP/MBS/placement selection, never delayed starts: offset zero for every policy.",
    "warmup_wall_s, warmup_iters, reshard_wall_time_s and simulator startup/build/switch/cache costs set to zero; measured save_s retained.",
    "May 2024 us-west-1 carbon trace repeats cyclically; wrap counts exclude an exact terminal boundary.",
    "Discretized hourly oracle with exact simulator replay and feasible static candidates; not a proof of global optimality.",
    "Memory estimate includes conservative DP array headroom plus 1 GiB per worker and 2 GiB coordinator; not a measured peak guarantee.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def workload_fingerprint(w):
    keys = ("model", "parameters", "tokens", "deadline_h", "token_quantum",
            "global_batch_size", "seq_length", "n_gpus", "action_count",
            "offset_h", "wrap_trace")
    inputs = {key: w[key] for key in keys}
    inputs.update(algorithm=ALGORITHM_VERSION, profile_sha256=digest(w["profile"]),
                  trace_sha256=digest(w["trace"]))
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def cached_summary(w):
    out = Path(w["out"])
    try:
        completion = json.loads((out / "completion.json").read_text())
        if completion["fingerprint"] != workload_fingerprint(w):
            return None
        if any(completion["artifacts"][name] != digest(out / name) for name in ARTIFACTS):
            return None
        return json.loads((out / "summary.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_completion(w):
    out = Path(w["out"])
    write_json(out / "completion.json", dict(
        fingerprint=workload_fingerprint(w),
        artifacts={name: digest(out / name) for name in ARTIFACTS},
    ))


def worker_limit(requested):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return min(requested, 16, cpus)


def run_pending(workloads, workers, save):
    """Admit jobs by their individual estimates, not the largest job's size."""
    from concurrent.futures import FIRST_COMPLETED, wait

    pending = list(workloads)
    if any(w["estimated_worker_bytes"] > WORKER_BUDGET_BYTES for w in pending):
        raise ValueError("Estimated worker exceeds 24 GiB budget; increase token quantum or reduce workload")
    if workers == 1:
        for w in pending:
            save(run_model(w))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        running = {}
        while pending or running:
            used = sum(running.values())
            for w in pending[:]:
                size = w["estimated_worker_bytes"]
                if len(running) < workers and used + size <= WORKER_BUDGET_BYTES:
                    running[pool.submit(run_model, w)] = size
                    used += size
                    pending.remove(w)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                del running[future]
                save(future.result())


def adaptive_quantum(tokens, rows):
    tier = 5_000_000 if tokens <= 3e9 else 10_000_000 if tokens <= 7e9 else 25_000_000 if tokens <= 12e9 else 50_000_000
    hourly = min(float(r["throughput_tok_s"]) * 3600 for r in rows)
    return max(1_000_000, min(tier, math.floor(hourly / 1_000_000) * 1_000_000))


def estimate_memory_bytes(tokens, quantum, deadline, actions):
    # int16 parents, hourly action tables, rolling float64 arrays and replay headroom.
    bins, hours = math.ceil(tokens / quantum) + 1, math.ceil(deadline) + 1
    return 8 * bins * hours + 16 * hours * actions + 128 * bins + 1024**3


def validate_menu(sweep):
    frame = pd.read_csv(sweep / "results.csv", keep_default_na=False)
    if len(frame) != 348 or set(frame.model) != MODELS:
        raise ValueError("Expected all nine models and exactly 348 measured rows")
    if frame.duplicated(["model", *TUPLE]).any():
        raise ValueError("Duplicate measured model/configuration tuple")
    manifest = json.loads((sweep / "manifest.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    for row in frame.to_dict("records"):
        validate_row(row)
        if (int(row["n_gpus"]), int(row["world_size"]), int(row["nodes"]), int(row["gpus_per_node"])) != (8, 8, 2, 4):
            raise ValueError("Expected two nodes of four GPUs")
        ranks = json.loads(row["rank_mapping"])
        if len(ranks) != 8 or {r["gpu_name"] for r in ranks} != {"NVIDIA A100-PCIE-40GB"}:
            raise ValueError("Unexpected recorded GPU hardware")
        if sorted(r["rank"] for r in ranks) != list(range(8)):
            raise ValueError("Invalid recorded ranks")
        case = cases[row["id"]]
        for field in ["model", *TUPLE, "seq_length"]:
            if str(row[field]) != str(case[field]):
                raise ValueError(f"Manifest mismatch: {row['id']} {field}")
        if int(row["global_batch_size"]) != int(manifest["global_batch_size"]):
            raise ValueError("Manifest global batch mismatch")
        status_path = sweep / "cases" / row["id"] / "attempts" / f"{int(row['attempt']):04}" / "status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed" or status.get("returncode") != 0:
            raise ValueError(f"Measurement not successful: {status_path}")
    return frame


def prepare(args):
    frame = validate_menu(args.sweep)
    selected = set(args.models or MODELS)
    if not selected <= MODELS:
        raise ValueError(f"Unknown models: {selected - MODELS}")
    trace_frame = pd.read_csv(args.trace)
    trace_t, trace_ci, _ = load_carbon_trace(args.trace)
    if len(trace_t) != 744 or any(trace_t[i+1]-trace_t[i] != 3600 for i in range(743)):
        raise ValueError("Expected complete 744-hour May trace")
    if set(trace_frame.year) != {2024} or set(trace_frame.month) != {5}:
        raise ValueError("Expected May 2024 trace")
    if not all(math.isfinite(c) and c >= 0 for c in trace_ci):
        raise ValueError("Invalid carbon values")
    workloads = []
    for model in sorted(MODELS):
        measured = frame[frame.model == model].copy()
        rows = measured.to_dict("records")
        batches = {(int(r["global_batch_size"]), int(r["seq_length"])) for r in rows}
        if len(batches) != 1:
            raise ValueError(f"Inconsistent batches for {model}")
        batch, seq = batches.pop()
        counts, sources = set(), []
        for row in rows:
            if int(row["tp"]) == int(row["pp"]) == 1:
                log = args.sweep / "cases" / row["id"] / "attempts" / f"{int(row['attempt']):04}" / "launcher.log"
                counts.update(int(v) for v in re.findall(r"number of parameters on .*?:\s*(\d+)", log.read_text()))
                sources.append({"path": str(log), "sha256": digest(log)})
        if len(counts) != 1:
            raise ValueError(f"Ambiguous canonical parameter count for {model}: {counts}")
        params = counts.pop()
        tokens = args.tokens if model in selected and args.tokens is not None else 20 * params
        fastest_h = min(float(r["save_s"]) + tokens / float(r["throughput_tok_s"]) for r in rows) / 3600
        deadline = args.deadline_hours if model in selected and args.deadline_hours is not None else math.ceil(1.25 * fastest_h)
        adaptive = adaptive_quantum(tokens, rows)
        quantum_override = args.oracle_token_quantum if model in selected else None
        quantum = quantum_override or adaptive
        if quantum_override is None:
            if estimate_memory_bytes(tokens, tokens, deadline, len(rows)) > 22 * 1024**3:
                raise ValueError(f"Even one token bin exceeds the memory budget for {model}; reduce deadline")
            while estimate_memory_bytes(tokens, quantum, deadline, len(rows)) > 22 * 1024**3:
                quantum += 1_000_000
        warning = ("Automatic quantum coarsened for 24 GiB memory budget; slow actions may round to zero hourly progress; no actions pruned."
                   if quantum > adaptive else "Check quantum sensitivity; discretization is approximate.")
        out = args.out / model
        out.mkdir(parents=True, exist_ok=True)
        for field in ["warmup_wall_s", "warmup_iters", "reshard_wall_time_s"]:
            measured[field] = 0
        profile = out / "strategy_profile.csv"
        measured.to_csv(profile, index=False)
        strategies = load_strategy_profile(profile)
        mapping = measured.copy()
        mapping.insert(0, "strategy_idx", range(len(mapping)))
        mapping["label"] = [strategy_label(s) for s in strategies]
        mapping.to_csv(out / "action_mapping.csv", index=False)
        w = dict(model=model, parameters=params, parameter_sources=sources, tokens=tokens,
                 deadline_h=deadline, fastest_static_h=fastest_h, token_quantum=quantum,
                 adaptive_token_quantum=adaptive, quantum_warning=warning,
                 global_batch_size=batch, seq_length=seq, n_gpus=8, action_count=len(rows),
                 profile=str(profile), trace=str(args.trace), out=str(out),
                 trace_sha256=digest(args.trace), trace_mean_g_per_kwh=float(trace_ci.mean()),
                 offset_h=0, wrap_trace=True, selected=model in selected,
                 estimated_worker_bytes=estimate_memory_bytes(tokens, quantum, deadline, len(rows)),
                 assumptions=ASSUMPTIONS)
        write_json(out / "workload.json", w)
        workloads.append(w)
    return workloads


def run_model(w):
    started = time.monotonic()
    print(f"Starting {w['model']}: {w['action_count']} actions, quantum={w['token_quantum']}, deadline={w['deadline_h']}h", flush=True)
    cfg = default_config(pretraining_tokens=w["tokens"], global_batch_size=w["global_batch_size"],
                         seq_length=w["seq_length"], n_gpus=8)
    cfg.strategies = load_strategy_profile(w["profile"])
    cfg.container_startup_s = cfg.fresh_build_s = cfg.switch_time_s = cfg.first_switch_cache_miss_s = 0.0
    trace_t, trace_ci, _ = load_carbon_trace(w["trace"])
    out = Path(w["out"])
    records, schedules = [], []
    best_sim, best_idx = None, None

    def record(policy, sim, index=None):
        completed = sim.tokens_completed >= w["tokens"] - 1e-6
        feasible = completed and sim.cursor_s <= w["deadline_h"] * 3600 + 1e-6
        row = dict(model=w["model"], policy=policy, strategy_idx=index, tokens=w["tokens"],
                   deadline_h=w["deadline_h"], action_count=w["action_count"], completed=completed,
                   feasible=feasible, carbon_g=sim.total_carbon_g, energy_kj=sim.total_energy_j / 1000,
                   completion_h=sim.cursor_s / 3600, tokens_done=sim.tokens_completed,
                   switches=sim.reshard_count, offset_h=0,
                   trace_wraps=max(0, math.ceil(sim.cursor_s / sim.trace_duration_s) - 1))
        records.append(row)
        previous_carbon = previous_energy = 0.0
        for rec in sim.history:
            item = dict(policy=policy, **asdict(rec))
            item.update({k: cfg.strategies[rec.strategy_idx][k] for k in TUPLE})
            item.update(interval_carbon_g=rec.carbon_g - previous_carbon,
                        interval_energy_j=rec.energy_j - previous_energy)
            schedules.append(item)
            previous_carbon, previous_energy = rec.carbon_g, rec.energy_j
        return feasible

    for i, strategy in enumerate(cfg.strategies):
        sim = rollout_static(cfg, trace_t, trace_ci, 0, i, max_hours=w["deadline_h"], wrap_trace=True)
        if record(static_policy_label(strategy), sim, i):
            if best_sim is None or sim.total_carbon_g < best_sim.total_carbon_g:
                best_sim, best_idx = sim, i
    oracle = rollout_oracle_zero_overhead(cfg, trace_t, trace_ci, 0, max_hours=w["deadline_h"],
                               token_quantum=w["token_quantum"], wrap_trace=True)
    oracle_feasible = record("configuration_oracle", oracle)
    pd.DataFrame(records).to_csv(out / "results.csv", index=False)
    pd.DataFrame(schedules).to_csv(out / "schedules.csv", index=False)
    rollouts = {"configuration_oracle": oracle}
    if best_sim is not None:
        rollouts["Best feasible fixed: " + strategy_label(cfg.strategies[best_idx])] = best_sim
    fig = plot_comparison(rollouts, cfg, title=w["model"] + ": same fixed start, all measured configurations")
    used_indices = sorted({rec.strategy_idx for sim in rollouts.values() for rec in sim.history})
    strategy_axis = fig.axes[1]
    strategy_axis.set_yticks(used_indices)
    strategy_axis.set_yticklabels([f"{i}: {strategy_label(cfg.strategies[i])}" for i in used_indices])
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    static_carbon = best_sim.total_carbon_g if best_sim is not None else None
    saving = (100 * (static_carbon - oracle.total_carbon_g) / static_carbon
              if oracle_feasible and static_carbon is not None and static_carbon > 0 else None)
    result = dict(records[-1], parameters=w["parameters"], token_quantum=w["token_quantum"],
                  feasible_static_count=sum(r["feasible"] for r in records[:-1]),
                  best_fixed_idx=best_idx, best_fixed_tuple=json.dumps({k: cfg.strategies[best_idx][k] for k in TUPLE}) if best_idx is not None else None,
                  best_fixed_carbon_g=static_carbon,
                  best_fixed_energy_kj=best_sim.total_energy_j / 1000 if best_sim is not None else None,
                  best_fixed_completion_h=best_sim.cursor_s / 3600 if best_sim is not None else None,
                  best_fixed_trace_wraps=max(0, math.ceil(best_sim.cursor_s / best_sim.trace_duration_s)-1) if best_sim is not None else None,
                  savings_pct=saving,
                  reason_if_infeasible="" if oracle_feasible else "Oracle did not finish by the common deadline")
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(out / "summary.json", result)
    write_completion(w)
    print(f"Finished {w['model']}: savings={saving}, feasible={oracle_feasible}, "
          f"elapsed={result['elapsed_seconds']:.2f}s", flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, default=ROOT / "runs/placement_sweep_8gpu")
    p.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    p.add_argument("--out", type=Path, default=ROOT / "runs/measured_models")
    p.add_argument("--models", nargs="+", help="Subset to execute; all nine menus are always validated")
    p.add_argument("--tokens", type=int)
    p.add_argument("--deadline-hours", type=float)
    p.add_argument("--oracle-token-quantum", "--token-quantum", type=float)
    p.add_argument("--workers", "--num-workers", type=int, default=4)
    p.add_argument("--dry-run", "--prepare-only", action="store_true")
    resume = p.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true", default=True)
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()
    for name in ["tokens", "deadline_hours", "oracle_token_quantum", "workers"]:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"{name} must be finite and positive")
    args.sweep, args.trace, args.out = args.sweep.resolve(), args.trace.resolve(), args.out.resolve()
    workloads = prepare(args)
    selected = [w for w in workloads if w["selected"]]
    workers = min(worker_limit(args.workers), len(selected))
    audit = dict(workloads=workloads, workers=workers, memory_budget_gib=24,
                 source_sha256={str(path): digest(path) for path in [args.sweep / "results.csv", args.sweep / "manifest.json", args.trace]},
                 assumptions=ASSUMPTIONS)
    write_json(args.out / "workloads.json", audit)
    print(f"Trace: {args.trace}; SHA-256={selected[0]['trace_sha256']}; "
          f"mean={selected[0]['trace_mean_g_per_kwh']:.6f} gCO2/kWh", flush=True)
    print(f"Validated 348 configurations across nine models; selected {len(selected)}, safe workers={workers}", flush=True)
    for w in selected:
        print(f"{w['model']}: tokens={w['tokens']}, deadline={w['deadline_h']}h, "
              f"quantum={w['token_quantum']} (adaptive={w['adaptive_token_quantum']}), "
              f"estimated worker={w['estimated_worker_bytes']/1024**3:.2f} GiB; {w['quantum_warning']}", flush=True)
    print("\n".join(ASSUMPTIONS), flush=True)
    print("Large workloads can take substantial CPU time; use --dry-run to inspect memory and workloads.", flush=True)
    if args.dry_run:
        return
    summaries = []
    def save(result):
        summaries.append(result)
        pd.DataFrame(summaries).to_csv(args.out / "summary.csv", index=False)
    pending = []
    for w in selected:
        cached = cached_summary(w) if args.resume else None
        if cached is not None:
            print(f"Resuming {w['model']}: verified completed artifacts", flush=True)
            save(cached)
        else:
            print(f"Scheduling {w['model']}: no matching completed fingerprint" if args.resume
                  else f"Scheduling {w['model']}: resume disabled", flush=True)
            pending.append(w)
    if pending:
        run_pending(pending, workers, save)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()

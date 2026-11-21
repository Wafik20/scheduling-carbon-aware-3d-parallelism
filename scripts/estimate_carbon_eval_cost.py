#!/usr/bin/env python3
"""Dry DP cost estimates; optional tiny synthetic oracle timing, never a full eval."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

GIB = 1024**3


def estimate(workload, *, episodes=32, regions=3, include_job_shift=False, quantum=None):
    q = quantum or workload["token_quantum"]
    if not math.isfinite(q) or q <= 0:
        raise ValueError("quantum must be finite and positive")
    bins = math.ceil(workload["tokens"] / q)
    hours = math.ceil(workload["deadline_h"])
    k = int(workload["strategies"])
    if min(hours, k, episodes, regions) <= 0:
        raise ValueError("positive dimensions required")
    states = (hours + 1) * (bins + 1) * k
    # Nine float64 T*K*K caches plus one K*K duration cache.
    cache = 8 * (9 * hours * k * k + k * k)
    conservative = 24 * states
    workers = min(16, max(1, (24 * GIB) // conservative))
    horizons = [hours] + (list(range(hours, 0, -1)) if include_job_shift else [])
    scans = sum(horizons) * (bins + 1) * k
    # All finite, unfinished states branch to K actions. Actual density unknown.
    transition_ceiling = scans * k
    return dict(model=workload["model"], workload=workload.get("slug", workload["model"]), quantum=q, bins=bins, hours=hours,
                strategies=k, states=states, arrays_gib=14 * states / GIB,
                transition_cache_gib=cache / GIB,
                conservative_gib=conservative / GIB, workers=workers,
                concurrent_arrays_gib=workers * (14 * states + cache) / GIB,
                over_budget=conservative > 24 * GIB,
                oracle_calls_per_episode=len(horizons), scans_per_episode=scans,
                transition_scenarios_per_episode={str(d): int(d * transition_ceiling)
                    for d in (0.01, 0.1, 0.5, 1.0)},
                repair_work_proxy_per_episode=sum(h*h*k for h in horizons),
                total_scans=scans * episodes * regions,
                episodes=episodes, regions=regions)


def benchmark():
    """Fixed tiny cases, no measured-profile or annual-trace evaluation."""
    import numpy as np
    from carbon_aware_3d.carbon_aware_env import SimConfig, rollout_oracle_dp
    results = []
    for hours, bins, k in [(6, 40, 3), (12, 100, 6)]:
        cfg = SimConfig(strategies=[dict(tp=1, pp=1, power_w=1000.+50*i,
            throughput_tok_s=1.+i*.2, warmup_wall_s=0., warmup_iters=0,
            save_s=0.) for i in range(k)], pretraining_tokens=hours*3600.,
            tokens_per_iter=1)
        trace_t = np.arange(48, dtype=float) * 3600
        trace_ci = 200 + 100*np.sin(np.arange(48))
        start = time.perf_counter()
        sim = rollout_oracle_dp(cfg, trace_t, trace_ci, 0., max_hours=hours,
            token_quantum=cfg.pretraining_tokens/bins, respect_deadline=True)
        elapsed = time.perf_counter()-start
        results.append(dict(hours=hours, bins=bins, strategies=k,
            seconds=elapsed, completed=bool(sim.done()),
            seconds_per_dense_scan=elapsed/(hours*(bins+1)*k)))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workloads", type=Path, default=Path("runs/workshop_tp_pp_dp_audit/workloads.json"))
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--regions", type=int, default=3)
    p.add_argument("--include-job-shift", action="store_true",
        help="fixed oracle plus joint oracle at every hourly start before deadline")
    p.add_argument("--quantum", type=float)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()
    rows = [estimate(w, episodes=args.episodes, regions=args.regions,
        include_job_shift=args.include_job_shift, quantum=args.quantum)
        for w in json.loads(args.workloads.read_text())]
    report = dict(estimates=rows, assumptions=[
        "Hourly decisions, no startup horizon deduction; 14-byte dense arrays.",
        "24 bytes/state is wrapper heuristic, not RSS guarantee; Python/trace overhead excluded.",
        "Density is finite unfinished fraction, not allocated-array fraction.",
        "Transitions O(T Q K^2); repair proxy O(T^2 K), not exact operation count.",
        "Outer starts summed over shrinking horizons; full sweep sequential models/regions."])
    if args.benchmark:
        report["tiny_oracle_benchmark"] = benchmark()
        rates = [b["seconds_per_dense_scan"] for b in report["tiny_oracle_benchmark"]]
        for row in rows:
            row["illustrative_cpu_hours_range"] = [
                row["total_scans"] * rate / 3600 for rate in (min(rates), max(rates))]
            row["illustrative_ideal_wall_hours_range"] = [
                hours / row["workers"] for hours in row["illustrative_cpu_hours_range"]]
        report["timing_warning"] = "Includes repair. Synthetic density differs; scaling seconds/scan is NOT a full-run prediction."
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

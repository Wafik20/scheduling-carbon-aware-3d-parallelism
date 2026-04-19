#!/usr/bin/env python3
"""
check_cluster_users.py
----------------------
One-shot check: who else is using the GPUs right now?

Called by the sweep script between runs (NOT as a daemon). Prints a
one-line summary to stdout and appends one row to a CSV log. Exit code
is 0 on clean, 1 on contamination — but the sweep script ignores the
exit code; we log only, never interrupt.

Contamination definition:
    Another user (username != --my-user) has a process using >5% GPU
    utilization on any GPU. The 5% threshold filters out:
      - idle CUDA contexts from crashed Jupyter kernels
      - monitoring tools (nvidia-smi itself, dcgm-exporter, etc.)
      - our own PowerMonitor's read handles
    while still catching anyone actually training.

Usage:
    python check_cluster_users.py --my-user $USER \
        --log contamination_log.csv \
        --tag "before_llama_3b_tp1_pp1_dp4"
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime


# Utilization threshold (percent) above which we count a GPU as busy.
# Keep this low enough to catch training (typically 70-100%) but high
# enough to ignore CUDA context warm-up (~1-3%) and monitoring handles.
UTIL_THRESHOLD = 5


def nvsmi(query, timeout=10):
    """Run an nvidia-smi query, return lines of stripped CSV. []
    on failure (driver hiccup, timeout, missing binary) — we prefer
    a missing sample to a crashed check."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def pid_to_user(pid):
    """Return username owning pid, or None if the process vanished /
    we can't read /proc. Using /proc directly beats `ps` on speed and
    handles container PID-namespace quirks better."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("Uid:"):
                    uid = int(line.split()[1])
                    import pwd
                    try:
                        return pwd.getpwuid(uid).pw_name
                    except KeyError:
                        return f"uid{uid}"
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def check(my_user):
    """
    Returns dict with the findings. Never raises on driver errors —
    returns empty/zero fields instead so the sweep doesn't abort.
    """
    # Map GPU UUID -> index so we can report indices, not UUIDs.
    uuid_to_idx = {}
    for line in nvsmi("gpu=index,uuid", timeout=5):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                uuid_to_idx[parts[1]] = int(parts[0])
            except ValueError:
                pass

    # Per-GPU utilization + power. We use this for two things:
    #   1. Apply the >5% threshold to decide "is this GPU busy"
    #   2. Report total cluster power as a ground-truth sanity check
    util_by_idx = {}
    power_by_idx = {}
    for line in nvsmi("gpu=index,utilization.gpu,power.draw", timeout=5):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            util_by_idx[idx] = int(parts[1])
            power_by_idx[idx] = float(parts[2])
        except ValueError:
            continue

    # All processes on GPUs right now.
    procs = []
    for line in nvsmi("compute-apps=gpu_uuid,pid,used_memory", timeout=5):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            uuid, pid, mem = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        procs.append((uuid_to_idx.get(uuid, -1), pid, mem))

    # Group by user. Contamination = any other user on a busy GPU.
    # We don't flag "other user holds a dead CUDA context on an idle
    # GPU" — that's the >5% util filter doing its job.
    other_users_busy = defaultdict(lambda: {"gpus": set(), "mem_mb": 0, "pids": []})
    my_gpus = set()
    my_mem = 0

    for gpu_idx, pid, mem in procs:
        user = pid_to_user(pid) or "unknown"
        util = util_by_idx.get(gpu_idx, 0)
        if user == my_user:
            my_gpus.add(gpu_idx)
            my_mem += mem
        elif util > UTIL_THRESHOLD:
            # Real contamination: another user AND their GPU is busy
            other_users_busy[user]["gpus"].add(gpu_idx)
            other_users_busy[user]["mem_mb"] += mem
            other_users_busy[user]["pids"].append(pid)

    contaminated = len(other_users_busy) > 0
    return {
        "contaminated": contaminated,
        "my_user": my_user,
        "my_gpus": sorted(my_gpus),
        "my_mem_mb": my_mem,
        "other_users": dict(other_users_busy),
        "per_gpu_util": util_by_idx,
        "per_gpu_power_w": power_by_idx,
        "total_power_w": round(sum(power_by_idx.values()), 1),
    }


def format_line(result, tag):
    """One-line human-readable summary for stdout."""
    if not result["contaminated"]:
        return (f"[check] {tag}  CLEAN  "
                f"(my_gpus={result['my_gpus']}, "
                f"total_power={result['total_power_w']}W)")

    parts = []
    for user, info in result["other_users"].items():
        parts.append(f"{user}@gpu{sorted(info['gpus'])}({info['mem_mb']}MB)")
    return (f"[check] {tag}  ⚠ CONTAMINATED  "
            f"others={'; '.join(parts)}  "
            f"total_power={result['total_power_w']}W")


def write_log_row(log_path, result, tag):
    """Append one CSV row. Creates header if new file."""
    new_file = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    fields = [
        "timestamp", "tag", "contaminated",
        "my_gpus", "my_mem_mb",
        "other_users", "other_gpus", "other_mem_mb", "other_pids",
        "per_gpu_util", "per_gpu_power_w", "total_power_w",
    ]

    # Flatten per-user dict into comma-joined strings for CSV friendliness.
    # The notebook can still reconstruct structure via .split(",").
    other_users = ",".join(sorted(result["other_users"].keys())) or "-"
    other_gpus = ",".join(str(g) for g in sorted(
        {g for info in result["other_users"].values() for g in info["gpus"]}
    )) or "-"
    other_mem = sum(info["mem_mb"] for info in result["other_users"].values())
    other_pids = ",".join(str(p) for info in result["other_users"].values()
                          for p in info["pids"]) or "-"
    per_util = ";".join(f"gpu{i}:{u}%" for i, u in sorted(result["per_gpu_util"].items()))
    per_power = ";".join(f"gpu{i}:{p:.0f}W" for i, p in sorted(result["per_gpu_power_w"].items()))

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "contaminated": "yes" if result["contaminated"] else "no",
        "my_gpus": ",".join(str(g) for g in result["my_gpus"]) or "-",
        "my_mem_mb": result["my_mem_mb"],
        "other_users": other_users,
        "other_gpus": other_gpus,
        "other_mem_mb": other_mem,
        "other_pids": other_pids,
        "per_gpu_util": per_util,
        "per_gpu_power_w": per_power,
        "total_power_w": result["total_power_w"],
    }

    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--my-user", default=os.environ.get("USER", ""),
                    help="Your username (defaults to $USER)")
    ap.add_argument("--log", default="contamination_log.csv",
                    help="CSV file to append to (default ./contamination_log.csv)")
    ap.add_argument("--tag", default="",
                    help="Label for this check (e.g. 'before_llama_3b_dp4'). "
                         "Written to the 'tag' column so you can join later.")
    args = ap.parse_args()

    if not args.my_user:
        print("ERROR: could not determine your username. Pass --my-user.",
              file=sys.stderr)
        sys.exit(2)

    result = check(args.my_user)
    print(format_line(result, args.tag), flush=True)
    write_log_row(args.log, result, args.tag)

    # Exit 1 on contamination so callers COULD branch on it if they want,
    # but the sweep script ignores this. Detect-only by default.
    sys.exit(1 if result["contaminated"] else 0)


if __name__ == "__main__":
    main()
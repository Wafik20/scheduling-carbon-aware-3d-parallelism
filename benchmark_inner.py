#!/usr/bin/env python3
"""
Inner benchmark script — runs INSIDE the NeMo container.
Calls torchrun for each config via subprocess. Container startup is paid once.

Measures the actual resharding time by parsing Megatron's internal timestamps:
  - "after megatron is initialized" → checkpoint loading begins
  - "before the start of training step" → checkpoint loaded, training begins
  The gap = resharding + checkpoint load time.
"""

import subprocess
import shutil
import time
import csv
import os
import re
import itertools
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────

MEGATRON_LM = "/u/data3/wat29/Megatron-LM"
WORK_DIR = Path("/u/data3/wat29/carbon_benchmark")
RESULTS_DIR = Path("/afs/cs.pitt.edu/usr0/wat29/scheduling-carbon-aware-3d-parallelism")
RESULTS_CSV = RESULTS_DIR / "reshard_times.csv"

# ── Model / training ─────────────────────────────────────────

N = 4
NUM_LAYERS = 24
HIDDEN_SIZE = 2048
NUM_HEADS = 16
SEQ_LEN = 1024
MICRO_BATCH = 2
GLOBAL_BATCH = 8

WARMUP_STEPS = 50
RESUME_STEPS = 5
LR_DECAY_ITERS = 1000


@dataclass
class Config:
    name: str
    dp: int
    tp: int
    pp: int

    def total(self) -> int:
        return self.dp * self.tp * self.pp

    def __str__(self) -> str:
        return f"{self.name}(DP={self.dp},TP={self.tp},PP={self.pp})"


def enumerate_configs(n: int) -> list[Config]:
    configs = []
    for dp in range(1, n + 1):
        for tp in range(1, n + 1):
            pp, rem = divmod(n, dp * tp)
            if rem == 0 and pp >= 1 and NUM_LAYERS % pp == 0:
                configs.append(Config(f"dp{dp}_tp{tp}_pp{pp}", dp, tp, pp))
    return configs


def build_flags(cfg: Config, save_dir: Path, load_dir: Path | None,
                train_iters: int) -> list[str]:
    flags = [
        f"--tensor-model-parallel-size={cfg.tp}",
        f"--pipeline-model-parallel-size={cfg.pp}",
        f"--num-layers={NUM_LAYERS}",
        f"--hidden-size={HIDDEN_SIZE}",
        f"--num-attention-heads={NUM_HEADS}",
        f"--seq-length={SEQ_LEN}",
        f"--max-position-embeddings={SEQ_LEN}",
        f"--micro-batch-size={MICRO_BATCH}",
        f"--global-batch-size={GLOBAL_BATCH}",
        f"--train-iters={train_iters}",
        "--eval-interval=1000",
        "--eval-iters=0",
        "--lr=1e-4",
        f"--lr-decay-iters={LR_DECAY_ITERS}",
        "--lr-decay-style=cosine",
        "--min-lr=1e-5",
        "--weight-decay=0.1",
        "--clip-grad=1.0",
        "--initial-loss-scale=1024",
        "--fp16",
        "--mock-data",
        "--log-interval=1",
        f"--save={save_dir}",
        f"--save-interval={train_iters}",
        "--ckpt-format=torch_dist",
        "--use-distributed-optimizer",
        "--dist-ckpt-optim-fully-reshardable",
    ]
    if load_dir is not None:
        flags += [
            f"--load={load_dir}",
            "--auto-detect-ckpt-format",
            "--override-opt_param-scheduler",
        ]
    return flags


def run_torchrun(cfg: Config, save_dir: Path, load_dir: Path | None,
                 train_iters: int, logfile: Path) -> int:
    """Launch torchrun as a subprocess (still inside the container)."""
    env = os.environ.copy()
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    flags = build_flags(cfg, save_dir, load_dir, train_iters)
    cmd = [
        "torchrun", f"--nproc_per_node={N}",
        "--master_port=29500",
        f"{MEGATRON_LM}/pretrain_gpt.py",
    ] + flags

    with open(logfile, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                text=True, env=env, cwd=MEGATRON_LM)
    return result.returncode


def parse_timestamps(logfile: Path) -> dict:
    """Extract key timestamps from Megatron logs."""
    info = {}
    fmt = "%Y-%m-%d %H:%M:%S.%f"

    for line in logfile.read_text().splitlines():
        if "after megatron is initialized" in line:
            m = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
            if m:
                info["megatron_init"] = datetime.strptime(m.group(1), fmt)

        if "loading distributed checkpoint" in line.lower():
            info["loading_line"] = line.strip()

        if "successfully loaded checkpoint" in line:
            info["loaded"] = True
            m = re.search(r"at iteration (\d+)", line)
            if m:
                info["resumed_iter"] = int(m.group(1))

        if "before the start of training step" in line:
            m = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
            if m:
                info["training_start"] = datetime.strptime(m.group(1), fmt)

        if "after training is done" in line:
            m = re.search(r"datetime: ([\d-]+ [\d:.]+)", line)
            if m:
                info["training_end"] = datetime.strptime(m.group(1), fmt)

        if "elapsed time per iteration" in line:
            m = re.search(r"elapsed time per iteration \(ms\): ([\d.]+)", line)
            if m:
                info.setdefault("iter_ms", []).append(float(m.group(1)))

    # Compute resharding time
    if "megatron_init" in info and "training_start" in info:
        delta = (info["training_start"] - info["megatron_init"]).total_seconds()
        info["reshard_seconds"] = delta

    return info


def check_errors(logfile: Path) -> list[str]:
    errors = []
    for line in logfile.read_text().splitlines():
        for pat in ["RuntimeError:", "AssertionError:", "KeyError:", "TypeError:"]:
            if pat in line and "error_injection" not in line and "import" not in line:
                errors.append(line.strip())
                break
    return errors[:5]


def train_source(cfg: Config) -> Path | None:
    save_dir = WORK_DIR / f"ckpt_src_{cfg.name}"
    logfile = WORK_DIR / f"train_{cfg.name}.log"

    if save_dir.exists() and list(save_dir.glob("iter_*")):
        print(f"    Reusing checkpoint: {save_dir}")
        return save_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"    Training {WARMUP_STEPS} steps with {cfg}...")
    t0 = time.perf_counter()
    rc = run_torchrun(cfg, save_dir, None, WARMUP_STEPS, logfile)
    elapsed = time.perf_counter() - t0

    if rc != 0:
        errs = check_errors(logfile)
        print(f"    FAILED (exit {rc})")
        for e in errs:
            print(f"      {e}")
        return None

    print(f"    Done in {elapsed:.1f}s")
    return save_dir


def benchmark_reshard(src_dir: Path, src: Config, dst: Config) -> dict:
    pair_id = f"{src.name}_to_{dst.name}"
    dst_dir = WORK_DIR / f"ckpt_{pair_id}"
    logfile = WORK_DIR / f"reshard_{pair_id}.log"

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    total_iters = WARMUP_STEPS + RESUME_STEPS
    print(f"    Resharding {src.name} → {dst.name}...")

    t0 = time.perf_counter()
    rc = run_torchrun(dst, dst_dir, src_dir, total_iters, logfile)
    wall = time.perf_counter() - t0

    if rc != 0:
        errs = check_errors(logfile)
        print(f"    FAILED (exit {rc})")
        for e in errs:
            print(f"      {e}")
        shutil.rmtree(dst_dir, ignore_errors=True)
        return {"error": "failed"}

    ts = parse_timestamps(logfile)
    reshard_s = ts.get("reshard_seconds", -1)
    resumed = ts.get("resumed_iter", "?")
    iter_ms = ts.get("iter_ms", [])
    avg_ms = sum(iter_ms) / len(iter_ms) if iter_ms else 0

    print(f"    OK — resumed at iter {resumed}")
    print(f"    Reshard time: {reshard_s:.1f}s | Wall: {wall:.1f}s | Iter: {avg_ms:.0f}ms")

    dst_mb = sum(f.stat().st_size for f in dst_dir.rglob("*") if f.is_file()) / 1e6
    shutil.rmtree(dst_dir, ignore_errors=True)

    return {
        "from_config": src.name, "to_config": dst.name,
        "from_dp": src.dp, "from_tp": src.tp, "from_pp": src.pp,
        "to_dp": dst.dp, "to_tp": dst.tp, "to_pp": dst.pp,
        "reshard_s": round(reshard_s, 2),
        "wall_clock_s": round(wall, 2),
        "avg_iter_ms": round(avg_ms, 2),
        "output_mb": round(dst_mb, 1),
    }


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    configs = enumerate_configs(N)

    print("=" * 60)
    print(f"Checkpoint Reshard Benchmark (inside container)")
    print(f"Model: GPT 1.3B | GPUs: {N}x L40")
    print(f"Warmup: {WARMUP_STEPS} steps | Resume: {RESUME_STEPS} steps")
    print("=" * 60)
    print(f"\nConfigs ({len(configs)}):")
    for c in configs:
        print(f"  {c}")

    # ── Phase 1: source checkpoints ──────────────────────────
    print("\nPHASE 1: Source checkpoints")
    print("-" * 40)
    src_ckpts: dict[str, Path] = {}
    for cfg in configs:
        print(f"\n[{cfg}]")
        ckpt = train_source(cfg)
        if ckpt:
            src_ckpts[cfg.name] = ckpt

    # ── Phase 2: reshard pairs ───────────────────────────────
    print(f"\nPHASE 2: Resharding pairs")
    print("-" * 40)

    fields = [
        "from_config", "to_config",
        "from_dp", "from_tp", "from_pp",
        "to_dp", "to_tp", "to_pp",
        "reshard_s", "wall_clock_s", "avg_iter_ms", "output_mb",
    ]

    results = []
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for src, dst in itertools.permutations(configs, 2):
            if src.name not in src_ckpts:
                continue
            print(f"\n[{src.name} → {dst.name}]")
            row = benchmark_reshard(src_ckpts[src.name], src, dst)
            if "error" not in row:
                w.writerow(row)
                f.flush()
                results.append(row)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if results:
        print(f"\n{'From':<18} {'To':<18} {'Reshard':>9} {'Wall':>9} {'Iter(ms)':>9}")
        print("-" * 65)
        for r in results:
            print(f"{r['from_config']:<18} {r['to_config']:<18} "
                  f"{r['reshard_s']:>8.1f}s {r['wall_clock_s']:>8.1f}s "
                  f"{r['avg_iter_ms']:>8.0f}")

        rs = [r["reshard_s"] for r in results]
        print(f"\nReshard: min={min(rs):.1f}s  max={max(rs):.1f}s  avg={sum(rs)/len(rs):.1f}s")
    else:
        print("No successful runs.")

    print(f"\nCSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
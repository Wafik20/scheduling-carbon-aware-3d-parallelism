"""
compute_power_multinode.py
--------------------------
Multi-node GPU power monitor designed for sweep experiments across
3D parallelism configurations. One sampler per node; per-node CSVs are
written under a sweep/config-specific directory and aggregated offline.

Differences from compute_power.py:
- Multi-node only (no single-node legacy path).
- maybe_start() is the only entry point; non-leader ranks always get None.
- Filenames are always rank{global_rank}.csv under a required log_dir.
- Adds optional `config_tag` to embed (TP, PP, DP) in every row, so a
  single combined dataframe across configs is unambiguous.
- Adds a wallclock_s column (seconds since start()) for easier alignment
  across nodes that booted samplers a few ms apart.

Usage from a training entrypoint (called on every rank — non-leaders no-op):

    from compute_power_multinode import PowerMonitor

    monitor = PowerMonitor.maybe_start(
        log_dir="power_logs/gpt2m-sweep_1729489/tp2_pp1_dp4",
        interval=0.5,
        config_tag="tp2_pp1_dp4",
    )
    # ... training loop ...
    if monitor:
        monitor.update_step(it)
    # ... at the end ...
    if monitor:
        monitor.stop()

Usage as a sidecar (one srun process per node):

    python compute_power_multinode.py \
        --log-dir power_logs/gpt2m-sweep_1729489/tp2_pp1_dp4 \
        --interval 0.5 \
        --config-tag tp2_pp1_dp4

Aggregation (offline):

    import glob, pandas as pd
    files = glob.glob("power_logs/<sweep>/<cfg>/rank*.csv")
    df = pd.concat(pd.read_csv(f) for f in files)
    cluster = df.groupby("wallclock_s")["total_power_W"].sum()
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import socket
import threading
import time
from datetime import datetime

from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetPowerUsage,
    nvmlInit,
    nvmlShutdown,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _is_node_leader() -> bool:
    """
    True if this process should sample power for its node.
    Order: LOCAL_RANK (torchrun) → SLURM_LOCALID (srun) → assume yes.
    """
    for var in ("LOCAL_RANK", "SLURM_LOCALID"):
        v = os.environ.get(var)
        if v is not None:
            return int(v) == 0
    return True


def _global_rank() -> int:
    """torchrun RANK → SLURM_PROCID → 0."""
    rank_str = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0"
    return int(rank_str)


# ── Monitor ────────────────────────────────────────────────────────────

class PowerMonitor:
    """Per-node NVML power sampler. Writes one CSV per node."""

    def __init__(
        self,
        log_dir: str,
        interval: float = 0.5,
        config_tag: str = "",
        hostname: str | None = None,
    ):
        if not log_dir:
            raise ValueError("log_dir is required for multi-node monitoring")

        self.log_dir = log_dir
        self.interval = float(interval)
        self.config_tag = config_tag
        self.hostname = hostname or socket.gethostname().split(".")[0]

        self.rank = _global_rank()
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        self.slurm_node_id = os.environ.get("SLURM_NODEID", "")

        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"rank{self.rank}.csv")

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._readings: list[tuple[str, float, int, float]] = []
        self._t0: float | None = None

        self._current_step = 0
        self._step_lock = threading.Lock()

    # ── Class entry point ──────────────────────────────────────────────

    @classmethod
    def maybe_start(cls, **kwargs) -> "PowerMonitor | None":
        """
        Start a monitor only on the per-node leader. All other ranks get
        None and should guard their stop/update_step calls.
        """
        if not _is_node_leader():
            return None
        monitor = cls(**kwargs)
        monitor.start()
        return monitor

    # ── Public API ─────────────────────────────────────────────────────

    def update_step(self, step: int) -> None:
        with self._step_lock:
            self._current_step = int(step)

    def start(self) -> None:
        nvmlInit()
        self._n_gpus = nvmlDeviceGetCount()
        self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self._n_gpus)]
        self._t0 = time.monotonic()
        print(
            f"[PowerMonitor:rank{self.rank}@{self.hostname}] "
            f"Sampling {self._n_gpus} GPU(s) every {self.interval}s "
            f"-> {self.log_file}"
            + (f"  [config={self.config_tag}]" if self.config_tag else ""),
            flush=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self._stop_evt.is_set():
            self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join()
        try:
            nvmlShutdown()
        except NVMLError:
            pass
        print(
            f"[PowerMonitor:rank{self.rank}@{self.hostname}] "
            f"Stopped. {len(self._readings)} samples collected.",
            flush=True,
        )

    def results(self) -> tuple[float, list]:
        if not self._readings:
            return 0.0, []
        powers = [r[3] for r in self._readings]  # (ts, wallclock, step, total_W)
        return sum(powers) / len(powers), self._readings

    # ── Background loop ────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        gpu_cols = [f"GPU_{i}_W" for i in range(self._n_gpus)]
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "wallclock_s",
                    "rank",
                    "hostname",
                    "slurm_job_id",
                    "slurm_node_id",
                    "config_tag",
                    "time_step",
                    "n_gpus",
                    "total_power_W",
                ]
                + gpu_cols
            )

            while not self._stop_evt.is_set():
                ts = datetime.now().isoformat()
                wallclock = round(time.monotonic() - self._t0, 3)

                with self._step_lock:
                    step = self._current_step

                per_gpu_W = []
                for h in self._handles:
                    try:
                        per_gpu_W.append(nvmlDeviceGetPowerUsage(h) / 1000.0)
                    except NVMLError:
                        per_gpu_W.append(0.0)

                total_W = sum(per_gpu_W)
                self._readings.append((ts, wallclock, step, total_W))

                writer.writerow(
                    [
                        ts,
                        wallclock,
                        self.rank,
                        self.hostname,
                        self.slurm_job_id,
                        self.slurm_node_id,
                        self.config_tag,
                        step,
                        self._n_gpus,
                        round(total_W, 2),
                    ]
                    + [round(w, 2) for w in per_gpu_W]
                )
                f.flush()

                # Sleep with early-exit on stop signal so shutdown is snappy
                self._stop_evt.wait(self.interval)


# ── Sidecar mode ───────────────────────────────────────────────────────

def _run_as_sidecar() -> None:
    """
    Standalone process that samples until SIGTERM/SIGINT.
    Intended to be launched once per node via:
        srun --overlap --ntasks-per-node=1 python compute_power_multinode.py ...
    """
    parser = argparse.ArgumentParser(
        description="Per-node GPU power sampler for multi-node sweeps."
    )
    parser.add_argument("--log-dir", required=True,
                        help="Directory to write rank<N>.csv into.")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Sampling interval in seconds (default: 0.5).")
    parser.add_argument("--config-tag", default="",
                        help="Optional tag (e.g. tp2_pp1_dp4) embedded in CSV rows.")
    args = parser.parse_args()

    monitor = PowerMonitor.maybe_start(
        log_dir=args.log_dir,
        interval=args.interval,
        config_tag=args.config_tag,
    )
    if monitor is None:
        # Non-leader rank on this node — stay alive but idle, so srun's
        # ntasks-per-node=1 doesn't trip on an early exit.
        signal.pause()
        return

    stop_evt = threading.Event()

    def _handle(_signum, _frame):
        stop_evt.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    print(
        f"[sidecar] Sampling on {monitor.hostname} (rank {monitor.rank}, "
        f"{monitor._n_gpus} GPUs) every {monitor.interval}s. "
        f"Waiting for SIGTERM/SIGINT.",
        flush=True,
    )
    stop_evt.wait()
    monitor.stop()


if __name__ == "__main__":
    _run_as_sidecar()
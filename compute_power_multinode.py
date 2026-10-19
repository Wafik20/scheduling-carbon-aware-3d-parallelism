"""Per-node NVML sidecar; shared sampling implementation lives in the package.

Legacy in-process calls write rank<N>.csv. Supplying --ready-dir enables
wrapper mode: hostname.csv and ready-dir/hostname after the first flush.
GPU_<i>_W columns correspond to the ordered semicolon-separated gpu_uuids.
"""
from __future__ import annotations

import argparse
import os
import signal
import threading
from pathlib import Path

from carbon_aware_3d.compute_power import PowerMonitor as NodePowerMonitor


def _is_node_leader():
    for key in ("LOCAL_RANK", "SLURM_LOCALID"):
        if key in os.environ:
            return int(os.environ[key]) == 0
    return True


def _global_rank():
    return int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "0")


class PowerMonitor(NodePowerMonitor):
    def __init__(self, log_dir, interval=0.5, config_tag="", hostname=None,
                 ready_dir=None, expected_gpus=None):
        if not log_dir:
            raise ValueError("log_dir is required for multi-node monitoring")
        self.rank = int(os.environ.get("SLURM_PROCID", "0")) if ready_dir else _global_rank()
        self.slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
        self.slurm_node_id = os.environ.get("SLURM_NODEID", "")
        self.log_dir = log_dir
        self.config_tag = config_tag
        super().__init__(interval=interval, hostname=hostname, ready_dir=ready_dir,
                         whole_job=bool(ready_dir), expected_gpus=expected_gpus)
        filename = f"{self.hostname}.csv" if ready_dir else f"rank{self.rank}.csv"
        self.log_file = str(Path(log_dir, filename))

    @classmethod
    def maybe_start(cls, **kwargs):
        if not kwargs.get("ready_dir") and not _is_node_leader():
            return None
        monitor = cls(**kwargs)
        monitor.start()
        return monitor

    def _metadata(self):
        return dict(super()._metadata(), rank=self.rank,
                    slurm_node_id=os.environ.get("SLURM_NODEID", ""),
                    config_tag=self.config_tag)

    def _reading(self, timestamp, elapsed, step, total):
        return timestamp, elapsed, step, total


def _run_as_sidecar():
    parser = argparse.ArgumentParser(description="Per-node GPU power sampler")
    parser.add_argument("--log-dir")
    parser.add_argument("--diagnose", action="store_true", help="Print allocation mapping and exit without sampling")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--config-tag", default="")
    parser.add_argument("--ready-dir")
    parser.add_argument("--stop-file")
    parser.add_argument("--expected-gpus", type=int)
    args = parser.parse_args()
    if args.diagnose:
        from carbon_aware_3d import compute_power
        if compute_power.nvml is None:
            parser.error("nvidia-ml-py is required")
        compute_power.nvml.nvmlInit()
        try:
            compute_power.allocated_handles(whole_job=True)
        finally:
            compute_power.nvml.nvmlShutdown()
        return
    if not args.log_dir:
        parser.error("--log-dir is required unless --diagnose is used")
    stop_event = threading.Event()
    def handle_signal(_signum, _frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    monitor = PowerMonitor.maybe_start(log_dir=args.log_dir, interval=args.interval,
                                      config_tag=args.config_tag, ready_dir=args.ready_dir,
                                      expected_gpus=args.expected_gpus)
    if monitor is None:
        raise RuntimeError("Sidecar must run once per node with local rank zero")
    try:
        while not stop_event.wait(0.1):
            monitor.check_error()
            if args.stop_file and Path(args.stop_file).exists():
                break
    finally:
        monitor.stop()


if __name__ == "__main__":
    _run_as_sidecar()

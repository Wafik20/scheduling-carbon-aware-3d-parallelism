"""NVML node power sampling shared by in-process and Slurm sidecar monitors.

Slurm numeric GPU IDs are device minor numbers, unlike CUDA_VISIBLE_DEVICES
(which may be remapped by cgroups). UUID visibility is unambiguous. Numeric
visibility without Slurm allocation metadata is rejected. MIG is unsupported:
parent-device power cannot be attributed to individual MIG allocations.
"""
from __future__ import annotations

import csv
import math
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import pynvml as nvml
except ModuleNotFoundError as exc:
    nvml = None
    _NVML_IMPORT_ERROR = exc
else:
    _NVML_IMPORT_ERROR = None


def _tokens(value):
    result = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith(("GPU-", "MIG-")):
            result.append(token)
        elif token.isdigit():
            result.append(int(token))
        elif "-" in token and all(part.isdigit() for part in token.split("-")):
            first, last = map(int, token.split("-"))
            result.extend(range(first, last + 1))
        else:
            raise ValueError(f"Unsupported GPU allocation identifier: {token!r}")
    return result


def allocated_handles(*, whole_job=False):
    """Resolve node allocation, never interpret CUDA-local ordinals as host IDs."""
    # Some Slurm sites populate only STEP_GPUS, even for a step covering
    # the full allocation. Empty JOB_GPUS is not an explicit zero-GPU grant.
    keys = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS") if whole_job else (
        "SLURM_STEP_GPUS", "SLURM_JOB_GPUS")
    allocation = next((os.environ[key].strip() for key in keys
                       if os.environ.get(key, "").strip()), None)
    if allocation is None and any(key in os.environ for key in keys):
        raise ValueError("No allocated GPUs available in Slurm GPU metadata")
    if allocation is not None:
        ids = _tokens(allocation)
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            ids = _tokens(visible)
            if any(isinstance(item, int) for item in ids):
                raise ValueError("Numeric CUDA_VISIBLE_DEVICES is ambiguous without Slurm GPU IDs; use GPU UUIDs")
        elif os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Slurm GPU allocation metadata is missing; refusing to sample all node GPUs")
        else:
            ids = list(range(nvml.nvmlDeviceGetCount()))
    if not ids:
        raise ValueError("No allocated GPUs available for power monitoring")
    handles, uuids = [], []
    # Slurm numeric GPU IDs refer to /dev/nvidia minor numbers, not CUDA or
    # NVML enumeration order. Cgroups may hide devices or reorder enumeration.
    slurm_allocation = allocation is not None
    physical_handles = {}
    if slurm_allocation and any(isinstance(item, int) for item in ids):
        for index in range(nvml.nvmlDeviceGetCount()):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                physical_handles[nvml.nvmlDeviceGetMinorNumber(handle)] = handle
            except nvml.NVMLError_NoPermission:
                continue
    for identifier in ids:
        if str(identifier).startswith("MIG-"):
            raise ValueError("MIG power attribution is unsupported")
        if isinstance(identifier, int):
            if slurm_allocation:
                if identifier not in physical_handles:
                    raise ValueError(f"Slurm GPU minor {identifier} is not visible through NVML; use UUIDs")
                handle = physical_handles[identifier]
            else:
                handle = nvml.nvmlDeviceGetHandleByIndex(identifier)
        else:
            handle = nvml.nvmlDeviceGetHandleByUUID(identifier)
        try:
            mig_mode, _ = nvml.nvmlDeviceGetMigMode(handle)
        except nvml.NVMLError_NotSupported:
            mig_mode = 0
        if mig_mode:
            raise ValueError("MIG-enabled GPU power attribution is unsupported")
        uuid = nvml.nvmlDeviceGetUUID(handle)
        if isinstance(uuid, bytes):
            uuid = uuid.decode()
        if uuid not in uuids:
            handles.append(handle)
            uuids.append(uuid)
    return handles, uuids


class PowerMonitor:
    """One node sampler. Read failures are NaN, never fabricated zero watts."""

    def __init__(self, interval=1.0, log_file="power_logs/power_log.csv", *,
                 hostname=None, ready_dir=None, whole_job=False):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.log_file = str(log_file)
        self.hostname = hostname or os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
        self.ready_dir = ready_dir
        self.whole_job = whole_job
        self._stop_evt = threading.Event()
        self._thread = None
        self._readings = []
        self._current_step = 0
        self._step_lock = threading.Lock()
        self._error = None
        self._initialized = False
        self._file = None

    def update_step(self, step):
        with self._step_lock:
            self._current_step = int(step)

    def _metadata(self):
        return {"hostname": self.hostname,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}

    def _reading(self, timestamp, elapsed, step, total):
        return timestamp, step, total

    def start(self):
        if nvml is None:
            raise ImportError("nvidia-ml-py is required for GPU power monitoring") from _NVML_IMPORT_ERROR
        if self._initialized or self._stop_evt.is_set():
            raise RuntimeError("Monitor cannot be started twice")
        try:
            nvml.nvmlInit()
            self._initialized = True
            self._handles, self.gpu_uuids = allocated_handles(whole_job=self.whole_job)
            self._n_gpus = len(self._handles)
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.log_file, "w", newline="")
            fields = ["timestamp", "elapsed_s", "wallclock_s", *self._metadata(),
                      "time_step", "n_gpus", "gpu_uuids", "total_power_W"]
            fields += [f"GPU_{i}_W" for i in range(self._n_gpus)]
            self._writer = csv.DictWriter(self._file, fieldnames=fields)
            self._writer.writeheader()
            self._t0 = time.monotonic()
            self._sample()  # Synchronous first flushed sample makes readiness meaningful.
            if self.ready_dir:
                Path(self.ready_dir).mkdir(parents=True, exist_ok=True)
                Path(self.ready_dir, self.hostname).touch()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
        except BaseException:
            self._cleanup()
            raise

    def _sample(self):
        timestamp = datetime.now(timezone.utc).isoformat()
        elapsed = time.monotonic() - self._t0
        with self._step_lock:
            step = self._current_step
        powers = []
        for handle in self._handles:
            try:
                powers.append(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except nvml.NVMLError:
                powers.append(float("nan"))
        total = sum(powers)
        row = dict(timestamp=timestamp, elapsed_s=elapsed, wallclock_s=elapsed,
                   **self._metadata(), time_step=step, n_gpus=self._n_gpus,
                   gpu_uuids=";".join(self.gpu_uuids), total_power_W=total)
        row.update({f"GPU_{i}_W": power for i, power in enumerate(powers)})
        self._writer.writerow(row)
        self._file.flush()
        self._readings.append(self._reading(timestamp, elapsed, step, total))

    def _monitor_loop(self):
        try:
            while not self._stop_evt.wait(self.interval):
                self._sample()
        except BaseException as exc:
            self._error = exc
            self._stop_evt.set()

    def check_error(self):
        if self._error is not None:
            raise RuntimeError("Power sampler failed") from self._error

    def _cleanup(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._initialized:
            self._initialized = False
            nvml.nvmlShutdown()

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
        self._cleanup()
        self.check_error()

    def results(self):
        if not self._readings:
            return 0.0, []
        return sum(r[-1] for r in self._readings) / len(self._readings), self._readings

"""NVML power sampling with UUID-based Slurm allocation isolation.

CUDA resolves remapped ordinals under Slurm device constraints. NVML inventory
is diagnostic only, never evidence that a GPU belongs to an allocation.
MIG power attribution is unsupported.
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


def _slurm_device_isolation():
    """Require Slurm device constraints and membership in this job's step cgroup."""
    import re
    import subprocess

    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit():
        return False
    membership = Path("/proc/self/cgroup").read_text()
    if not re.search(rf"/job_{job}/step_[^/\n]+(?:/|$)", membership):
        return False
    config = subprocess.run(["scontrol", "show", "config"], check=True,
                            capture_output=True, text=True, timeout=10).stdout
    settings = dict(line.split("=", 1) for line in config.splitlines() if "=" in line)
    settings = {key.strip(): value.strip() for key, value in settings.items()}
    print(f"[GPU isolation:{socket.gethostname()}] "
          f"TaskPlugin={settings.get('TaskPlugin')} "
          f"ConstrainDevices={settings.get('ConstrainDevices')} "
          f"cgroup={membership.strip()!r}", flush=True)
    return ("task/cgroup" in settings.get("TaskPlugin", "").split(",")
            and settings.get("ConstrainDevices", "").lower() == "yes")


def _cuda_visible_uuids():
    """Ask the CUDA driver to resolve its visible ordinals, without Torch."""
    import ctypes
    import uuid

    cuda = ctypes.CDLL("libcuda.so.1")
    def check(code):
        if code != 0:
            raise RuntimeError(f"CUDA GPU identity query failed (driver error {code})")
    check(cuda.cuInit(0))
    count = ctypes.c_int()
    check(cuda.cuDeviceGetCount(ctypes.byref(count)))
    result = []
    for index in range(count.value):
        device = ctypes.c_int()
        check(cuda.cuDeviceGet(ctypes.byref(device), index))
        identity = (ctypes.c_ubyte * 16)()
        check(cuda.cuDeviceGetUuid(ctypes.byref(identity), device))
        result.append("GPU-" + str(uuid.UUID(bytes=bytes(identity))))
    return result


def allocated_handles(*, whole_job=False):
    """Resolve CUDA-visible UUIDs under verified Slurm device constraints.

    Slurm numeric IDs are allocation identifiers, not NVML indices/minors.
    STEP_GPUS is node-local; JOB_GPUS may describe a different numbering scope.
    """
    import json

    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
    def diagnostic(**values):
        print(f"[GPU mapping:{hostname}] " + json.dumps(values, sort_keys=True), flush=True)
    variables = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_NODEID", "SLURM_JOB_GPUS",
                 "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")
    diagnostic(environment={key: os.environ.get(key) for key in variables})
    inventory = []
    for index in range(nvml.nvmlDeviceGetCount()):
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            identity = nvml.nvmlDeviceGetUUID(handle)
            if isinstance(identity, bytes):
                identity = identity.decode()
            inventory.append(dict(index=index, minor=nvml.nvmlDeviceGetMinorNumber(handle), uuid=identity))
        except nvml.NVMLError as exc:
            inventory.append(dict(index=index, error=str(exc)))
    diagnostic(nvml_devices=inventory)
    try:
        step = os.environ.get("SLURM_STEP_GPUS", "").strip()
        job = os.environ.get("SLURM_JOB_GPUS", "").strip()
        allocation = step or job
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if allocation:
            ids = _tokens(allocation)
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate Slurm allocation identifiers")
            if whole_job and step and job and len(_tokens(step)) != len(_tokens(job)):
                raise ValueError("Monitoring step GPU count differs from job allocation; cannot establish whole-job coverage")
            if any(str(item).startswith("MIG-") for item in ids):
                raise ValueError("MIG power attribution is unsupported")
            if all(isinstance(item, str) and item.startswith("GPU-") for item in ids):
                selected = ids
                source = "Slurm UUID allocation"
            else:
                if not visible or not _slurm_device_isolation():
                    raise ValueError("Numeric Slurm GPU ownership requires CUDA visibility and verified Slurm device cgroup isolation")
                selected = _cuda_visible_uuids()
                if len(selected) != len(ids) or len(set(selected)) != len(ids):
                    raise ValueError("CUDA-visible UUID count does not match Slurm allocation; ownership is ambiguous")
                source = "CUDA UUIDs within Slurm device cgroup"
        elif os.environ.get("SLURM_JOB_ID") or any(key in os.environ for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")):
            raise ValueError("No allocated GPUs: Slurm GPU allocation metadata is missing")
        elif visible is not None:
            selected = _tokens(visible)
            if any(not isinstance(item, str) for item in selected):
                raise ValueError("Numeric CUDA_VISIBLE_DEVICES is ambiguous without Slurm GPU ownership")
            source = "explicit UUID visibility"
        else:
            selected = [item["uuid"] for item in inventory if "uuid" in item]
            source = "non-Slurm unrestricted devices"
        if not selected:
            raise ValueError("No allocated GPUs available for power monitoring")
        handles, uuids = [], []
        for identity in selected:
            if identity.startswith("MIG-"):
                raise ValueError("MIG power attribution is unsupported")
            handle = nvml.nvmlDeviceGetHandleByUUID(identity)
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
        diagnostic(source=source, selected_uuids=uuids)
        return handles, uuids
    except Exception as exc:
        diagnostic(mapping_error=str(exc))
        raise


class PowerMonitor:
    """One node sampler. Read failures are NaN, never fabricated zero watts."""

    def __init__(self, interval=1.0, log_file="power_logs/power_log.csv", *,
                 hostname=None, ready_dir=None, whole_job=False, expected_gpus=None):
        self.interval = float(interval)
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.log_file = str(log_file)
        self.hostname = hostname or os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
        self.ready_dir = ready_dir
        self.whole_job = whole_job
        self.expected_gpus = expected_gpus
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
            if self.expected_gpus is not None and self._n_gpus != self.expected_gpus:
                raise ValueError(f"{self.hostname}: expected {self.expected_gpus} allocated GPUs, "
                                 f"resolved {self._n_gpus}; refusing readiness")
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

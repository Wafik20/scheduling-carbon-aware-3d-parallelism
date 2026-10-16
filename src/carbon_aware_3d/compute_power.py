"""
compute_power.py
----------------
Monitors total GPU cluster power consumption using nvidia-ml-py.
Runs as a background thread during training.

The current training step can be fed in from the host process via
`update_step(i)` so that every power reading in the CSV is tagged
with the iteration that was running when the sample was taken.

Usage:
    from carbon_aware_3d.compute_power import PowerMonitor

    monitor = PowerMonitor(interval=1.0, log_file="power_logs/gpt2/tp1_pp1_dp4.csv")
    monitor.start()

    # ... in your training loop or stdout parser ...
    monitor.update_step(42)

    monitor.stop()
    avg_power, readings = monitor.results()
"""

import csv
import os
import threading
import time
from datetime import datetime

try:
    from pynvml import (
        NVMLError,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlInit,
        nvmlShutdown,
    )
except ModuleNotFoundError as exc:
    NVMLError = Exception
    nvmlDeviceGetCount = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetPowerUsage = None
    nvmlInit = None
    nvmlShutdown = None
    _NVML_IMPORT_ERROR = exc
else:
    _NVML_IMPORT_ERROR = None


class PowerMonitor:
    def __init__(self, interval=1.0, log_file="power_logs/power_log.csv"):
        self.interval = interval
        self.log_file = log_file
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._readings = []

        # Thread-safe training step tracker
        self._current_step = 0
        self._step_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────

    def update_step(self, step: int):
        """Called from the main thread to report the current training iteration."""
        with self._step_lock:
            self._current_step = step

    def start(self):
        if _NVML_IMPORT_ERROR is not None:
            raise ImportError(
                "nvidia-ml-py is required for GPU power monitoring. "
                "Install it with `python -m pip install nvidia-ml-py`."
            ) from _NVML_IMPORT_ERROR
        nvmlInit()
        self._n_gpus = nvmlDeviceGetCount()
        self._handles = [nvmlDeviceGetHandleByIndex(i) for i in range(self._n_gpus)]
        print(f"[PowerMonitor] Monitoring {self._n_gpus} GPU(s) every {self.interval}s")
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join()
        nvmlShutdown()
        print(f"[PowerMonitor] Stopped. {len(self._readings)} samples collected.")

    def results(self):
        if not self._readings:
            return 0.0, []
        powers = [r[2] for r in self._readings]
        avg_power = sum(powers) / len(powers)
        return avg_power, self._readings

    # ── Background loop ───────────────────────────────────────────────

    def _monitor_loop(self):
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            gpu_cols = [f"GPU_{i}_W" for i in range(self._n_gpus)]
            writer.writerow(["timestamp", "time_step", "total_power_W"] + gpu_cols)

            while not self._stop_evt.is_set():
                ts = datetime.now().isoformat()

                with self._step_lock:
                    step = self._current_step

                per_gpu_W = []
                for h in self._handles:
                    try:
                        mw = nvmlDeviceGetPowerUsage(h)
                        per_gpu_W.append(mw / 1000.0)
                    except NVMLError:
                        per_gpu_W.append(0.0)

                total_W = sum(per_gpu_W)
                self._readings.append((ts, step, total_W))
                writer.writerow(
                    [ts, step, round(total_W, 2)]
                    + [round(w, 2) for w in per_gpu_W]
                )
                f.flush()
                time.sleep(self.interval)


if __name__ == "__main__":
    log_path = "power_logs/test_power_log.csv"
    monitor = PowerMonitor(interval=1.0, log_file=log_path)
    monitor.start()
    print("Sampling for 10 seconds (simulating steps)...")
    for fake_step in range(10):
        monitor.update_step(fake_step)
        time.sleep(1)
    monitor.stop()
    avg, readings = monitor.results()
    print(f"\nAverage total cluster power: {avg:.2f} W")
    print(f"Samples collected         : {len(readings)}")
    print(f"Log saved to              : {log_path}")

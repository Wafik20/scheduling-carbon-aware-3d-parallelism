"""
compute_power.py
----------------
Monitors total GPU cluster power consumption using nvidia-ml-py.
Runs as a background thread during training.

Usage:
    from compute_power import PowerMonitor

    monitor = PowerMonitor(interval=1.0, log_file="power_logs/gpt2/tp1_pp1_dp4.csv")
    monitor.start()

    # ... run your training here ...

    monitor.stop()
    avg_power, readings = monitor.results()
"""

import csv
import os
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


class PowerMonitor:
    def __init__(self, interval=1.0, log_file="power_logs/power_log.csv"):
        self.interval = interval
        self.log_file = log_file
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._readings = []

    def start(self):
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

    def _monitor_loop(self):
        os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            gpu_cols = [f"GPU_{i}_W" for i in range(self._n_gpus)]
            writer.writerow(["timestamp", "total_power_W"] + gpu_cols)

            while not self._stop_evt.is_set():
                ts = datetime.now().isoformat()
                per_gpu_W = []

                for h in self._handles:
                    try:
                        mw = nvmlDeviceGetPowerUsage(h)
                        per_gpu_W.append(mw / 1000.0)
                    except NVMLError:
                        per_gpu_W.append(0.0)

                total_W = sum(per_gpu_W)
                self._readings.append((ts, total_W))
                writer.writerow([ts, round(total_W, 2)] + [round(w, 2) for w in per_gpu_W])
                f.flush()  # Ensure data is written in case of crash
                time.sleep(self.interval)

    def results(self):
        if not self._readings:
            return 0.0, []
        powers = [r[1] for r in self._readings]
        avg_power = sum(powers) / len(powers)
        return avg_power, self._readings


if __name__ == "__main__":
    log_path = "power_logs/test_power_log.csv"
    monitor = PowerMonitor(interval=1.0, log_file=log_path)
    monitor.start()
    print("Sampling for 10 seconds...")
    time.sleep(10)
    monitor.stop()
    avg, readings = monitor.results()
    print(f"\nAverage total cluster power: {avg:.2f} W")
    print(f"Samples collected         : {len(readings)}")
    print(f"Log saved to              : {log_path}")
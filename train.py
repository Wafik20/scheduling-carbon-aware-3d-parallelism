# test_monitor.py
import torch
import time
from compute_power import PowerMonitor

monitor = PowerMonitor(interval=0.5, log_file="test_power.csv")
monitor.start()

# Simple GPU stress test to generate real power draw
print("Running GPU workload...")
device = torch.device("cuda")
for _ in range(100):
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(0.1)

monitor.stop()
avg_power, readings = monitor.results()
print(f"Samples collected : {len(readings)}")
print(f"Avg cluster power : {avg_power:.2f} W")
"""Checks for the ownership guard and CUDA identity API without GPU hardware."""
from types import SimpleNamespace
from unittest.mock import Mock
import ctypes
import subprocess

import pytest
from carbon_aware_3d import compute_power as power


@pytest.mark.parametrize("config,member,expected", [
    ("TaskPlugin = task/cgroup,task/affinity\nConstrainDevices = yes", "/job_123/step_0/user/task_0", True),
    ("TaskPlugin = task/cgroup\nConstrainDevices = no", "/job_123/step_0", False),
    ("TaskPlugin = task/affinity\nConstrainDevices = yes", "/job_123/step_0", False),
    ("TaskPlugin = task/cgroup\nConstrainDevices = yes", "/job_999/step_0", False),
])
def test_device_isolation_guard(monkeypatch, config, member, expected):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(power.Path, "read_text", lambda self: "0::" + member)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=config))
    assert power._slurm_device_isolation() is expected


def test_cuda_uuid_resolution(monkeypatch):
    def count(ptr):
        ptr._obj.value = 2
        return 0
    def device(ptr, index):
        ptr._obj.value = index
        return 0
    def identity(ptr, dev):
        for i in range(16):
            ptr._obj[i] = dev.value + 1
        return 0
    driver = SimpleNamespace(cuInit=lambda flags: 0, cuDeviceGetCount=count,
                             cuDeviceGet=device, cuDeviceGetUuid=identity)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    assert power._cuda_visible_uuids() == [
        "GPU-01010101-0101-0101-0101-010101010101",
        "GPU-02020202-0202-0202-0202-020202020202"]
    driver.cuInit = lambda flags: 999
    with pytest.raises(RuntimeError, match="driver error"):
        power._cuda_visible_uuids()

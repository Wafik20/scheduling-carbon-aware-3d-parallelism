from carbon_aware_3d.compute_power import PowerMonitor


def test_results_average_watts_not_training_steps():
    monitor = PowerMonitor()
    monitor._readings = [("first", 10, 300.0), ("second", 20, 500.0)]

    average_watts, readings = monitor.results()

    assert average_watts == 400.0
    assert readings == monitor._readings


def test_results_without_samples():
    assert PowerMonitor().results() == (0.0, [])


import csv
import math
from types import SimpleNamespace

import pytest
from carbon_aware_3d import compute_power as power


class NVMLFailure(Exception):
    pass


class Unsupported(NVMLFailure):
    pass


@pytest.fixture
def fake_nvml(monkeypatch):
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_JOB_ID",
                "CUDA_VISIBLE_DEVICES", "SLURMD_NODENAME"):
        monkeypatch.delenv(key, raising=False)
    state = SimpleNamespace(closed=0, failures=set(), mig=False,
                            isolated=True, cuda_uuids=["GPU-0", "GPU-2"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-2")
    monkeypatch.setattr(power, "_slurm_device_isolation", lambda: state.isolated)
    monkeypatch.setattr(power, "_cuda_visible_uuids", lambda: state.cuda_uuids)
    def shutdown():
        state.closed += 1
    def measure(handle):
        if handle in state.failures:
            raise NVMLFailure("unavailable")
        return (handle + 1) * 100000
    fake = SimpleNamespace(
        NVMLError=NVMLFailure, NVMLError_NotSupported=Unsupported,
        NVMLError_NoPermission=PermissionError,
        nvmlInit=lambda: None, nvmlShutdown=shutdown,
        nvmlDeviceGetCount=lambda: 4,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMinorNumber=lambda h: [2, 0, 3, 1][h],
        nvmlDeviceGetHandleByUUID=lambda uuid: int(uuid.split("-")[1]),
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}",
        nvmlDeviceGetMigMode=lambda h: (int(state.mig), 0),
        nvmlDeviceGetPowerUsage=measure)
    monkeypatch.setattr(power, "nvml", fake)
    return state


def test_numeric_allocation_uses_cuda_uuids_not_device_minors(fake_nvml, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_GPUS", "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    # All minor numbers disagree with the Slurm allocation identifiers.
    monkeypatch.setattr(power.nvml, "nvmlDeviceGetMinorNumber", lambda handle: 10 + handle)
    assert power.allocated_handles() == ([0, 2], ["GPU-0", "GPU-2"])
    monkeypatch.setenv("SLURM_STEP_GPUS", "7,9")
    assert power.allocated_handles(whole_job=True)[0] == [0, 2]
    monkeypatch.setenv("SLURM_STEP_GPUS", "7")
    fake_nvml.cuda_uuids = ["GPU-2"]
    assert power.allocated_handles()[0] == [2]
    with pytest.raises(ValueError):
        power.allocated_handles(whole_job=True)


def test_uuid_visibility_and_ambiguous_visibility(fake_nvml, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2,GPU-0")
    assert power.allocated_handles()[0] == [2, 0]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="ambiguous"):
        power.allocated_handles()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(ValueError, match="metadata"):
        power.allocated_handles()


def test_empty_allocation_fails_and_cleans_up(fake_nvml, monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_GPUS", "")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monitor = PowerMonitor(log_file=tmp_path / "power.csv", ready_dir=tmp_path / "ready")
    with pytest.raises(ValueError, match="No allocated GPUs"):
        monitor.start()
    assert fake_nvml.closed == 1
    assert not (tmp_path / "ready").exists()


def test_mig_rejected(fake_nvml, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-example")
    with pytest.raises(ValueError, match="MIG"):
        power.allocated_handles()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")
    fake_nvml.mig = True
    with pytest.raises(ValueError, match="MIG"):
        power.allocated_handles()


def test_csv_readiness_nan_and_legacy_tuple(fake_nvml, monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_GPUS", "2,3")
    monkeypatch.setenv("SLURMD_NODENAME", "node42")
    fake_nvml.failures.add(2)
    monitor = PowerMonitor(interval=60, log_file=tmp_path / "power.csv",
                           ready_dir=tmp_path / "ready")
    monitor.update_step(7)
    monitor.start()
    assert (tmp_path / "ready" / "node42").exists()
    with open(monitor.log_file) as stream:
        rows = list(csv.DictReader(stream))
    monitor.stop()
    monitor.stop()
    assert fake_nvml.closed == 1
    row = rows[0]
    assert row["timestamp"].endswith("+00:00")
    assert row["hostname"] == "node42"
    assert row["gpu_uuids"] == "GPU-0;GPU-2"
    assert float(row["GPU_0_W"]) == 100
    assert math.isnan(float(row["GPU_1_W"]))
    assert math.isnan(float(row["total_power_W"]))
    avg, readings = monitor.results()
    assert math.isnan(avg)
    assert len(readings[0]) == 3 and readings[0][1] == 7


def test_thread_error_surfaces_and_cleanup(fake_nvml, monkeypatch, tmp_path):
    monitor = PowerMonitor(interval=60, log_file=tmp_path / "power.csv")
    monitor.start()
    monitor._stop_evt.set()
    monitor._thread.join()
    monitor._stop_evt.clear()
    monkeypatch.setattr(monitor._stop_evt, "wait", lambda interval: False)
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(monitor, "_sample", fail)
    monitor._monitor_loop()
    with pytest.raises(RuntimeError, match="sampler failed"):
        monitor.stop()
    assert fake_nvml.closed == 1


def test_multinode_legacy_and_wrapper_names(fake_nvml, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "compute_power_multinode", Path(__file__).parents[1] / "compute_power_multinode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    legacy = module.PowerMonitor(log_dir=tmp_path, hostname="node", interval=60)
    assert legacy.log_file.endswith("rank0.csv")
    legacy.start()
    legacy.stop()
    assert len(legacy.results()[1][0]) == 4
    wrapper = module.PowerMonitor(log_dir=tmp_path, hostname="node", ready_dir=tmp_path / "ready")
    assert wrapper.log_file.endswith("node.csv")


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_bad_interval(interval):
    with pytest.raises(ValueError):
        PowerMonitor(interval=interval)


@pytest.mark.parametrize("job_ids", [None, "", "   "])
def test_crc_step_only_remapped_nvml_index(fake_nvml, monkeypatch, tmp_path, job_ids):
    monkeypatch.setenv("SLURM_JOB_ID", "4446291")
    monkeypatch.setenv("SLURM_STEP_GPUS", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    if job_ids is not None:
        monkeypatch.setenv("SLURM_JOB_GPUS", job_ids)
    uuid = "GPU-1d4782de-6603-406a-8226-d506a22f1a08"
    fake_nvml.cuda_uuids = [uuid]
    monkeypatch.setattr(power.nvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(power.nvml, "nvmlDeviceGetMinorNumber", lambda h: 0)
    monkeypatch.setattr(power.nvml, "nvmlDeviceGetHandleByUUID", lambda u: 0)
    monkeypatch.setattr(power.nvml, "nvmlDeviceGetUUID", lambda h: uuid)
    assert power.allocated_handles(whole_job=True) == ([0], [uuid])
    monitor = PowerMonitor(log_file=tmp_path / "crc.csv", whole_job=True)
    monitor.start()
    monitor.stop()
    with open(monitor.log_file) as stream:
        row = next(csv.DictReader(stream))
    assert row["gpu_uuids"] == uuid
    assert row["n_gpus"] == "1"
    assert float(row["total_power_W"]) == float(row["GPU_0_W"]) == 100


@pytest.mark.parametrize("physical_ids", [(3, 5), (1, 7)])
def test_step_only_multi_gpu_node_local_remapping(fake_nvml, monkeypatch, physical_ids):
    monkeypatch.setenv("SLURM_JOB_GPUS", "")
    monkeypatch.setenv("SLURM_STEP_GPUS", ",".join(map(str, physical_ids)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_nvml.cuda_uuids = ["GPU-1", "GPU-0"]
    assert power.allocated_handles(whole_job=True) == ([1, 0], ["GPU-1", "GPU-0"])


@pytest.mark.parametrize("isolated,uuids,visible", [
    (False, ["GPU-2"], "0"),
    (True, [], "0"),
    (True, ["GPU-0", "GPU-2"], "0"),
    (True, ["GPU-2"], ""),
])
def test_numeric_allocation_fails_closed(fake_nvml, monkeypatch, isolated, uuids, visible):
    monkeypatch.setenv("SLURM_STEP_GPUS", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    fake_nvml.isolated = isolated
    fake_nvml.cuda_uuids = uuids
    with pytest.raises(ValueError):
        power.allocated_handles(whole_job=True)


def test_step_fallback_excludes_other_visible_nvml_gpus(fake_nvml, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_GPUS", "")
    monkeypatch.setenv("SLURM_STEP_GPUS", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    fake_nvml.cuda_uuids = ["GPU-1"]
    assert power.allocated_handles(whole_job=True) == ([1], ["GPU-1"])


def test_missing_gpu_prevents_readiness(fake_nvml, monkeypatch, tmp_path):
    monkeypatch.setattr(power, "allocated_handles", lambda **kwargs: ([0], ["GPU-0"]))
    monitor = PowerMonitor(log_file=tmp_path / "power.csv", ready_dir=tmp_path / "ready",
                           expected_gpus=2)
    with pytest.raises(ValueError, match="expected 2 allocated GPUs"):
        monitor.start()
    assert not (tmp_path / "ready").exists()
    assert fake_nvml.closed == 1

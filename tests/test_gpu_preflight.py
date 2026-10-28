import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from carbon_aware_3d import gpu_preflight as pre


def mappings():
    return [dict(rank=r, local_rank=r // 2, hostname=f"node{r % 2}",
                 gpu_uuid=f"GPU-{r}", pci_bus_id=f"0000:{r:02x}:00.0") for r in range(4)]


def test_valid_mapping():
    pre.validate_mappings(mappings(), 4, 2, 2)


def test_duplicate_gpu():
    records = mappings()
    records[3]["gpu_uuid"] = records[1]["gpu_uuid"]
    with pytest.raises(ValueError, match=r"duplicate GPU assignments on node1.*1, 3"):
        pre.validate_mappings(records, 4, 2, 2)


@pytest.mark.parametrize("job, duplicate_ranks", [("4578577", None), ("4579126", [2, 3])])
def test_recorded_multinode_allocations(job, duplicate_ranks):
    # Reduced copies of saved pre-NCCL rank records; no live Slurm dependency.
    path = Path(__file__).parent / "fixtures" / "gpu_preflight" / f"{job}.json"
    records = json.loads(path.read_text())
    if duplicate_ranks is None:
        pre.validate_mappings(records, 4, 2, 2)
    else:
        first, second = (records[r] for r in duplicate_ranks)
        assert first["gpu_uuid"] == second["gpu_uuid"]
        assert first["pci_bus_id"] == second["pci_bus_id"]
        assert first["environment"]["SLURM_GPUS_ON_NODE"] == "1"
        with pytest.raises(ValueError, match=r"duplicate GPU assignments on gpu-n30.*2, 3"):
            pre.validate_mappings(records, 4, 2, 2)


def test_missing_rank_and_node_count():
    with pytest.raises(ValueError, match="incomplete"):
        pre.validate_mappings(mappings()[:-1], 4, 2, 2)
    records = mappings()
    records[3]["hostname"] = "node0"
    with pytest.raises(ValueError, match="expected 2 distinct"):
        pre.validate_mappings(records, 4, 2, 2)


def test_timeout_preserves_artifacts(tmp_path):
    pre.write_json(tmp_path / "rank0.json", mappings()[0])
    now = [0.0]
    def advance(seconds):
        now[0] += seconds
    with pytest.raises(TimeoutError, match=r"missing ranks \[1, 2, 3\]"):
        pre.wait_for_mappings(tmp_path, 4, timeout=.2,
                              clock=lambda: now[0], sleep=advance)
    assert now[0] == pytest.approx(.2)
    assert json.loads((tmp_path / "rank0.json").read_text()) == mappings()[0]


def test_arrivals_and_rank_failure(tmp_path):
    records = mappings()
    def arrive(_):
        for record in records:
            pre.write_json(tmp_path / f"rank{record['rank']}.json", record)
    assert pre.wait_for_mappings(tmp_path, 4, sleep=arrive) == records
    pre.write_json(tmp_path / "rank2.json", dict(rank=2, error="CUDA failed"))
    with pytest.raises(ValueError, match="rank 2 failed: CUDA failed"):
        pre.wait_for_mappings(tmp_path, 4)


def test_slurm_snapshot_unavailable_and_timeout(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 5
        if len(calls) == 1:
            raise FileNotFoundError("scontrol unavailable")
        raise pre.subprocess.TimeoutExpired(command, 5)
    monkeypatch.setattr(pre.subprocess, "run", run)
    result = pre.slurm_snapshot("123", "1")
    assert set(result) == {"job", "step", "config"}
    assert all("unavailable" in item for item in result.values())


@pytest.mark.parametrize("duplicate", [False, True])
def test_preflight_saves_identity_and_validation(tmp_path, monkeypatch, duplicate):
    import sys
    from carbon_aware_3d import compute_power
    env = dict(SLURM_JOB_ID="123", SLURM_STEP_ID="1", PROFILE_MAPPING_RUN="fresh",
               RANK="3", SLURM_LOCALID="1", SLURM_JOB_NUM_NODES="2",
               POWER_GPUS_PER_NODE="2", CUDA_VISIBLE_DEVICES="0", SLURM_STEP_GPUS="1,6")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(pre.socket, "gethostname", lambda: "node1")
    uuid = "GPU-1" if duplicate else "GPU-3"
    monkeypatch.setattr(compute_power, "_cuda_visible_uuids", lambda: [uuid])
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(
        nvmlInit=lambda: None, nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda identity: identity,
        nvmlDeviceGetPciInfo=lambda handle: SimpleNamespace(busId=b"0000:4c:00.0")))
    directory = tmp_path / "gpu_preflight/123.1-fresh"
    directory.mkdir(parents=True)
    for record in mappings()[:-1]:
        pre.write_json(directory / f"rank{record['rank']}.json", record)
    if duplicate:
        with pytest.raises(ValueError, match="duplicate GPU"):
            pre.run_preflight(SimpleNamespace(world_size=4), tmp_path, lambda args: 0)
    else:
        pre.run_preflight(SimpleNamespace(world_size=4), tmp_path, lambda args: 0)
    record = json.loads((directory / "rank3.json").read_text())
    assert record["gpu_uuid"] == uuid
    assert record["pci_bus_id"] == "0000:4c:00.0"
    assert record["environment"]["SLURM_STEP_GPUS"] == "1,6"
    assert "cgroup" in record and "mountinfo" in record
    assert json.loads((directory / "validation-rank3.json").read_text())["ok"] is not duplicate


def test_selection_failure_is_saved_before_wait(tmp_path, monkeypatch):
    for key, value in dict(SLURM_JOB_ID="123", SLURM_STEP_ID="1", PROFILE_MAPPING_RUN="failed",
                           RANK="1", SLURM_LOCALID="1").items():
        monkeypatch.setenv(key, value)
    def fail(args):
        raise ValueError("bad visible ordinal")
    import sys
    monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace())
    with pytest.raises(ValueError, match="bad visible ordinal"):
        pre.run_preflight(SimpleNamespace(world_size=4), tmp_path, fail)
    directory = tmp_path / "gpu_preflight/123.1-failed"
    assert "bad visible ordinal" in json.loads((directory / "rank1.json").read_text())["error"]
    assert not json.loads((directory / "validation-rank1.json").read_text())["ok"]

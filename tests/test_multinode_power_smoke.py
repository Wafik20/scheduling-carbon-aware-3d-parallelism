import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validator", ROOT / "scripts/validate_multinode_power_smoke.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


@pytest.fixture(params=[1, 2, 4])
def artifacts(tmp_path, request):
    gpus_per_node = request.param
    world_size = 2 * gpus_per_node
    (tmp_path / "layout.json").write_text(json.dumps(dict(
        world_size=world_size, nodes=["node0", "node1"], gpus_per_node=gpus_per_node)))
    ranks = tmp_path / "ranks"
    ranks.mkdir()
    logs = tmp_path / "job_123/run_test"
    logs.mkdir(parents=True)
    for rank in range(world_size):
        (ranks / f"rank{rank}.json").write_text(json.dumps(dict(
            rank=rank, world_size=world_size, hostname=f"node{rank // gpus_per_node}",
            local_rank=rank % gpus_per_node, cuda_device=0, visible_device_count=1,
            steps=100, elapsed_s=90)))
    for node in range(2):
        rows = [dict(timestamp="2026-10-18T00:00:00+00:00", elapsed_s=t,
                     hostname=f"node{node}", n_gpus=gpus_per_node,
                     gpu_uuids=";".join(f"GPU-{node * gpus_per_node + i}" for i in range(gpus_per_node)),
                     total_power_W=100 * gpus_per_node, **{f"GPU_{i}_W": 100 for i in range(gpus_per_node)})
                for t in (0, 90)]
        with (logs / f"node{node}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return tmp_path


def test_valid_rank_power_artifacts(artifacts):
    validator.validate(artifacts)


@pytest.mark.parametrize("field,value", [("total_power_W", "nan"), ("total_power_W", "399"),
                                         ("n_gpus", "3"), ("gpu_uuids", "GPU-a;GPU-b;GPU-c;GPU-d")])
def test_bad_power_artifacts_rejected(artifacts, field, value):
    path = artifacts / "job_123/run_test/node0.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0][field] = value
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        validator.validate(artifacts)


@pytest.mark.parametrize("extra", [False, True])
def test_wrong_gpu_columns_rejected(artifacts, extra):
    path = artifacts / "job_123/run_test/node0.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if extra:
            row["GPU_99_W"] = "100"
        else:
            del row["GPU_0_W"]
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="GPU power columns"):
        validator.validate(artifacts)


def test_node_visible_device_mapping(artifacts):
    layout = json.loads((artifacts / "layout.json").read_text())
    for path in (artifacts / "ranks").glob("rank*.json"):
        rank = json.loads(path.read_text())
        rank.update(cuda_device=rank["local_rank"], visible_device_count=layout["gpus_per_node"])
        path.write_text(json.dumps(rank))
    validator.validate(artifacts)


@pytest.mark.parametrize("count,device", [(0, 0), (1, 1), (4, 1), (4, 4)])
def test_bad_device_mapping_rejected(artifacts, count, device):
    path = artifacts / "ranks/rank0.json"
    rank = json.loads(path.read_text())
    rank.update(visible_device_count=count, cuda_device=device)
    path.write_text(json.dumps(rank))
    with pytest.raises(ValueError, match="CUDA device mapping"):
        validator.validate(artifacts)


def test_missing_rank_rejected(artifacts):
    (artifacts / "ranks/rank0.json").unlink()
    with pytest.raises(ValueError, match="exactly .* ranks"):
        validator.validate(artifacts)


@pytest.mark.parametrize("mutation", ["missing_node", "wrong_node", "world_size", "duplicate_local_rank"])
def test_rank_layout_mismatch_rejected(artifacts, mutation):
    paths = sorted((artifacts / "ranks").glob("rank*.json"))
    if mutation == "missing_node":
        for path in paths:
            if json.loads(path.read_text())["hostname"] == "node1":
                path.unlink()
    else:
        path = paths[0]
        rank = json.loads(path.read_text())
        if mutation == "wrong_node":
            rank["hostname"] = "unexpected-node"
        elif mutation == "world_size":
            rank["world_size"] += 1
        else:
            rank["local_rank"] = 1
        path.write_text(json.dumps(rank))
    with pytest.raises(ValueError):
        validator.validate(artifacts)


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("socket_mode", [False, True])
def test_batch_composes_existing_wrappers_and_propagates_failure(tmp_path, world_size, socket_mode):
    import os
    import subprocess
    import sys
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (scripts / "run_container.sh").touch()
    (scripts / "run_with_power.sh").write_text(
        '#!/bin/bash\nenv -0 > "$SLURM_SUBMIT_DIR/env"\nprintf "%s\\0" "$@" > "$SLURM_SUBMIT_DIR/argv"\nexit 17\n')
    scontrol = bindir / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "node0\\nnode1\\n"\n')
    scontrol.chmod(0o755)
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_NODELIST='node[0-1]',
               SLURM_JOB_ID='123', SLURM_NTASKS=str(world_size), POWER_PYTHON=sys.executable,
               SMOKE_NCCL_SOCKET=str(int(socket_mode)), SMOKE_SOCKET_IFNAME='ib0',
               SLURM_CPU_BIND='mask_cpu:0x1', SLURM_CPU_BIND_TYPE='mask_cpu',
               SLURM_CPU_BIND_LIST='0x1', SLURM_CPU_BIND_VERBOSE='verbose',
               PATH=str(bindir) + os.pathsep + os.environ['PATH'])
    result = subprocess.run(['bash', str(ROOT / 'scripts/slurm_multinode_power_smoke.sh')],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 17, result.stderr
    args = (tmp_path / 'argv').read_bytes().decode().split('\0')[:-1]
    assert args[0] == 'srun'
    assert '--nodes=2' in args and f'--ntasks={world_size}' in args
    assert f'--ntasks-per-node={world_size // 2}' in args and '--gpus-per-task=1' in args
    assert '--gpu-bind=single:1' in args and '--overlap' in args
    assert args.count('--cpu-bind=none') == 1
    assert f'--gpus-per-node={world_size // 2}' in args
    assert args[args.index('--duration') + 1] == '90'
    assert ('scripts/nccl_socket_smoke.py' if socket_mode else 'scripts/smoke_train_multinode.py') in args
    if socket_mode:
        environment = dict(line.split('=', 1) for line in (tmp_path / 'env').read_text().split('\0')[:-1])
        for key, value in dict(NCCL_NET='Socket', NCCL_NET_PLUGIN='none', NCCL_IB_DISABLE='1',
                              NCCL_COLLNET_ENABLE='0', NCCL_DMABUF_ENABLE='0', NCCL_DEBUG='INFO',
                              NCCL_SOCKET_IFNAME='=ib0').items():
            assert environment[key] == environment['APPTAINERENV_' + key] == value

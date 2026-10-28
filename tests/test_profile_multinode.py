"""Exercise batch command composition without Slurm, Apptainer, or GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
BATCH = ROOT / "batch_files/profile_multinode.sbatch"


@pytest.mark.parametrize("hca", ["", "mlx5_0"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_batch_composes_per_gpu_ib_profile(tmp_path, hca, world_size):
    batch_dir = tmp_path / "batch_files"
    batch_dir.mkdir()
    (batch_dir / BATCH.name).write_text(BATCH.read_text())
    batch = ROOT / f"batch_files/profile_multinode_{world_size}gpu.sbatch" if world_size in (2, 4) else BATCH
    gpus_per_node = world_size // 2
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_with_power.sh").write_text('exec "$@"\n')
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "gpu-a\\ngpu-b\\n"\n')
    scontrol.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "print(json.dumps(dict(argv=sys.argv[1:], env=dict(os.environ))))\n")
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_NODELIST="gpu-[a,b]",
               SLURM_NTASKS=str(world_size), SLURM_JOB_ID="12345", SLURM_CPUS_PER_TASK="6",
               SLURM_MEM_PER_CPU="1", SLURM_MEM_PER_GPU="2", SLURM_MEM_PER_NODE="3",
               NCCL_IB_HCA="stale", APPTAINERENV_NCCL_IB_HCA="stale",
               SMOKE_NCCL_IB_HCA=hca)
    env.pop("MASTER_PORT", None)
    result = subprocess.run(["bash", str(batch), "--work-dir", "runs/with spaces"],
                            env=env, capture_output=True, text=True, check=True)
    captured = json.loads(result.stdout)
    args, effective = captured["argv"], captured["env"]
    for option in ("--nodes=2", f"--ntasks={world_size}", f"--ntasks-per-node={gpus_per_node}",
                   f"--gpus-per-node={gpus_per_node}", "--gpus-per-task=1", "--gpu-bind=single:1",
                   "--gres-flags=allow-task-sharing",
                   "--cpus-per-task=6", "--cpu-bind=none", "--overlap", "--exact"):
        assert option in args
    assert "--gpu-bind=none" not in args
    assert "--distribution=block" in args
    if world_size == 4:
        # Slurm block distribution fills each node before assigning the next.
        ranks_by_node = [list(range(start, start + gpus_per_node))
                         for start in range(0, world_size, gpus_per_node)]
        assert ranks_by_node == [[0, 1], [2, 3]]
    assert args[-7:] == ["train_profile.py", "--in-container", "--slurm",
                         "--num-gpus", str(world_size), "--work-dir", "runs/with spaces"]
    assert effective["MASTER_ADDR"] == "gpu-a"
    assert effective["MASTER_PORT"] == "32345"
    assert effective["POWER_GPUS_PER_NODE"] == str(gpus_per_node)
    assert effective["PROFILE_TASK_GPUS"] == "1"
    import uuid
    uuid.UUID(effective["PROFILE_MAPPING_RUN"])
    assert effective["OMP_NUM_THREADS"] == "6"
    for name in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE",
                 "NCCL_IB_HCA", "APPTAINERENV_NCCL_IB_HCA"):
        assert name not in effective
    for name, value in dict(NCCL_NET="IB", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="0",
                            NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET",
                            NCCL_COLLNET_ENABLE="0").items():
        assert effective[name] == effective[f"APPTAINERENV_{name}"] == value
    assert effective["APPTAINERENV_SMOKE_NCCL_IB_HCA"] == hca


def test_two_gpu_allocation_and_preflight():
    from carbon_aware_3d.gpu_preflight import validate_mappings
    wrapper = (ROOT / "batch_files/profile_multinode_2gpu.sbatch").read_text()
    directives = dict(line.removeprefix("#SBATCH --").split("=", 1)
                      for line in wrapper.splitlines() if line.startswith("#SBATCH --"))
    assert {key: directives[key] for key in ("nodes", "ntasks", "ntasks-per-node", "gres")} == {
        "nodes": "2", "ntasks": "2", "ntasks-per-node": "1", "gres": "gpu:1"}
    records = [dict(rank=r, local_rank=0, hostname=f"node{r}", gpu_uuid=f"GPU-{r}")
               for r in range(2)]
    validate_mappings(records, world_size=2, nodes=2, per_node=1)


@pytest.mark.parametrize("fails", [False, True])
def test_memlock_precedes_container(tmp_path, fails):
    task = BATCH.read_text().split("bash -c '", 1)[1].split("'", 1)[0]
    init = tmp_path / "init.sh"
    init.write_text('ulimit() { test "$*" = "-Sl unlimited" || exit 90; '
                    + ('return 1;' if fails else 'export MOCK_MEMLOCK=unlimited;') + ' }\n')
    result = subprocess.run(["bash", "-c", task, "memlock-task", "bash", "-c",
                             'printf "%s" "$MOCK_MEMLOCK"'],
                            env=dict(os.environ, BASH_ENV=str(init)),
                            text=True, capture_output=True)
    assert result.returncode == (1 if fails else 0)
    assert result.stdout == ("" if fails else "unlimited")


@pytest.mark.parametrize("world_size, order", [(2, "tp-dp-pp"), (4, "tp-dp-pp"), (4, "dp-tp-pp")])
def test_profile_menu(world_size, order):
    from train_profile import read_profile_menu_preview
    path = ROOT / f"configs/profile_menu_{world_size}gpu_{order}.json"
    cases = json.loads(path.read_text())["strategies"]
    assert cases == [dict(tp=2, pp=1, dp=world_size // 2, micro_batch_size=2, order=order)]
    assert read_profile_menu_preview(path) == (2, 1, 2, [2])
    assert world_size == cases[0]["tp"] * cases[0]["pp"] * cases[0]["dp"]

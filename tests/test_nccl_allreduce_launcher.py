"""No Slurm allocation or container is launched by these tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("extra", [[], ["--host-staged"]])
def test_probe_launch(tmp_path, extra):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_with_power.sh").write_text('exec "$@"\n')
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/bin/bash\nprintf "gpu-n29\ngpu-n30\n"\n')
    scontrol.chmod(0o755)
    srun = tmp_path / "srun"
    srun.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "print(json.dumps(dict(argv=sys.argv[1:], env=dict(os.environ))))\n")
    srun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
               SLURM_SUBMIT_DIR=str(tmp_path), SLURM_JOB_NODELIST="gpu-n[29-30]",
               SLURM_NTASKS="2", SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="4")
    result = subprocess.run(["bash", str(ROOT / "batch_files/nccl_allreduce_probe.sbatch"), *extra],
                            env=env, text=True, capture_output=True, check=True)
    captured = json.loads(result.stdout.splitlines()[-1])
    args = captured["argv"]
    for flag in ("--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--gpus-per-node=1",
                 "--gpus-per-task=1", "--gpu-bind=single:1", "--gres-flags=allow-task-sharing",
                 "--distribution=block", "--overlap", "--exact", "--kill-on-bad-exit=0"):
        assert flag in args
    assert "scripts/nccl_allreduce_probe.py" in args
    assert any(a.endswith("/launch-rank%t.err") for a in args)
    assert any(a.endswith("/launch-rank%t.out") for a in args)
    assert ("--host-staged" in args) == bool(extra)
    assert captured["env"]["POWER_GPUS_PER_NODE"] == "1"
    assert captured["env"]["NCCL_NET"] == "IB"
    assert captured["env"]["NCCL_IB_DISABLE"] == "0"
    assert captured["env"]["PROFILE_MAPPING_RUN"]
    assert len(list((tmp_path / "runs").iterdir())) == 1

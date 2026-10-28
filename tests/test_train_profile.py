import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_launcher(tmp_path, *extra):
    env = os.environ.copy()
    env.pop("CONTAINER_IMAGE", None)
    return subprocess.run(
        [sys.executable, str(ROOT / "train_profile.py"), "--model", "gpt2",
         "--profile-menu", str(ROOT / "configs/profile_menu.example.json"),
         "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "out.json"),
         "--dry-run", *extra], env=env, text=True, capture_output=True,
    )


def test_container_mode_reuses_runner_mount_without_nested_apptainer(tmp_path):
    result = run_launcher(tmp_path, "--in-container")
    assert result.returncode == 0, result.stderr
    command = result.stdout.split("Full cmd:", 1)[1].splitlines()[0]
    assert "apptainer" not in command
    assert "torchrun --nproc_per_node=4" in command
    assert "/opt/Megatron-Bridge/3rdparty/Megatron-LM/profile_strategies.py" in command
    assert "--micro-batch-size 2" in command


def test_host_mode_still_requires_image_and_uses_apptainer(tmp_path):
    assert run_launcher(tmp_path).returncode != 0
    image = tmp_path / "test.sif"
    image.touch()
    result = run_launcher(tmp_path, "--container-image", str(image),
                          "--megatron-dir", "/custom/megatron")
    assert result.returncode == 0, result.stderr
    assert "Full cmd: apptainer exec --nv" in result.stdout
    assert "/custom/megatron/profile_strategies.py" in result.stdout


def test_slurm_requires_container(tmp_path):
    result = run_launcher(tmp_path, "--slurm")
    assert result.returncode != 0
    assert "--slurm requires --in-container" in result.stderr


def test_slurm_runs_one_python_worker(tmp_path, monkeypatch):
    for key, value in dict(SLURM_PROCID="5", SLURM_LOCALID="1", SLURM_NTASKS="8",
                           MASTER_ADDR="node0", MASTER_PORT="29500").items():
        monkeypatch.setenv(key, value)
    result = run_launcher(tmp_path, "--in-container", "--slurm", "--num-gpus", "8")
    assert result.returncode == 0, result.stderr
    command = result.stdout.split("Full cmd:", 1)[1].splitlines()[0]
    assert "torchrun" not in command
    assert "apptainer" not in command
    assert "profile_strategies.py" in command


def test_slurm_rank_environment():
    from train_profile import configure_slurm_ranks
    env = dict(SLURM_PROCID="5", SLURM_LOCALID="1", SLURM_NTASKS="8",
               MASTER_ADDR="node0", MASTER_PORT="29500", RANK="0", LOCAL_RANK="0")
    assert configure_slurm_ranks(env, 8) == 5
    assert (env["RANK"], env["LOCAL_RANK"], env["WORLD_SIZE"]) == ("5", "1", "8")
    env["CUDA_VISIBLE_DEVICES"] = "GPU-task-bound-uuid"
    configure_slurm_ranks(env, 8)
    assert env["LOCAL_RANK"] == "0"
    assert env["SLURM_LOCALID"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-task-bound-uuid"



def test_slurm_world_size_mismatch(tmp_path, monkeypatch):
    for key, value in dict(SLURM_PROCID="0", SLURM_LOCALID="0", SLURM_NTASKS="8",
                           MASTER_ADDR="node0", MASTER_PORT="29500").items():
        monkeypatch.setenv(key, value)
    result = run_launcher(tmp_path, "--in-container", "--slurm", "--num-gpus", "4")
    assert result.returncode != 0
    assert "total world size" in result.stderr


import pytest


@pytest.mark.parametrize("status", [0, 7])
def test_slurm_configures_before_launch_and_verifies_only_success(tmp_path, monkeypatch, status):
    import train_profile
    from scripts import nccl_builtin_ib_smoke as ib
    from scripts import memlock_smoke
    events = []
    # The launcher mutates these variables; restore them between test modules.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "PROFILE_SLURM_LAUNCH"):
        monkeypatch.setenv(key, "")
    for key, value in dict(SLURM_PROCID="1", SLURM_LOCALID="1", SLURM_NTASKS="8",
                           MASTER_ADDR="node0", MASTER_PORT="29500").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["train_profile.py", "--model", "gpt2",
        "--profile-menu", str(ROOT / "configs/profile_menu.example.json"),
        "--work-dir", str(tmp_path / "work"), "--output", str(tmp_path / "out.json"),
        "--slurm", "--in-container", "--num-gpus", "8"])
    monkeypatch.setattr(memlock_smoke, "check_memlock", lambda: events.append("memlock"))
    monkeypatch.setattr(ib, "configure", lambda log: events.append(("configure", log)))
    from scripts import nccl_oob_preflight as oob
    def network_preflight(work):
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=validated0")
        monkeypatch.setenv("NCCL_SOCKET_FAMILY", "AF_INET")
        events.append("oob")
    monkeypatch.setattr(oob, "configure_profile_oob", network_preflight)
    monkeypatch.setattr(ib, "verify_log", lambda log: events.append(("verify", log)))
    def launch(command):
        assert command[0] == sys.executable
        assert "torchrun" not in command
        assert os.environ["RANK"] == "1"
        assert os.environ["WORLD_SIZE"] == "8"
        assert os.environ["NCCL_SOCKET_IFNAME"] == "=validated0"
        assert os.environ["NCCL_SOCKET_FAMILY"] == "AF_INET"
        events.append("launch")
        return status
    monkeypatch.setattr(train_profile.subprocess, "call", launch)
    with pytest.raises(SystemExit) as exc:
        train_profile.main()
    assert exc.value.code == status
    log = tmp_path / "work" / "nccl-rank1.log"
    assert events == ["memlock", ("configure", log), "oob", "launch"] + (
        [("verify", log)] if status == 0 else [])


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("visible", ["3", "GPU-task-uuid"])
def test_slurm_task_binding_is_not_local_cuda_ordinal(rank, visible):
    from train_profile import configure_slurm_ranks
    env = dict(SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank // 2), SLURM_NTASKS="8",
               MASTER_ADDR="node0", MASTER_PORT="29500", CUDA_VISIBLE_DEVICES=visible)
    configure_slurm_ranks(env, 8)
    assert env["RANK"] == str(rank)
    assert env["LOCAL_RANK"] == "0"
    assert env["SLURM_LOCALID"] == str(rank // 2)
    assert env["CUDA_VISIBLE_DEVICES"] == visible

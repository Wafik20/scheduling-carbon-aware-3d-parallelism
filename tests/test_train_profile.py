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

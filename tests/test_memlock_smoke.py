import importlib.util
from pathlib import Path
import resource
import subprocess
import os
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("memlock", ROOT / "scripts/memlock_smoke.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("rank", [0, 1])
def test_container_limits_logged_on_each_rank(monkeypatch, capsys, rank):
    monkeypatch.setenv("SLURM_PROCID", str(rank))
    monkeypatch.setattr(resource, "getrlimit", lambda key: (-1, -1))
    module.check_memlock()
    output = capsys.readouterr().out
    assert f"rank={rank}" in output
    assert "soft=unlimited hard=unlimited" in output
    monkeypatch.setattr(resource, "getrlimit", lambda key: (8388608, -1))
    with pytest.raises(RuntimeError, match="did not inherit"):
        module.check_memlock()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("fails", [False, True])
def test_task_sets_limit_before_container(tmp_path, rank, fails):
    # Exercise the launcher's actual shell command with a mock shell builtin.
    launcher = (ROOT / 'scripts/slurm_multinode_power_smoke.sh').read_text()
    task = launcher.split("training_command=(bash -c '", 1)[1].split("'", 1)[0]
    init = tmp_path / 'init.sh'
    init.write_text('ulimit() { test "$*" = "-Sl unlimited" || exit 90; '
                    + ('return 1;' if fails else 'export MOCK_MEMLOCK=unlimited;') + ' }\n')
    env = dict(os.environ, BASH_ENV=str(init), SLURM_PROCID=str(rank))
    result = subprocess.run(['bash', '-c', task, 'memlock-task', 'bash', '-c',
                             'printf "%s:%s" "$SLURM_PROCID" "$MOCK_MEMLOCK"'],
                            env=env, text=True, capture_output=True)
    if fails:
        assert result.returncode == 1
        assert not result.stdout
    else:
        assert result.returncode == 0
        assert result.stdout == f'{rank}:unlimited'

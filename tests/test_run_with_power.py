"""Exercise the wrapper without Slurm or GPUs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "run_with_power.sh"


@pytest.fixture
def mock_cluster(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scripts = {
        "scontrol": "import os; print(os.environ['MOCK_NODES'])",
        "srun": r"""
import json, os, pathlib, signal, sys, time
args = sys.argv[1:]
root = pathlib.Path(os.environ['MOCK_ROOT'])
(root / 'srun.json').write_text(json.dumps({'args': args, 'env': dict(os.environ)}))
ready = pathlib.Path(args[args.index('--ready-dir') + 1])
stop = pathlib.Path(args[args.index('--stop-file') + 1])
(root / 'sampler.pid').write_text(str(os.getpid()))
mode = os.environ.get('MOCK_MODE', '')
if mode == 'fail':
    sys.exit(4)
if mode != 'timeout':
    for node in os.environ['MOCK_NODES'].split():
        time.sleep(.08)
        (ready / node).touch()
if mode == 'die':
    time.sleep(.2)
    sys.exit(5)
if mode == 'hang':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(.02)
while not stop.exists():
    time.sleep(.02)
(root / 'stopped').touch()
""",
    }
    for name, source in scripts.items():
        path = bindir / name
        path.write_text(f"#!{sys.executable}\n" + source)
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
               POWER_PYTHON=sys.executable, POWER_OUTPUT_DIR=str(tmp_path / 'logs'),
               POWER_STARTUP_TIMEOUT='2', POWER_STOP_TIMEOUT='.3',
               SLURM_JOB_NODELIST='node[1-2]', SLURM_JOB_ID='123',
               MOCK_ROOT=str(tmp_path), MOCK_NODES='node1\nnode2',
               CUDA_VISIBLE_DEVICES='0', SLURM_STEP_GPUS='0',
               SLURM_JOB_GPUS='0,1,2,3', SRUN_GPUS_PER_TASK='1')
    return tmp_path, env


def run(env, *command):
    return subprocess.run([str(WRAPPER), *command], env=env, text=True,
                          capture_output=True, timeout=10, cwd=env['MOCK_ROOT'])


@pytest.mark.parametrize('nodes', ['node1', 'node1\nnode2'])
def test_ready_argv_cwd_exit_and_allocation(mock_cluster, nodes):
    root, env = mock_cluster
    env['MOCK_NODES'] = nodes
    source = """
import json, os, pathlib, sys
root = pathlib.Path(os.environ['MOCK_ROOT'])
ready = next((root / 'logs' / 'job_123').glob('run_*/ready'))
assert all((ready / node).exists() for node in os.environ['MOCK_NODES'].split())
(root / 'training.json').write_text(json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd(), 'cuda': os.environ['CUDA_VISIBLE_DEVICES']}))
sys.exit(7)
"""
    result = run(env, sys.executable, '-c', source, 'a b', '', '"literal"', '$HOME')
    assert result.returncode == 7, result.stderr
    data = json.loads((root / 'training.json').read_text())
    assert data == {'argv': ['a b', '', '"literal"', '$HOME'], 'cwd': str(root), 'cuda': '0'}
    launch = json.loads((root / 'srun.json').read_text())
    args = launch['args']
    for flag in ['--nodes', '--ntasks']:
        assert args[args.index(flag) + 1] == str(len(nodes.split()))
    assert args[args.index('--ntasks-per-node') + 1] == '1'
    assert '--overlap' not in args
    assert '--exact' in args and '--export=ALL' in args
    assert not any('--exclusive' in arg or '--gres=none' in arg for arg in args)
    assert 'CUDA_VISIBLE_DEVICES' not in launch['env']
    assert 'SLURM_STEP_GPUS' not in launch['env']
    assert 'SRUN_GPUS_PER_TASK' not in launch['env']
    assert 'SLURM_JOB_GPUS' not in launch['env']
    assert (root / 'stopped').exists()
    assert len(list((root / 'logs' / 'job_123').glob('run_*'))) == 1


@pytest.mark.parametrize('mode, message', [('fail', 'before readiness'), ('timeout', 'timed out')])
def test_startup_failure_does_not_run_command(mock_cluster, mode, message):
    root, env = mock_cluster
    env.update(MOCK_MODE=mode, POWER_STARTUP_TIMEOUT='.3')
    result = run(env, sys.executable, '-c', "open('training', 'w').close()")
    assert result.returncode == 1
    assert message in result.stderr
    assert not (root / 'training').exists()


def test_runtime_death_preserves_training_status(mock_cluster):
    root, env = mock_cluster
    env['MOCK_MODE'] = 'die'
    result = run(env, sys.executable, '-c', 'import time; time.sleep(.5)')
    assert result.returncode == 0, result.stderr
    assert 'WARNING' in result.stderr


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_cleans_training_and_sampler(mock_cluster, signum):
    root, env = mock_cluster
    source = "import os, pathlib, time; pathlib.Path('training.pid').write_text(str(os.getpid())); time.sleep(30)"
    process = subprocess.Popen([str(WRAPPER), sys.executable, '-c', source], env=env,
                               cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / 'training.pid').exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        process.send_signal(signum)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 128 + signum, stderr
        for name in ['training.pid', 'sampler.pid']:
            with pytest.raises(ProcessLookupError):
                os.kill(int((root / name).read_text()), 0)
        assert (root / 'stopped').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_uncooperative_sampler_is_killed(mock_cluster):
    root, env = mock_cluster
    env['MOCK_MODE'] = 'hang'
    result = run(env, sys.executable, '-c', 'pass')
    assert result.returncode == 0, result.stderr
    with pytest.raises(ProcessLookupError):
        os.kill(int((root / 'sampler.pid').read_text()), 0)


def test_command_not_found_cleans_sampler(mock_cluster):
    root, env = mock_cluster
    result = run(env, '/no/such/training-command')
    assert result.returncode == 1
    assert (root / 'stopped').exists()


def test_interrupt_during_startup(mock_cluster):
    root, env = mock_cluster
    env['MOCK_MODE'] = 'timeout'
    process = subprocess.Popen([str(WRAPPER), sys.executable, '-c', "open('training', 'w').close()"],
                               env=env, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not (root / 'sampler.pid').exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode == 130, stderr
        assert not (root / 'training').exists()
        assert (root / 'stopped').exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("gpu_request", ["1", "2", "4", "a100:4"])
def test_explicit_per_node_gpu_gpu_request(mock_cluster, gpu_request):
    root, env = mock_cluster
    env['POWER_GPUS_PER_NODE'] = gpu_request
    inherited = dict(SLURM_GPUS_PER_TASK='1', SLURM_GPUS_PER_NODE='1',
                     SLURM_TRES_PER_TASK='cpu:4,gres/gpu:1',
                     SLURM_GPU_BIND='single:1', SLURM_TRES_BIND='gres/gpu:single:1',
                     SRUN_OVERLAP='1')
    env.update(inherited)
    source = "import os; " + "; ".join(
        f"assert os.environ[{key!r}] == {value!r}" for key, value in inherited.items())
    result = run(env, sys.executable, '-c', source)
    assert result.returncode == 0, result.stderr
    launch = json.loads((root / 'srun.json').read_text())
    args = launch['args']
    assert '--overlap' not in args
    assert not (inherited.keys() & launch['env'].keys())
    assert f'--gpus-per-node={gpu_request}' in args
    assert args[args.index('--expected-gpus') + 1] == gpu_request.split(':')[-1]
    assert args[args.index('--ntasks-per-node') + 1] == '1'
    assert '--gpu-bind=none' in args

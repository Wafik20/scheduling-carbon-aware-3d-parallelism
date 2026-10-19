"""Run the actual sidecar through a fake two-node Slurm launch and fake NVML."""
import csv
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("short_node", [False, True])
def test_actual_sidecars_on_two_mock_nodes(tmp_path, short_node):
    repo = Path(__file__).resolve().parents[1]
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "pynvml.py").write_text("""
class NVMLError(Exception): pass
class NVMLError_NotSupported(NVMLError): pass
class NVMLError_NoPermission(NVMLError): pass
def nvmlInit(): pass
def nvmlShutdown(): pass
def nvmlDeviceGetCount(): return 4
def nvmlDeviceGetHandleByIndex(i): return i
def nvmlDeviceGetHandleByUUID(uuid): return int(uuid.split('-')[1])
def nvmlDeviceGetMinorNumber(h): return h
def nvmlDeviceGetUUID(h): return 'GPU-' + str(h)
def nvmlDeviceGetMigMode(h): return (0, 0)
def nvmlDeviceGetPowerUsage(h): return (h + 1) * 10000
""")
    (fake / "sitecustomize.py").write_text("""
import os
from carbon_aware_3d import compute_power as power
power._slurm_device_isolation = lambda: True
power._cuda_visible_uuids = lambda: (
    ['GPU-0', 'GPU-2'] if os.environ.get('SLURMD_NODENAME') == 'node1'
    else (['GPU-1'] if os.environ.get('SHORT_NODE') == '1' else ['GPU-1', 'GPU-3']))
""")
    commands = {
        "scontrol": "print('node1'); print('node2')",
        "srun": """
import os, subprocess, sys
args = sys.argv[1:]
start = next(i for i, a in enumerate(args) if a.endswith('compute_power_multinode.py')) - 1
children = []
for rank, node in enumerate(['node1', 'node2']):
    env = dict(os.environ, SLURMD_NODENAME=node, SLURM_PROCID=str(rank),
               SLURM_LOCALID='0', SLURM_NODEID=str(rank), SLURM_JOB_GPUS='',
               SLURM_STEP_GPUS='7,9', CUDA_VISIBLE_DEVICES='0,1')
    if rank == 1 and os.environ.get('SHORT_NODE') == '1':
        env.update(SLURM_STEP_GPUS='7', CUDA_VISIBLE_DEVICES='0')
    children.append(subprocess.Popen(args[start:], env=env))
codes = [p.wait() for p in children]
sys.exit(max(codes))
""",
    }
    for name, source in commands.items():
        file = fake / name
        file.write_text(f"#!{sys.executable}\n" + source)
        file.chmod(0o755)
    output = tmp_path / "logs"
    env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
               PYTHONPATH=str(fake) + os.pathsep + str(repo / 'src'),
               POWER_PYTHON=sys.executable, POWER_OUTPUT_DIR=str(output),
               POWER_GPUS_PER_NODE='2', SHORT_NODE=str(int(short_node)),
               POWER_INTERVAL='.02', POWER_STARTUP_TIMEOUT='5', POWER_STOP_TIMEOUT='1',
               SLURM_JOB_ID='123', SLURM_JOB_NODELIST='node[1-2]')
    training = """
import csv, os, pathlib, time
pathlib.Path(os.environ['POWER_OUTPUT_DIR'], 'training-started').touch()
run = next(pathlib.Path(os.environ['POWER_OUTPUT_DIR']).glob('job_123/run_*'))
for node in ['node1', 'node2']:
    with (run / (node + '.csv')).open() as f:
        assert len(list(csv.DictReader(f))) >= 1
time.sleep(.1)
"""
    result = subprocess.run(['bash', str(repo / 'scripts/run_with_power.sh'),
                             sys.executable, '-c', training], env=env,
                            capture_output=True, text=True, timeout=10)
    if short_node:
        assert result.returncode != 0
        assert 'expected 2 allocated GPUs' in result.stderr
        assert not (output / 'training-started').exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    runs = list(output.glob('job_123/run_*'))
    assert len(runs) == 1
    assert sorted(p.name for p in runs[0].glob('*.csv')) == ['node1.csv', 'node2.csv']
    for node, gpu, watts in [('node1', 'GPU-0;GPU-2', 40), ('node2', 'GPU-1;GPU-3', 60)]:
        with (runs[0] / (node + '.csv')).open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 2
        assert all(row['gpu_uuids'] == gpu for row in rows)
        assert all(float(row['total_power_W']) == watts for row in rows)
        assert all(row['slurm_job_id'] == '123' for row in rows)
        assert all(row['timestamp'].endswith('+00:00') for row in rows)

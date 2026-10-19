"""Slurm sidecar lifecycle for run_with_power.sh (stdlib only)."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def stop_group(process, timeout):
    """Terminate the entire session, including children of an exited leader."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main(argv):
    if not argv:
        print("Usage: run_with_power.sh COMMAND [ARG ...]", file=sys.stderr)
        return 2
    repo = Path(__file__).resolve().parents[1]
    sampler = training = None
    stop_file = None
    stop_timeout = 10.0
    status = 1

    def interrupted(signum, _frame):
        raise Interrupted(signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        interval = float(os.environ.get("POWER_INTERVAL", "0.5"))
        startup_timeout = float(os.environ.get("POWER_STARTUP_TIMEOUT", "60"))
        stop_timeout = float(os.environ.get("POWER_STOP_TIMEOUT", "10"))
        if not all(0 < value < float("inf") for value in (interval, startup_timeout, stop_timeout)):
            raise ValueError("power intervals and timeouts must be finite and positive")
        nodelist = os.environ.get("SLURM_JOB_NODELIST")
        if not nodelist:
            raise ValueError("SLURM_JOB_NODELIST is required; run inside a Slurm allocation")
        nodes = subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).split()
        if not nodes or len(set(nodes)) != len(nodes):
            raise ValueError("scontrol returned an empty or duplicate node list")
        output = Path(os.environ.get("POWER_OUTPUT_DIR", str(repo / "power_logs"))).resolve()
        job = os.environ.get("SLURM_JOB_ID", "unknown")
        if not job.replace("_", "").replace("-", "").isalnum():
            raise ValueError("invalid SLURM_JOB_ID")
        group = output / ("job_" + job)
        group.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run_", dir=group))
        ready = run / "ready"
        ready.mkdir()
        stop_file = run / "stop"
        print(f"Power logs: {run}", flush=True)
        env = os.environ.copy()
        # A parent training step may expose only a subset of the job GPUs.
        # Let srun establish fresh node-local visibility for the full allocation.
        for key in list(env):
            if key in {"LOCAL_RANK", "RANK", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
                       "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK", "SLURM_GPUS_PER_SOCKET",
                       "SLURM_TRES_PER_TASK", "SLURM_TRES_PER_NODE", "SLURM_TRES_PER_SOCKET",
                       "SLURM_CPUS_PER_GPU", "SLURM_MEM_PER_GPU", "SLURM_GPU_BIND", "SLURM_TRES_BIND", "SLURM_TRES_PER_JOB",
                       "SLURM_NTASKS_PER_NODE", "SLURM_GRES", "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST",
                       "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_VERBOSE"} or key.startswith("SRUN_"):
                env.pop(key)
        # A uniform per-node request avoids Slurm choosing a smaller step GRES
        # set when the monitoring step has fewer tasks/CPUs than training.
        gpu_request = os.environ.get("POWER_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_PER_NODE")
        gpu_options = []
        sampler_options = []
        if gpu_request:
            # Preserve optional GRES type, e.g. a100:2.
            parts = gpu_request.split(":")
            if len(parts) > 2 or not parts[-1].isdigit() or int(parts[-1]) <= 0:
                raise ValueError("POWER_GPUS_PER_NODE must be COUNT or TYPE:COUNT")
            gpu_options = [f"--gpus-per-node={gpu_request}"]
            sampler_options = ["--expected-gpus", parts[-1]]
        else:
            print("WARNING: no uniform GPU count specified; relying on Slurm step GRES inheritance", file=sys.stderr)
        count = str(len(nodes))
        command = ["srun", "--overlap", "--exact", "--export=ALL", "--kill-on-bad-exit=1", "--nodes", count, "--ntasks", count,
                   "--ntasks-per-node", "1", "--cpus-per-task", "1", "--cpu-bind=none", "--gpu-bind=none",
                   *gpu_options, sys.executable, str(repo / "compute_power_multinode.py"),
                   "--log-dir", str(run), "--interval", str(interval),
                   "--ready-dir", str(ready), "--stop-file", str(stop_file), *sampler_options]
        sampler = subprocess.Popen(command, env=env, start_new_session=True)
        deadline = time.monotonic() + startup_timeout
        while True:
            if sampler.poll() is not None:
                raise RuntimeError(f"power sampler exited before readiness (status {sampler.returncode})")
            if all((ready / node).is_file() for node in nodes):
                break
            if time.monotonic() >= deadline:
                missing = [node for node in nodes if not (ready / node).is_file()]
                raise RuntimeError(f"power sampler startup timed out; missing nodes: {', '.join(missing)}")
            time.sleep(0.05)
        training = subprocess.Popen(argv, start_new_session=True)
        warned = False
        while training.poll() is None:
            if sampler.poll() is not None and not warned:
                print("WARNING: power sampler exited during training; subsequent power data is missing", file=sys.stderr)
                warned = True
            time.sleep(0.05)
        status = training.returncode
        if status < 0:
            status = 128 - status
    except Interrupted as exc:
        status = 128 + exc.signum
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Power wrapper: {exc}", file=sys.stderr)
        status = 1
    finally:
        # Further signals must not interrupt bounded process-group cleanup.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        if stop_file is not None:
            try:
                stop_file.touch()
            except OSError as exc:
                print(f"WARNING: cannot write power stop file: {exc}", file=sys.stderr)
        stop_group(training, stop_timeout)
        if sampler is not None:
            try:
                sampler.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                pass
            stop_group(sampler, stop_timeout)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

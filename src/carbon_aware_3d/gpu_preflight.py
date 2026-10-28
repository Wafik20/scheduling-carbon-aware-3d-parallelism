"""Filesystem-only GPU allocation preflight; no distributed process groups."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_available(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return {"unavailable": str(exc)}


def slurm_snapshot(job, step):
    result = {}
    for name, command in (
        ("job", ["scontrol", "show", "job", "-dd", job]),
        ("step", ["scontrol", "show", "step", "-dd", f"{job}.{step}"]),
        ("config", ["scontrol", "show", "config"]),
    ):
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=5)
            result[name] = dict(command=command, returncode=proc.returncode,
                                stdout=proc.stdout, stderr=proc.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[name] = {"command": command, "unavailable": str(exc)}
    return result


def validate_mappings(records, world_size, nodes, per_node):
    if {r["rank"] for r in records} != set(range(world_size)) or len(records) != world_size:
        raise ValueError("GPU preflight: incomplete or duplicate global ranks")
    groups = {}
    for record in records:
        if record.get("error"):
            raise ValueError(f"GPU preflight: rank {record['rank']} failed: {record['error']}")
        groups.setdefault(record["hostname"], []).append(record)
    if len(groups) != nodes or world_size != nodes * per_node:
        raise ValueError(f"GPU preflight: expected {nodes} nodes x {per_node} GPUs; got {list(groups)}")
    for host, group in groups.items():
        owners = {}
        for record in group:
            identity = record["gpu_uuid"]
            if not identity or not identity.startswith("GPU-"):
                raise ValueError(f"GPU preflight: invalid physical UUID on rank {record['rank']}")
            owners.setdefault(identity, []).append(record["rank"])
        duplicates = {uuid: ranks for uuid, ranks in owners.items() if len(ranks) > 1}
        if duplicates:
            raise ValueError(f"GPU preflight: duplicate GPU assignments on {host}: {duplicates}")
        if len(group) != per_node or len(owners) != per_node:
            raise ValueError(f"GPU preflight: {host} expected {per_node} distinct GPUs; got {len(owners)}")
        if {r["local_rank"] for r in group} != set(range(per_node)):
            raise ValueError(f"GPU preflight: invalid local ranks on {host}")


def wait_for_mappings(directory, world_size, timeout=60, *, clock=time.monotonic, sleep=time.sleep):
    deadline = clock() + timeout
    while True:
        missing, records = [], []
        for rank in range(world_size):
            path = directory / f"rank{rank}.json"
            if not path.exists():
                missing.append(rank)
            else:
                record = json.loads(path.read_text())
                if record.get("error"):
                    raise ValueError(f"GPU preflight: rank {rank} failed: {record['error']}")
                records.append(record)
        if not missing:
            return records
        if clock() >= deadline:
            raise TimeoutError(f"GPU preflight timed out; missing ranks {missing}; artifacts: {directory}")
        sleep(min(0.1, max(0, deadline - clock())))


def run_preflight(args, work_dir, select_device):
    """Run in the profiler process before runtime services or Megatron init."""
    env = os.environ
    job, step = env["SLURM_JOB_ID"], env["SLURM_STEP_ID"]
    token = env["PROFILE_MAPPING_RUN"]
    if not all(c.isalnum() or c in "-_" for c in token + job + step):
        raise ValueError("Invalid GPU preflight run identifier")
    directory = Path(work_dir) / "gpu_preflight" / f"{job}.{step}-{token}"
    directory.mkdir(parents=True, exist_ok=True)
    rank = int(env["RANK"])
    record = dict(hostname=socket.gethostname(), rank=rank,
                  local_rank=int(env["SLURM_LOCALID"]),
                  environment={key: env.get(key) for key in (
                      "SLURM_JOB_ID", "SLURM_STEP_ID", "CUDA_VISIBLE_DEVICES",
                      "SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")},
                  cgroup=read_available("/proc/self/cgroup"),
                  mountinfo=read_available("/proc/self/mountinfo"))
    membership = record["cgroup"]
    if isinstance(membership, str):
        record["devices_list"] = {}
        for line in membership.splitlines():
            _, controllers, path = line.split(":", 2)
            if "devices" in controllers.split(","):
                location = Path("/sys/fs/cgroup/devices") / path.lstrip("/") / "devices.list"
                record["devices_list"][str(location)] = read_available(location)
        record["device_policy_note"] = (
            "cgroup v2 device BPF rules are not exposed in devices.list; "
            "membership/mounts and Slurm configuration are diagnostic evidence, not a policy audit")
    # Snapshot before publishing: another rank may fail and terminate the step.
    if record["local_rank"] == 0:
        write_json(directory / f"slurm-rank{rank}.json", slurm_snapshot(job, step))
    try:
        from carbon_aware_3d.compute_power import _cuda_visible_uuids
        import pynvml
        device = select_device(args)
        uuids = _cuda_visible_uuids()
        identity = uuids[device]
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(identity)
            pci = pynvml.nvmlDeviceGetPciInfo(handle).busId
            record.update(cuda_device=device, visible_device_count=len(uuids), gpu_uuid=identity,
                          pci_bus_id=pci.decode() if isinstance(pci, bytes) else pci)
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    write_json(directory / f"rank{rank}.json", record)
    try:
        records = wait_for_mappings(directory, args.world_size)
        validate_mappings(records, args.world_size, int(env["SLURM_JOB_NUM_NODES"]),
                          int(env["POWER_GPUS_PER_NODE"]))
    except Exception as exc:
        write_json(directory / f"validation-rank{rank}.json", dict(ok=False, error=str(exc)))
        raise
    write_json(directory / f"validation-rank{rank}.json", dict(ok=True))
    print(f"[GPU preflight] rank={rank} validated {len(records)} mappings: {directory}", flush=True)

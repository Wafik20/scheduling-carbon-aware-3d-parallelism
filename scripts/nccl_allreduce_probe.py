"""Two-node, one-GPU-per-node built-in IB all-reduce diagnostic (no model)."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

if __package__:
    from .nccl_builtin_ib_smoke import check_memlock, configure, verify_log
    from .smoke_train_multinode import select_cuda_device, slurm_ranks
    from .nccl_oob_preflight import configure_oob
else:
    from nccl_builtin_ib_smoke import check_memlock, configure, verify_log
    from smoke_train_multinode import select_cuda_device, slurm_ranks
    from nccl_oob_preflight import configure_oob


PAYLOAD_SIZES = (1, 1024, 1572864, 4194304)
ITERATIONS = 5
TIMEOUT = timedelta(seconds=120)


def configure_worker(host_staged, log):
    if host_staged:
        os.environ["NCCL_NET_GDR_LEVEL"] = "LOC"
    configure(log)


def validate_layout():
    rank, world_size = slurm_ranks(os.environ)
    if (world_size != 2 or int(os.environ["SLURM_JOB_NUM_NODES"]) != 2
            or int(os.environ["POWER_GPUS_PER_NODE"]) != 1
            or int(os.environ["SLURM_LOCALID"]) != 0):
        raise ValueError("Probe requires exactly two nodes, one task/GPU per node, local rank 0")
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK="0")
    return rank


def run_payloads(torch, dist, group, rank):
    for size in PAYLOAD_SIZES:
        payload = torch.empty(size, dtype=torch.bfloat16, device="cuda:0")
        for iteration in range(ITERATIONS):
            payload.fill_(rank + 1)
            print(f"rank={rank} all_reduce elements={size} iteration={iteration + 1}", flush=True)
            dist.all_reduce(payload, op=dist.ReduceOp.SUM, group=group)
            torch.cuda.synchronize()
            if not torch.all(payload == 3).item():
                raise RuntimeError(f"Incorrect SUM: rank={rank} elements={size} iteration={iteration + 1}")
            print(f"rank={rank} verified elements={size} iteration={iteration + 1}", flush=True)


def worker(args):
    rank = validate_layout()
    configure_oob(args.result_dir, rank)
    configure_worker(args.host_staged, args.result_dir / f"rank{rank}.nccl.log")
    check_memlock()
    import torch
    import torch.distributed as dist
    from carbon_aware_3d.gpu_preflight import run_preflight

    def select_device(_args):
        device, count = select_cuda_device(torch.cuda, 0)
        if device != 0 or count != 1:
            raise RuntimeError("Probe requires exactly one visible CUDA device, selected as cuda:0")
        return device

    run_preflight(SimpleNamespace(world_size=2), args.result_dir, select_device)
    env = os.environ
    mapping_path = (args.result_dir / "gpu_preflight" /
                    f"{env['SLURM_JOB_ID']}.{env['SLURM_STEP_ID']}-{env['PROFILE_MAPPING_RUN']}" /
                    f"rank{rank}.json")
    mapping = json.loads(mapping_path.read_text())
    metadata = dict(rank=rank, hostname=socket.gethostname(), python=sys.version,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    nccl=torch.cuda.nccl.version(), gpu=torch.cuda.get_device_name(0),
                    gpu_uuid=mapping["gpu_uuid"], host_staged=args.host_staged,
                    environment={k: v for k, v in sorted(env.items())
                                 if k.startswith(("NCCL_", "SLURM_")) or k in
                                 ("CUDA_VISIBLE_DEVICES", "MASTER_ADDR", "MASTER_PORT",
                                  "RANK", "WORLD_SIZE", "LOCAL_RANK")})
    (args.result_dir / f"rank{rank}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    print(f"rank={rank} init_process_group", flush=True)
    dist.init_process_group(backend="nccl", rank=rank, world_size=2,
                            device_id=torch.device("cuda:0"), timeout=TIMEOUT)
    print(f"rank={rank} new_group ranks=[0, 1]", flush=True)
    group = dist.new_group(ranks=[0, 1], backend="nccl", timeout=TIMEOUT)
    run_payloads(torch, dist, group, rank)
    # On failure, let the process exit without risking a blocking cleanup.
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    print(f"rank={rank} ALL-REDUCE PROBE PASSED", flush=True)


def verify_host_staging(log):
    # Ignore capability/setting messages: only transport channel choices matter.
    for line in log.read_text().splitlines():
        if re.search(r"Channel\s+\d+", line) and "GDRDMA" in line:
            raise RuntimeError(f"Host-staged probe selected a GDRDMA channel: {line}")


def supervise(args):
    rank = int(os.environ["SLURM_PROCID"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    stdout = args.result_dir / f"rank{rank}.stdout.log"
    stderr = args.result_dir / f"rank{rank}.stderr.log"
    status_path = args.result_dir / f"rank{rank}.status.json"
    nccl_log = args.result_dir / f"rank{rank}.nccl.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--result-dir", str(args.result_dir.resolve())]
    if args.host_staged:
        command.append("--host-staged")
    status = dict(rank=rank, command=command, host_staged=args.host_staged,
                  stdout=str(stdout), stderr=str(stderr), nccl_log=str(nccl_log), verified=False)
    # Exclusive creation prevents accidental truncation if a result directory is reused.
    # NCCL opens its debug path with "w"; it must not reopen the stderr file.
    with stdout.open("x") as out, stderr.open("x") as err, nccl_log.open("x"):
        try:
            result = subprocess.run(command, stdout=out, stderr=err)
            status["exit_code"] = result.returncode
        except OSError as exc:
            status["exit_code"] = 127
            status["error"] = str(exc)
            print(str(exc), file=err, flush=True)
    # Persist the raw child status (including negative signal codes) before verification.
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    code = status["exit_code"]
    channels = [line for line in nccl_log.read_text().splitlines()
                if re.search(r"Channel\s+\d+", line) and "via NET/IB" in line]
    status["socket_selection_lines"] = [
        line for line in nccl_log.read_text().splitlines()
        if "OOB " in line or "Bootstrap" in line or "NCCL_SOCKET_IFNAME" in line
    ]
    status["ib_channel_lines"] = channels
    status["observed_gdrdma"] = any("GDRDMA" in line for line in channels)
    print(f"rank={rank} GDRDMA channels observed: {status['observed_gdrdma']} "
          f"(IB channel lines: {len(channels)})", flush=True)
    if code == 0:
        try:
            verify_log(nccl_log)
            if args.host_staged:
                verify_host_staging(nccl_log)
            status["verified"] = True
        except Exception as exc:
            status["verification_error"] = str(exc)
            print(str(exc), file=sys.stderr, flush=True)
            code = 1
    status["supervisor_exit_code"] = code if code >= 0 else 128 - code
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status["supervisor_exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--host-staged", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args)
        return 0
    return supervise(args)


if __name__ == "__main__":
    sys.exit(main())

# Standalone cross-node IB all-reduce probe

## Evidence so far

| Job | Hosts | Result |
| --- | --- | --- |
| 4579816 | gpu-n29 / gpu-n30 | Unique GPUs, preflight passed; first TP all-reduce failed: status 11, vendor 137, mlx5_0. Rank 1 NCCL file empty. |
| 4578989 | gpu-n28 / gpu-n31 | Unique GPUs; cross-node TP failed with the same status/vendor signature (rank 2), then rank 0 retry exhaustion. |
| 4578577 | gpu-n28 / gpu-n32 | Cross-node TP completed with four unique GPUs. |

All three loaded NCCL 2.30.7+cuda13.3 and used GDRDMA. The successful run also
reported port-speed query errno 93 and enumerated IB/RoCE adapters. Neither those
warnings nor the version alone explains the failures. The failed send completion
localizes an error to the communication path; it does not identify a hardware,
firmware, driver, GPUDirect, or NCCL root cause. No GPU experiment was run while
implementing this probe. See `job_4579816_ib_failure.md` for precise evidence.

## Job 4580014: NET initialization failure, not an all-reduce failure

Both rank preflights passed, then rank 0 failed in eager communicator initialization
with `Failed to initialize any NET plugin`. Rank 1's stderr contains secondary
TCPStore broken-pipe errors after rank 0 exited; no rank 1 final status was saved.
The job snapshot records **partition a100**, gpu-n34/gpu-n40, A100-PCIE GPUs, not
the a100_nvlink/SXM host pair used by 4578577. This is not a controlled same-host
comparison. GPU visibility does not establish RDMA device availability.

- `NCCL_NET=IB` forces IB; an unavailable IB backend must fail, not silently fall
  back to sockets. `NCCL_NET_PLUGIN=none` disables external plugins, not built-in
  IB. Successful 4578577 logs show those settings and `Initialized NET plugin IB`.
- Both runtime banners report **2.30.7+cuda13.3**, git `a5187fa2e`. The probe's
  `torch.cuda.nccl.version()` reports 2.30.5 and inherited `NCCL_VERSION` says
  2.30.4. Those are not substitutes for the runtime banner, nor evidence by
  themselves that this job loaded a different runtime than the successful job.
- Probe `device_id=cuda:0` forces the failure to surface during init. The mounted
  profiler's `megatron/training/initialize.py:403–406` also passes `device_id`.
  Removing eager initialization is not a justified network fix.
- Both probe logs are damaged at the beginning of NET initialization (rank 0's
  line ends `NCCL_NET_PLUGIN set by environment to Traceback`). NCCL and Python
  reopened/wrote the same underlying stderr file at independent offsets. A CPU
  reproduction confirms this overwrite hazard. The fix is a separate per-rank
  NCCL file, retaining existing failure status and Python stderr capture.

Source cross-check: public NVIDIA/NCCL commit
`3e46c089f4ae4732722a419eb6096aead7e47a30`, `src/debug.cc:265` opens the debug
filename with `fopen(..., "w")`; `src/plugin/net.cc:297–334` treats `none` as
no external plugins but still registers internal IB/Socket. Lines 357–389 fail
when no eligible backend initializes/assigns successfully. This is corroborating
upstream source, not the exact container build: public lookup of runtime commit
`a5187fa2e` was unavailable.

No transport flag, timeout, library, eager-init behavior, or profiler was changed.
CPU regression tests emulate NCCL opening its own file with `w`, then a Python
exception/signal; both streams survive separately. The missing low-level IB
initialization reason cannot be recovered from the overwritten logs. Runtime
network recovery is **not** claimed; the next controlled run must provide intact
NET initialization messages before any hardware/driver/plugin remedy is chosen.

## Job 4580461: OOB connection timeout

Both ranks initialized and assigned the IB NET plugin (`rank*.nccl.log:9–22`).
Rank 0 on gpu-n37 advertised `eth5:10.201.2.216`, while rank 1 on gpu-n48
advertised `ib0:10.203.2.227` (line 11 in each file). Rank 0 then timed out
connecting to 10.203.2.227 ports 49313 and 56007 (lines 26–29). Neither metadata
file contains `NCCL_SOCKET_IFNAME` or `NCCL_SOCKET_FAMILY`. The launcher and shared
built-in IB configure helper left these unset, permitting independent discovery.

This is a TCP connection-establishment failure, not a demonstrated GPU-RDMA
completion error. Different interface names/subnets alone do not prove a fault:
routing can connect them. The actual timeouts show the selected peer endpoints
were not reachable at that time; logs cannot distinguish routing, filtering, or
peer-listener issues. Do not infer that gpu-n48 has a working eth5.

The standalone probe now inventories local interface names, UP/loopback flags,
and primary IPv4 addresses before importing Torch (the ioctl inventory does not
enumerate secondary address aliases). It selects the local interface owning
an IPv4 address of the node hostname (or validates an existing single exact
`NCCL_SOCKET_IFNAME` override), exchanges listener endpoints through the unique
shared artifact directory, and verifies TCP connections in both directions.
Ambiguous/missing mappings and connection failures stop before NCCL. The
filesystem rendezvous and TCP handshake share a 60-second diagnostic deadline;
ready-file polling is 100 ms. `MASTER_ADDR` is unchanged. Only after
success does it set per-rank `NCCL_SOCKET_IFNAME==<actual-name>` (the value starts
with `=`) and `NCCL_SOCKET_FAMILY=AF_INET`. Names may differ between nodes.
No eth5, ib0, subnet, or node IP is hardcoded. Hostname-address ownership is a
candidate-selection assumption, not a claim of reachability; the TCP check is
required. This check cannot guarantee all future ephemeral ports remain reachable.

`NCCL_NET=IB`, `NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=0`, GPU preflight,
power monitoring, and all profiler settings are preserved. The new socket setting
chooses IP connection establishment; it does not switch GPU collectives to the
NCCL Socket data transport. Interface/endpoint checks are recorded in per-rank
OOB JSON diagnostics; actual NCCL OOB/Bootstrap/interface-selection lines remain
in `rankN.nccl.log` and are copied into final status JSON when the worker exits.

Upstream `src/misc/socket.cc::ncclFindInterfaces` honors an explicit
`NCCL_SOCKET_IFNAME`; otherwise it prefers `ib*`, then other interfaces.
This explains why independently valid local discovery can choose incompatible
networks. Source reviewed: https://github.com/NVIDIA/nccl/blob/master/src/misc/socket.cc
(not a verified source checkout of the container's runtime commit).

## One controlled baseline test

From the repository root:

```bash
sbatch -M gpu -p a100_nvlink --nodelist=gpu-n29,gpu-n30 \
  batch_files/nccl_allreduce_probe.sbatch
```

This requests the failed host pair, not specific physical GPUs. Compare recorded
UUIDs/PCI IDs before attributing a difference to software. No nodes are excluded.

The probe reuses `run_container.sh`, built-in IB `configure`/`verify_log`, memlock
checks, GPU UUID preflight, and `run_with_power.sh`. No Megatron model or optimizer
is loaded. A separate NCCL subgroup `[0,1]` exercises the TP-like all-reduce path.
BF16 buffers cover 1, 1024, **1,572,864** (the failed element count), and 4,194,304
elements, five reductions each. Each input is rank+1; every result must equal 3.
Default profiler, smoke-test, and Slurm scripts are untouched.

Diagnostic-only differences: 120-second collective timeout, ten-minute batch
limit, and `--kill-on-bad-exit=0` so one failure does not immediately kill the
peer before it can save logs. Scheduler cancellation/node failure can still
prevent a final status record; logging cannot guarantee output a process never
emits. There is no retry or automatic transport fallback.

## Artifacts, including failures

The batch log prints a unique `runs/nccl_allreduce_JOB_XXXXXX` directory.
Slurm writes `launch-rank*.out/err` before Python/container startup. Each rank's
supervisor redirects the worker's stdout/stderr directly to files. NCCL writes
to its own `rankN.nccl.log`, separate from `rankN.stderr.log`: NCCL opens its debug
path for writing, so using `/dev/stderr` can truncate/overwrite redirected Python
stderr through independent file offsets. The supervisor records the worker exit
status in `rankN.status.json`; worker output is `rankN.stdout.log` and Python
exceptions are in `rankN.stderr.log`. `observed_gdrdma` and `ib_channel_lines` are saved even on worker failure.
Success requires actual IB transport log verification, not just environment flags.

Host-side `job-rank*.txt`, `step-rank*.txt`, and `config-rank*.txt` capture scontrol
before Apptainer (the old in-container snapshots could not find scontrol).
Existing preflight records retain UUID/PCI/cgroup information. Progress is flushed
before each collective; metadata includes runtime versions and effective NCCL
settings. Keep the entire artifact directory plus the Slurm log and power logs.

## Controlled host-staged comparison (optional, separate experiment)

Repeat the same command with `--host-staged` after the script filename. This
changes only `NCCL_NET_GDR_LEVEL=LOC` in the probe worker, disabling GPU-direct
network access while keeping NCCL's built-in IB transport. Compare effective env,
GPU UUIDs and actual channel logs: baseline must show GDRDMA to be called a
GPU-direct test, and host-staged must not show GDRDMA channels.

This is an IB host-staging comparison for CUDA all-reduce buffers, **not** a CPU
tensor/Gloo test and not a raw verbs benchmark. It does not alter profiler settings.
For stronger isolation, administrators should run matching host-memory and
CUDA-enabled perftest RDMA tests on the recorded HCA/ports; tool support and fabric
permissions must be verified on compute nodes first. Login-node `ib_write_bw`
is installed, but its `--help` advertises no CUDA option; it is not evidence that
a GPU-memory perftest is available in the target container. Only help/capability
checks were run here, not RDMA traffic.

Interpretation:
- Baseline fails, host-staged passes on the same GPUs/HCA: suspect the GDR path
  (registration/DMA-BUF, topology, driver/firmware or NCCL), not proof of one cause.
- Both fail: investigate receiver kernel/mlx5 logs, HCA/switch counters and raw
  host-memory verbs; failure is not necessarily GDR-specific.
- Both pass: intermittent/topology-sensitive failure remains possible; the small
  subgroup test does not reproduce Megatron's full communicator lifecycle.

Request administrator decoding of vendor 137 for the installed firmware/driver.
Do not use it as a universal hardware diagnosis. Preserve both rank logs before
changing any HCA selection, NCCL algorithm, transport, or other setting.

# Profiling rank order

Rank order controls how global ranks are assigned to tensor (TP), data (DP),
and pipeline (PP) parallel groups. It does not change the requested parallel
degrees or physically move GPUs between nodes.

## Static order per run

A profile-menu entry can optionally specify `order`, for example:

```json
[
  {"tp": 2, "pp": 2, "micro_batch_size": 2, "order": "tp-dp-pp"}
]
```

The six supported orders are `tp-dp-pp`, `tp-pp-dp`, `dp-tp-pp`,
`dp-pp-tp`, `pp-tp-dp`, and `pp-dp-tp`. The leftmost axis varies fastest. Explicit orders are limited to plain
TP/PP/DP: CP/EP/GTP degrees must be one, expert TP must equal TP, and
virtual pipeline parallelism must be disabled.

Order is static for the whole profiling run. Every row must specify the
same explicit order, or every row must omit it. Mixed orders, including
mixed explicit and omitted orders, are rejected. This is not dynamic
order switching during training. Omitting `order` from every row preserves
the legacy default. Do not assume that a reordered list
of menu cases changes rank order.

For a controlled comparison, use separate runs and separate output files
for each order. Keep TP, PP, world size (therefore DP), microbatch size,
global batch size, model, sequence length, and measurement settings fixed.
Only axes with degree greater than one affect the mapping. If `k` of
TP, DP, and PP have degree greater than one, the six order strings yield
`k!` distinct mappings at fixed degrees. With two GPUs, at most one axis
has degree greater than one, so changing order alone gives no distinct
mapping.

## Check the actual mapping

Each calibration strategy records `order` and a `rank_mapping` list.
Each mapping entry contains `rank`, `hostname`, `local_rank`, `cuda_device`,
`cuda_visible_devices`, `gpu_uuid`, `gpu_name`, `tp_group`, `pp_group`,
and `dp_group`. The work directory also contains a per-case sidecar,
`rank_mapping_tp{tp}_pp{pp}_mbs{micro_batch_size}.json`, captured before
model construction.

Inspect these actual TP/DP/PP groups and host/device placements, not just
the requested order. Group membership identifies which communication crosses nodes;
an order string alone cannot establish that. Keep this mapping with the
calibration output when comparing time and power measurements.

## Smaller functionality test: 2 nodes × 2 GPUs

Use [the 4-GPU instructions](profile_4gpu.md) for separate `tp-dp-pp` and
`dp-tp-pp` runs with TP=2, PP=1, DP=2. The existing eight-GPU defaults below
are unchanged.

## First multi-node experiment: 2 nodes × 4 GPUs

Use `batch_files/profile_multinode.sbatch`, not the single-node batch file.
It uses one Slurm task per GPU, the existing Apptainer runner and power
wrapper, and the working built-in InfiniBand configuration with unlimited
soft memlock. `train_profile.py --slurm --in-container` launches one profiler
process directly per task, without nested torchrun agents. All ranks share
`MASTER_ADDR`/`MASTER_PORT`; Slurm supplies global rank and world size.
Training requests four GPUs per node and one per task with
`--gpus-per-task=1 --gpu-bind=single:1 --gres-flags=allow-task-sharing`.
Slurm gives each task one CUDA-visible GPU while allowing device-cgroup access
to allocated peers for NCCL. The profiler selects visible ordinal zero, even
when `CUDA_VISIBLE_DEVICES` contains a nonzero job-relative ID or GPU UUID.
`SLURM_LOCALID` is retained for rank-order recording, not used as an ordinal
inside the singleton mask. The batch sets `PROFILE_TASK_GPUS=1` so unexpected
multi-GPU visibility fails before initialization. NCCL is explicitly bound to
the selected device. Global rank mappings must still contain distinct UUIDs.
See [GPU troubleshooting](gpu_troubleshooting.md) and [preflight artifacts](gpu_preflight.md).

From the scheduler repository root, submit the first ordering manually:

```bash
sbatch batch_files/profile_multinode.sbatch \
  --model gpt2 \
  --profile-menu configs/profile_menu_8gpu_tp-dp-pp.json \
  --warmup-min-seconds 2 --warmup-max-seconds 5 --measure-seconds 5 \
  --output runs/profile_8gpu_tp-dp-pp_check/calibration.json \
  --work-dir runs/profile_8gpu_tp-dp-pp_check
```

The batch defaults request the GPU cluster, `a100_nvlink`, account
`cs2731_2026f`, two nodes, four GPUs and four tasks per node. Paths must be
on shared storage under this repository, mounted on every node by the
container runner. Use a fresh work/output directory for every job. The
short timing windows above are a functionality check, not research-quality
calibration; omit them for the existing 10–60-second warmup and 60-second
measurement defaults.

Six menus are provided as `configs/profile_menu_8gpu_<order>.json`, each
with TP=2, PP=2, DP=2 and microbatch 2. Submit each order separately, changing
both menu and output/work paths. No sweep or dynamic placement switching
is performed. All six produce distinct rank groups, though some can have
similar cross-node communication costs on a two-node layout.

The existing output schema is retained. `power_w` is the sum over WORLD
of one CUDA-UUID-selected GPU's mean power per rank, not just node zero.
The existing power wrapper also records allocation-wide per-node GPU CSVs
throughout the job. Measurements start behind a world barrier and stop
by a rank-zero decision broadcast to all ranks; throughput uses global
batch tokens, not an extra multiplication by world size.

Before trusting results, inspect the eight rank mappings for two hosts and
eight distinct GPU UUIDs, and inspect TP/PP/DP memberships. Per-rank
`nccl-rank*.log` files in the work directory are checked for built-in IB
and absence of the external Spectrum-X plugin after a successful run.
The existing single-node `profile.sbatch` / torchrun path is unchanged.

## Validated TCP/OOB network before profiling

The Slurm launcher reuses `scripts/nccl_oob_preflight.py` before spawning
Megatron. It discovers each rank's local primary IPv4 interfaces, selects the
one owning the node hostname address (or validates an exact override), and
checks bidirectional TCP reachability. Only after success are
`NCCL_SOCKET_IFNAME` (exact `=name`) and `NCCL_SOCKET_FAMILY=AF_INET` applied to
the environment inherited by Megatron. No interface name or node IP is hardcoded.
Job 4581054 demonstrated this on gpu-n37 (`eth5`, 10.201.2.216) and gpu-n52
(`eth0`, 10.201.2.231): different names, reachable management addresses. Both
probe ranks exited zero with IB verification; `host_staged` was false.

With the existing two-node block distribution, matching local ranks are paired:
2 GPUs: 0↔1; 4 GPUs: 0↔2, 1↔3; 8 GPUs: 0↔4, 1↔5, 2↔6, 3↔7.
Every rank validates its own selection. This assumes the existing uniform
per-node network namespace; it is not an all-to-all port/firewall test.
Diagnostics live at
`WORK_DIR/oob_preflight/JOB.STEP-PROFILE_MAPPING_RUN/rankN.oob*.json`.
The launcher's existing invocation UUID prevents stale endpoint reuse.
The existing 60-second preflight deadline is unchanged; failure prevents
that rank from launching Megatron. Dry runs and single-node torchrun skip it.

`NCCL_NET=IB`, GPU binding, UUID validation, power monitoring, parallel rank
order, menus, and measurement behavior are unchanged. TCP/OOB interface
selection does not replace the IB GPU-data transport.

To rerun the 2-node × 1-GPU functionality check on the successful node pair,
from the repository root (use fresh output/work paths for another rerun). The
explicit `a100` partition matches job 4581054; batch defaults stay unchanged:

```bash
sbatch -M gpu -p a100 --nodelist=gpu-n37,gpu-n52 \
  batch_files/profile_multinode_2gpu.sbatch \
  --model gpt2 \
  --profile-menu configs/profile_menu_2gpu_tp-dp-pp.json \
  --warmup-min-seconds 2 --warmup-max-seconds 5 --measure-seconds 5 \
  --output runs/profile_2gpu_oob_check/calibration.json \
  --work-dir runs/profile_2gpu_oob_check
```

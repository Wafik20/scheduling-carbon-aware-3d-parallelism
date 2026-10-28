# GPU allocation and NCCL troubleshooting

Keep the tested training flags together:
`--gpus-per-node=N --gpus-per-task=1 --gpu-bind=single:1
--gres-flags=allow-task-sharing`. Each task selects CUDA ordinal zero within
its singleton mask. Slurm local IDs are topology metadata, not CUDA ordinals.
`allow-task-sharing` permits allocated peer-device access for NCCL without
removing task-specific CUDA visibility. Do not disable UUID validation or set
`NCCL_MULTI_RANK_GPU_ENABLE=1`.

The sampler starts first **without** `--overlap`, one task per node, requesting
and validating exactly N GPUs. Training retains `--overlap` to share resources.
The wrapper must run from the batch/allocation shell, not a GPU-consuming task.
See [pre-NCCL artifacts](gpu_preflight.md) for live allocation capture and
missing/duplicate-rank diagnostics. Accounting GPU counts are not device maps.

## Evidence retained in historical artifacts

- **4573053:** sampler had three physical devices on gpu-n28 despite four
  accounted GPUs. Slurm 25.05's overlapping-step two-pass allocator can count
  the same GPU twice. Starting the sampler without overlap avoids that path.
- **4576499:** ranks selected valid CUDA zero but NCCL failed importing peer
  memory (P2P/SHM cuMem handles), not selecting a global-rank ordinal. Explicit
  NCCL `device_id` removes barrier guessing; peer access is a separate concern.
- **4577382:** removing task binding exposed only two devices, insufficient
  for local IDs 2/3. Never use modulo indexing to hide this failure.
- **4578536:** NCCL proved ranks 1/3 on gpu-n29 shared PCI bus ID `4c000`.
  Singleton masks alone cannot establish distinct physical GPUs. The preflight
  now checks UUIDs before NCCL can fail on duplicates.
- **4578577:** succeeded with gpu-n29 excluded, on gpu-n28/gpu-n32, four unique
  UUIDs, successful preflight, and built-in IB. This does not establish that
  gpu-n29 is inherently faulty or that all allocations are fixed.

Upstream Slurm's [GRES step allocator](https://github.com/SchedMD/slurm/blob/master/src/stepmgr/gres_stepmgr.c)
now excludes GPUs already selected in the first pass. Both per-node and
per-task finite requests can hit the older overlap defect; binding flags do
not fix that controller bug. Installed patch status requires administrator
confirmation. Capture live job/step GRES and core bitmaps if a new preflight
fails, and request an allocator update/backport rather than weakening checks.
The completed historical steps no longer have live controller records.

The successful short run's warmup did not converge, used mixed A100 40GB/80GB
nodes, and had one menu case (zero reshard transitions). Treat it as a functional
test, not research-quality switching-cost calibration. The power step was
terminated during wrapper cleanup; the job/training completed and CSVs remain.

A separate legacy-path issue remains unchanged: with omitted menu `order` and
`use_tp_pp_dp_mapping=True`, initial initialization honors that flag but later
transitions use the existing `tp-cp-ep-dp-pp` fallback. Use explicit static
orders for placement studies; correcting legacy transition semantics is not
part of this behavior-preserving cleanup.

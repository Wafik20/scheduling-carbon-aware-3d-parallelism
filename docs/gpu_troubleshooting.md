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

## Recurrence: job 4579126

The block rank layout worked: gpu-n28 had ranks 0/1 and gpu-n30 had ranks 2/3.
However, `runs/placement_block_check/gpu_preflight/4579126.1-386a58d0-4f72-4182-916f-102475069350/`
records both gpu-n30 ranks on UUID `GPU-af3445b2-2f34-ce06-4092-546f9f0efe49`,
PCI `00000000:48:00.0`, with `CUDA_VISIBLE_DEVICES=0`, `SLURM_STEP_GPUS=0`,
and `SLURM_GPUS_ON_NODE=1`. The sampler saw two allocated devices on that node
(minors 0/6, UUIDs `GPU-af3445b2-2f34-ce06-4092-546f9f0efe49` and
`GPU-34592d83-10b7-4bb7-13a0-46317568e844`). See the job log lines 6–9,
1093–1114. Preflight correctly rejected the duplicate before NCCL initialization.
Successful job 4578577 used singleton masks 0/1 and two distinct UUIDs per node.
Reduced records from both jobs are regression fixtures under
`tests/fixtures/gpu_preflight/`; they test rejection, not a live allocator fix.

Both jobs' saved `slurm-rank*.json` snapshots report `scontrol` unavailable in
the container. Thus the controller/stepd version, patch level, live allocation
bitmaps, and CPU/GPU affinity cannot be established from these snapshots.
The login-node `srun --version` reports 23.11.10; that is not proof of the GPU
cluster controller version. The earlier 4578536 diagnosis above is retained
historical documentation; its original job log is not present in this checkout.

### Smallest reliable fix belongs in Slurm

The observed short step allocation matches the upstream overlap allocator defect.
In Slurm tag `slurm-25-05-0-1`, `src/stepmgr/gres_stepmgr.c`:

- `_step_get_gres_needed` (2244–2247) prefers the finite per-node request over
  per-task. Dropping only `--gpus-per-task=1` does not change that path.
- `_set_step_gres_bit_alloc` (2344–2414) creates a fresh candidate bitmap on each
  pass. For overlap, it does not exclude devices selected on the previous pass.
- `gres_stepmgr_step_alloc` (2689–2699) runs a core-local pass then an unrestricted
  pass. Selecting the same device twice can satisfy the numeric request while
  the merged bitmap contains only one physical GPU.

Upstream master fixes this in `_set_step_gres_bit_alloc` by excluding this
step's existing bitmap from `gres_bit_avail` for non-shared GRES, independently
of `decr_job_alloc`:

```c
if (gres_ss->gres_bit_alloc && gres_ss->gres_bit_alloc[node_offset] &&
    !gres_id_shared(gres_state_job->config_flags)) {
    bit_and_not(gres_bit_avail, gres_ss->gres_bit_alloc[node_offset]);
}
```

Ask the cluster administrator to verify the deployed implementation and apply
the corresponding upstream fix/backport if missing. This preserves overlapping
power sampling, NCCL peer access, task binding, and IB. Do not claim a confirmed
installed Slurm bug until that version/bitmap evidence is available.

`allow-task-sharing` permits access to allocated peer devices; it does not request
duplicate rank ownership. `single:1` selects distinct slots when the step bitmap
contains enough GPUs, but cannot supply a missing device. `map_gpu` or
`gpu-bind=none` cannot repair the short allocation either. Removing overlap
would conflict with the sampler holding those GPUs. Omitting finite GPU requests
is not a reliable bypass: Slurm 25.05 `src/stepmgr/stepmgr.c`,
`_copy_job_tres_to_step`, copies the job's per-node/per-task TRES when the step
has no explicit GRES request. The job's finite GPU request can therefore return
before allocation. No speculative launcher workaround is applied.
The launcher and all validation remain unchanged by this investigation.

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

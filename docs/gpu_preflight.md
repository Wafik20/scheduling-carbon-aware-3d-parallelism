# Pre-NCCL GPU mapping diagnostics

The multi-node batch launcher assigns a fresh `PROFILE_MAPPING_RUN` token shared
by every task. In `profile_strategies.py`, the Slurm path runs the filesystem
preflight after parsing arguments but before runtime services or Megatron
distributed initialization. Non-Slurm profiling is unchanged. This assumes the training step uses the whole
allocation and every node sees the same shared work directory. Direct `--slurm`
launches outside the batch wrapper must export one common, fresh
`PROFILE_MAPPING_RUN` token and `POWER_GPUS_PER_NODE` before starting tasks;
do not generate the token independently in each rank.

Each rank selects its CUDA device, resolves its physical UUID using the existing
CUDA driver helper, and queries PCI bus ID through NVML by that UUID. It writes
an atomic JSON file, then waits up to **60 seconds** for all ranks. Validation
requires all global ranks, the requested node count, the expected number of
unique UUIDs per node (`POWER_GPUS_PER_NODE`), and distinct local IDs per node.
A duplicate reports hostname, UUID, and offending global ranks. Rank collection
errors are published as error records; a rank that never writes causes a bounded
missing-rank timeout. No NCCL or other distributed process group is used.

## Artifacts after failure

Under the profiling work directory:

```text
gpu_preflight/<job>.<step>-<invocation-token>/
  rank0.json ... rankN.json
  slurm-rank0.json ...       # one snapshot per node's local rank zero
  validation-rank0.json ...
```

- `rank*.json`: hostname, global/local ranks, selected CUDA ordinal, physical
  GPU UUID, PCI bus ID, Slurm job/step GPU variables, CUDA mask, `/proc/self/cgroup`
  and mount information. Available cgroup-v1 `devices.list` is included.
- `slurm-rank*.json`: live `scontrol show job -dd`, `show step -dd`, and
  `show config` outputs, exit status, or unavailable/timeout details. Each
  command is limited to **5 seconds** (at most 15 seconds per node, before
  mapping publication). Missing Slurm tools inside the container do not
  invalidate physical UUID checks; they are recorded as unavailable.
- `validation-rank*.json`: success or the duplicate/missing-rank error.
  The timeout bounds polling, not CUDA/NVML driver calls or blocked filesystem I/O.
  A rank killed by Slurm may not reach this final write, but published mapping
  and snapshot files remain. Missing mapping files identify ranks that never
  reached publication. Cgroup-v2 device BPF rules require administrator tooling;
  recorded membership is not an independent device-policy audit.

Example inspection, substituting the new run directory:

```bash
find runs/profile_4gpu_tp-dp-pp_check/gpu_preflight -type f -name '*.json' -print
python -m json.tool runs/profile_4gpu_tp-dp-pp_check/gpu_preflight/JOB.STEP-TOKEN/rank1.json
python -m json.tool runs/profile_4gpu_tp-dp-pp_check/gpu_preflight/JOB.STEP-TOKEN/validation-rank1.json
```

For the failure pattern in job 4578536, ranks 1 and 3 on the same hostname
would report the same physical UUID and PCI ID, and validation would fail before
NCCL. The existing later rank-order/UUID check remains in place. Calibration
output schema, sampler behavior, GPU binding, IB settings, and profiling loops
are unchanged. This diagnoses short/duplicate Slurm allocations; it does not
repair the controller allocator.

CPU tests: `.venv/bin/python -m pytest -q tests/test_gpu_preflight.py`.
Job 4578577 exercised this path successfully: four unique UUIDs and all four
validation records passed. Historical artifacts remain in runs/profile_4gpu_no_n29.

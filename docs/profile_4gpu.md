# Small placement test: two nodes, two GPUs per node

The separate `batch_files/profile_multinode_4gpu.sbatch` requests two nodes,
four tasks (two per node), and `gpu:2` per node. It runs the existing shared
launcher with Bash, so the latter's eight-task SBATCH defaults are not applied.
The original `profile_multinode.sbatch` retains its 2x4 submission defaults.
The shared launch body derives world size and GPUs per node from four/eight
Slurm tasks on exactly two nodes.

Both menus use TP=2, PP=1, DP=2, microbatch=2. Only order differs. Global batch
remains the launcher default of 16; timing, memory, CPU, and IB defaults are
unchanged. With PP=1, these two orders still produce distinct TP/DP groups:
`tp-dp-pp` makes TP ranks contiguous; `dp-tp-pp` makes DP ranks contiguous.
Actual inter-node traffic depends on Slurm placement; use recorded hostnames
and group memberships, not assumed block placement.

## Reproduce job 4578577 and compare the second ordering

Job 4578577 succeeded on gpu-n28/gpu-n32 with four distinct UUIDs and passing
preflight. The commands retain its `--exclude=gpu-n29` and short timings but
use fresh paths, preserving `runs/profile_4gpu_no_n29` and its checkpoints.
Run manually from the scheduler repository root:

```bash
sbatch -M gpu --partition=a100_nvlink --account=cs2731_2026f --exclude=gpu-n29 \
  batch_files/profile_multinode_4gpu.sbatch \
  --model gpt2 --profile-menu configs/profile_menu_4gpu_tp-dp-pp.json \
  --warmup-min-seconds 2 --warmup-max-seconds 5 --measure-seconds 5 \
  --output runs/profile_4gpu_tp-dp-pp_repro/calibration.json \
  --work-dir runs/profile_4gpu_tp-dp-pp_repro

sbatch -M gpu --partition=a100_nvlink --account=cs2731_2026f --exclude=gpu-n29 \
  batch_files/profile_multinode_4gpu.sbatch \
  --model gpt2 --profile-menu configs/profile_menu_4gpu_dp-tp-pp.json \
  --warmup-min-seconds 2 --warmup-max-seconds 5 --measure-seconds 5 \
  --output runs/profile_4gpu_dp-tp-pp_repro/calibration.json \
  --work-dir runs/profile_4gpu_dp-tp-pp_repro
```

Defaults use `a100_nvlink`; to use RTX 6000 add `--partition=rtx6k` before the
batch filename. Use fresh work/output paths on subsequent attempts. These
short timings test functionality, not calibration quality.

## Preserved checks

Training still requests `--gpus-per-task=1 --gpu-bind=single:1` and
`--gres-flags=allow-task-sharing`: singleton CUDA selection with peer-device
access for NCCL. The strict singleton guard, explicit NCCL device binding,
UUID-based power mapping, and distinct-rank UUID validation are unchanged.
The sampler receives `POWER_GPUS_PER_NODE=2`, requests two GPUs per node, and
requires exactly two resolved UUIDs before readiness. Apptainer, built-in IB,
memlock, and rank-order recording use the same path as the eight-GPU run.

Require two power CSVs, two GPU power columns/UUIDs in each, and four distinct
training GPU UUIDs across two hosts (two per host). Check the recorded TP/DP
groups and NCCL network IB verification. CPU tests cover both launch sizes,
menus, sampler expected counts, and four/eight-rank singleton UUID mappings.
The successful run is a functionality check: warmup did not converge and
the single-case menu measured no switching transitions.

The known Slurm overlapping-step allocation defect is not solved by shrinking
the experiment. If the controller supplies a short GPU bitmap, retain strict
failure and follow `gpu_troubleshooting.md`; do not reinterpret local ranks
modulo the visible device count.

## Lightweight tests

```bash
.venv/bin/python -m pytest -q tests/test_profile_multinode.py tests/test_train_profile.py tests/test_run_with_power.py tests/test_power_monitor.py
```

The mounted Megatron fork's tests cover four/eight-rank mapping and profiler
control logic. Run without GPU passthrough:

```bash
module load apptainer
apptainer exec \
  --bind /vast/cs2731_2026f/wat29/Megatron-LM:/opt/Megatron-Bridge/3rdparty/Megatron-LM \
  --pwd /opt/Megatron-Bridge/3rdparty/Megatron-LM \
  /vast/cs2731_2026f/wat29/containers/nemo_26.08.sif \
  python -m pytest -q --noconftest tests/test_profile_cuda_device.py tests/test_restored_profiler.py
```

# Profiler

The profiler measures how each feasible 3D parallelism configuration behaves
on a target model and hardware setup. These measurements are the input to the
carbon-aware scheduler.

## What Gets Measured

For each configuration `pi = (TP, PP, DP)`, the profiler measures or derives:

- steady-state power
- throughput
- energy per step
- energy per token
- switching or resharding cost between configurations

Throughput comes from Megatron Core / Megatron-LM training logs or metrics.
Power is sampled with NVML and aggregated across the GPUs assigned to the job.

The paper emphasizes that these values are model- and hardware-dependent. A
configuration that is lowest-power for one model or node topology may not be
lowest-carbon once throughput and switching costs are included.

## Main Files

- `train_profile.py`
  - Launches Megatron's `profile_strategies.py` inside the configured
    container.
  - Reads a profile menu such as `calibrations/profile_menu_4gpu.json`.
  - Writes a calibration JSON with per-strategy measurements.
- `src/carbon_aware_3d/compute_power.py`
  - Single-node NVML sampler used by training launchers.
- `compute_power_multinode.py`
  - Multi-node NVML sampler. One sampler per node writes `rank<N>.csv`.
- `scripts/benchmarks/benchmark_transform.py`
  - Launches the resharding benchmark inside the container.
- `scripts/benchmarks/benchmark_inner.py`
  - Runs inside the container and writes `results/resharding/reshard_times.csv`.
- `scripts/profile_per_layer.py`
  - Exploratory per-layer timing profiler for architecture-level analysis.

## Inputs

Typical profiler inputs include:

- model name
- number of GPUs
- profile menu of `(tp, pp)` pairs
- Megatron path inside the container
- Apptainer container image
- bind mounts
- warmup and measurement durations

The `DP` value is implied by:

```text
DP = world size / (TP * PP)
```

## Outputs

Profiler outputs are usually stored under:

- `calibrations/`: scheduler/profile JSON files
- `power_logs/`: NVML power traces
- `results/resharding/`: resharding timing CSVs
- `results/layer_profiles/`: per-layer timing CSVs and plots

## Inspecting Commands

These commands inspect arguments without launching training:

```bash
python train_profile.py --help
python compute_power_multinode.py --help
python scripts/profile_per_layer.py --help
```

For a dry-run style check of valid configurations:

```bash
python scripts/run_all_configs.py --model gpt2 --dry-run
```

## Running Profiling Jobs

Profiling jobs are cluster-specific and may be expensive. Review container
paths, bind mounts, Slurm account/partition settings, and output directories
before running them.

Example shape:

```bash
python train_profile.py \
  --model gpt2_medium \
  --profile-menu calibrations/profile_menu_4gpu.json \
  --output calibrations/calibration_gpt2_medium_4gpu.json \
  --num-gpus 4 \
  --container-image /path/to/nemo.sif \
  --container-bind /path/to/Megatron-LM:/opt/megatron-lm \
  --megatron-dir /opt/megatron-lm \
  --work-dir runs/profile_workdir
```

This is a template, not a guaranteed paper-reproduction command. The exact
container, Megatron checkout, model settings, and hardware should match the
experiment you intend to reproduce.

## Hardware Notes

The paper reports that on a single 4-GPU L40 setup, `TP=4` had the lowest
measured average power but also lower throughput. That result is not a
universal rule. Multi-node communication can change the ranking of both power
and throughput, so new hardware should be profiled rather than assumed to
match the paper setup.


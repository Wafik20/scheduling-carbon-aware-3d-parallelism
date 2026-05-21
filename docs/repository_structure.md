# Repository Structure

This repository is organized around a profiler, a scheduler/simulator, and
preserved experiment artifacts.

## Core Source

- `src/carbon_aware_3d/carbon_aware_env.py`
  - Carbon trace loading.
  - Pure-Python simulator (`CarbonSim`).
  - Static, Greedy/smart-switch, fixed-start dynamic, job-shift, and joint
    shift+dynamic rollout helpers.
  - Optional Gymnasium wrapper for RL experiments.
  - Plotting helpers.
- `src/carbon_aware_3d/compute_power.py`
  - Single-node NVML power monitor used by training launchers.

## Main Root Entrypoints

- `evaluate_carbon_policy.py`
  - Main scheduler/simulator entrypoint.
  - Compares static policies, Greedy, Fixed-Start Dynamic Parallelism, Job
    Shifting, and Job Shifting + Dynamic Parallelism.
- `train_profile.py`
  - Megatron profiling launcher.
  - Reads profile menus from `calibrations/` and writes measured calibration
    JSON.
- `train.py`
  - Fixed or scheduled Megatron training launcher with power monitoring.
- `train_elastic.py`
  - Planned elastic Megatron schedule launcher.
- `train_online_elastic.py`
  - Online elastic launcher driven by calibration JSON and carbon forecast CSV.
- `train_naive_elastic.py`
  - Naive carbon-bucket elastic launcher.
- `train_carbon_policy.py`
  - Optional PPO/DQN policy training. This appears exploratory relative to the
    paper-facing evaluation path.
- `compute_power_multinode.py`
  - Multi-node NVML sidecar sampler.
- `make_static_schedule.py`
  - Writes one-phase schedule JSON files.

## Supporting Folders

- `scripts/`
  - Utility scripts for data generation, profiling, cluster checks, and sweeps.
- `scripts/benchmarks/`
  - Resharding benchmark launcher and inner benchmark script.
- `batch_files/`
  - Slurm submission scripts. These contain original cluster defaults and
    should be reviewed before reuse.
- `calibrations/`
  - Strategy menus, scheduler config files, and measured calibration JSONs.
- `schedules/`
  - Example one-phase and phased schedule JSON files.
- `data/`
  - Carbon-intensity traces. The paper uses Electricity Maps traces for AWS
    `us-west-1` in 2024; verify the exact CSV before reproducing a result.
- `power_logs/`
  - NVML power traces from profiling/training runs.
- `runs/`
  - Simulator and training run outputs. This directory can become large and is
    ignored by default.
- `figures/experiment_plots/`
  - Generated plots from previous analysis runs.
- `results/layer_profiles/`
  - Per-layer profiling tables and plots.
- `results/resharding/`
  - Resharding benchmark outputs.
- `results/logs/`
  - Historical local `.out` logs.
- `notebooks/`
  - Exploratory visualization notebooks. Some cells may contain historical
    absolute paths.
- `archive/`
  - Old notes or one-off artifacts kept for provenance.
- `docs/`
  - Researcher-facing documentation.

## Documentation Files

- `docs/overview.md`: conceptual overview.
- `docs/profiler.md`: profiling workflow and measured quantities.
- `docs/scheduler.md`: simulator inputs, outputs, and commands.
- `docs/policies.md`: policy definitions and naming notes.
- `docs/reproducing_experiments.md`: practical command examples.
- `docs/experiment_outputs.md`: output artifact guide.
- `docs/cleanup_report.md`: historical cleanup note.

## Compatibility Notes

Run the root entrypoints from the repository root. For direct imports in an
interactive session, install the package in editable mode:

```bash
python -m pip install -e .
```

or set:

```bash
export PYTHONPATH=src
```

Some scripts and notebooks still reflect the original development cluster.
Treat hardcoded paths as examples unless they match your environment.


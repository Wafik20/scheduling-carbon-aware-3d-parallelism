# Repository Structure

The repository is organized around a small simulator package, runnable
entrypoints, and preserved experiment artifacts.

## Core Source

- `src/carbon_aware_3d/carbon_aware_env.py`
  - Carbon trace loading.
  - `CarbonSim`, the pure-numpy simulator.
  - Gymnasium wrapper for optional RL.
  - Static, Greedy, fixed-start oracle, job-shift, and joint shift+dynamic
    rollout helpers.
  - Plotting helpers.
- `src/carbon_aware_3d/compute_power.py`
  - Single-node NVML power monitor used by the training launchers.

## Root Entrypoints

- `evaluate_carbon_policy.py`: main no-GPU simulator evaluation.
- `train.py`: fixed or scheduled Megatron training with power monitoring.
- `train_elastic.py`: planned elastic Megatron schedule launcher.
- `train_online_elastic.py`: online elastic launcher driven by a calibration
  JSON and carbon forecast CSV.
- `train_naive_elastic.py`: simple carbon-bucket baseline launcher.
- `train_carbon_policy.py`: optional PPO/DQN training.
- `train_profile.py`: calibration/profile launcher.
- `compute_power_multinode.py`: sidecar sampler for multi-node sweeps.
- `make_static_schedule.py`: writes one-phase schedule JSONs.

## Supporting Folders

- `scripts/`: runnable utilities that are not the main paper-facing
  entrypoint. `scripts/benchmarks/` holds the resharding benchmark.
- `batch_files/`: Slurm job scripts. These still assume the original cluster
  paths unless environment variables or script arguments override them.
- `calibrations/`: strategy menus and scheduler calibration JSON.
- `schedules/`: example static and phase schedules.
- `data/`: carbon-intensity traces and a README describing the expected CSV
  schema.
- `power_logs/`: measured GPU power traces used by analysis notebooks and
  plots.
- `runs/`: simulator/training outputs. This directory is ignored by default
  because production runs can become large.
- `figures/experiment_plots/`: generated analysis plots formerly under
  `plots/`.
- `results/layer_profiles/`: per-layer profiling results formerly under
  `layer_profile_results/`.
- `results/logs/`: historical `.out` logs formerly under `logs/`.
- `results/resharding/`: resharding benchmark CSVs.
- `notebooks/`: exploratory notebooks formerly in the repository root.
- `archive/`: loose root artifacts kept because their provenance was unclear.

## Compatibility Notes

The main root commands still run from the repository root. Internal imports
now point at the package under `src/`. For ad hoc Python sessions, either run
`python -m pip install -e .` or set `PYTHONPATH=src`.

Old artifact paths:

- `plots/` -> `figures/experiment_plots/`
- `layer_profile_results/` -> `results/layer_profiles/`
- `logs/` -> `results/logs/`
- `reshard_times.csv` -> `results/resharding/reshard_times.csv`
- root notebooks -> `notebooks/`


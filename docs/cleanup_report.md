# Cleanup Report

This report records the repository cleanup performed to make the project
easier to clone, inspect, and reuse.

## Moved or Renamed

- `carbon_aware_env.py` -> `src/carbon_aware_3d/carbon_aware_env.py`
- `compute_power.py` -> `src/carbon_aware_3d/compute_power.py`
- `benchmark_inner.py` -> `scripts/benchmarks/benchmark_inner.py`
- `benchmark_transform.py` -> `scripts/benchmarks/benchmark_transform.py`
- `visualize_multinode.ipynb` -> `notebooks/visualize_multinode.ipynb`
- `visualize_optimization.ipynb` -> `notebooks/visualize_optimization.ipynb`
- `visualize_training.ipynb` -> `notebooks/visualize_training.ipynb`
- `plots/` -> `figures/experiment_plots/`
- `layer_profile_results/` -> `results/layer_profiles/`
- `logs/` -> `results/logs/`
- `reshard_times.csv` -> `results/resharding/reshard_times.csv`
- `paper_sections/` -> `docs/paper_sections/`

## Archived

- `output.png` -> `archive/loose-root-files/output.png`
- `todo.md` -> `archive/notes/todo.md`
- `profile_workdir/` -> `archive/empty_workdirs/profile_workdir/`

These were not deleted because they may contain local notes or one-off
artifacts that could still be useful.

## Code and Configuration Updates

- Updated root entrypoints to import reusable code from
  `src/carbon_aware_3d/`.
- Fixed `scripts/run_all_configs.py` so it invokes root `train.py`.
- Updated `scripts/profile_per_layer.py` default output path to
  `results/layer_profiles/`.
- Updated the reshard benchmark to write under `results/resharding/`.
- Updated `scripts/run_hybrid_local.sh` to write `.out` files under
  `results/logs/`.
- Added `pyproject.toml` and `requirements.txt`.
- Updated `env.yaml`.
- Relaxed `.gitignore` so small carbon trace CSVs in `data/` can be tracked,
  while large training datasets/checkpoints remain ignored.

## Documentation Added

- `README.md`
- `docs/repository_structure.md`
- `docs/overview.md`
- `docs/profiler.md`
- `docs/scheduler.md`
- `docs/policies.md`
- `docs/reproducing_experiments.md`
- `docs/experiment_outputs.md`
- `data/README.md`

## Unresolved

- The initial cleanup happened before manuscript-specific context was
  available. Later documentation was updated to use the paper title
  "Carbon-Aware Dynamic Parallelism for Distributed Training" and the
  paper-facing policy terminology.
- Some notebooks still contain historical absolute paths in saved outputs or
  exploratory cells. They were moved but not rewritten.
- Slurm batch files still contain original cluster defaults. Override
  `PROJECT_DIR`, `CONTAINER`, `MEGATRON_HOST`, and related variables before
  submitting on another cluster.

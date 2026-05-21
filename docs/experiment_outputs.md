# Experiment Outputs

## Simulator Runs

`evaluate_carbon_policy.py` writes to the directory passed with `--out`.
Common files include:

- `results.csv`: one row per episode and policy.
- `aggregate.png`: aggregate carbon comparison.
- `aggregate_feasible_only.png`: aggregate comparison after filtering to
  policies that completed within the deadline, when applicable.
- `rollout_offset_<h>h.png`: detailed carbon-intensity, strategy, and
  cumulative-carbon plot for selected episodes.
- `synergy_summary.csv`: produced when joint job shifting and dynamic
  parallelism are evaluated.
- `shift_switch_diagnostics.csv`: produced by `--debug-shift-switch`.

Because `runs/` can grow quickly, it is ignored by default. Keep only curated
run outputs in version control.

## Power Logs

`power_logs/` stores measured NVML traces. Single-node launchers write one CSV
per model/config. Multi-node sweeps write:

```text
power_logs/<sweep>/<tp_pp_dp>/rank<rank>.csv
```

Each row records timestamped GPU power and, when available, the current
training step/config tag.

## Calibration and Schedules

- `calibrations/*.json`: strategy menus and measured constants consumed by
  online schedulers.
- `schedules/*.json`: one-phase or phased schedules consumed by
  `train.py` / `train_elastic.py`.

`make_static_schedule.py` can regenerate simple one-phase schedules.

## Figures

Generated analysis plots from the earlier root-level `plots/` directory now
live under `figures/experiment_plots/`. These are preserved artifacts, not the
source of truth for the simulator.

## Layer Profiles

`results/layer_profiles/` contains per-layer timing CSVs, summary JSON/CSV,
and generated plots from `scripts/profile_per_layer.py`.

## Resharding Benchmarks

`results/resharding/reshard_times.csv` contains reshard timing measurements
from `scripts/benchmarks/benchmark_transform.py` and
`scripts/benchmarks/benchmark_inner.py`.

## Archived Files

`archive/` contains loose root-level files whose purpose was unclear or
one-off:

- `archive/loose-root-files/output.png`
- `archive/notes/todo.md`
- `archive/empty_workdirs/profile_workdir/`

They were moved for discoverability and kept intact.


# Experiment Outputs

This document explains where outputs are stored and what they usually mean.
It does not guarantee that every directory is a paper reproduction artifact;
some outputs appear to be exploratory or historical.

## Simulator Runs

`evaluate_carbon_policy.py` writes to the directory passed with `--out`,
usually under `runs/`.

Common files:

- `results.csv`
  - Per-episode, per-policy results.
  - Includes carbon, energy, wall time, wait time, completion status, and
    strategy sequence information.
- `aggregate.png`
  - Aggregate comparison plot across policies.
- `aggregate_feasible_only.png`
  - Aggregate plot restricted to policies that complete within the deadline,
    when applicable.
- `rollout_offset_<h>h.png`
  - Per-episode detail plot showing carbon intensity, selected strategy, and
    cumulative carbon.
- `superadditivity.csv`
  - Joint job-shifting/dynamic-parallelism summary when that path is enabled.
- `shift_switch_diagnostics.csv`
  - Debug table for candidate start times and dynamic schedules when
    diagnostics are enabled.

Because `runs/` can grow quickly, it is ignored by default.

## Carbon Traces

Carbon traces live under `data/`.

Expected columns:

```text
year, month, day, time, carbon
```

The `carbon` column is carbon intensity in gCO2/kWh. The paper uses
Electricity Maps traces for AWS `us-west-1` in 2024. The current repository
contains several 2024 traces and some AWS region-specific examples.

## Calibration Files

`calibrations/` contains JSON files used by profiling and scheduling scripts.
These may include:

- profile menus listing `(tp, pp)` choices
- scheduler configs
- measured strategy tables
- static fast/slow profile examples

Calibration files are the bridge between GPU profiling and simulator
evaluation.

## Power Logs

`power_logs/` stores NVML power traces.

Single-node launchers generally write one CSV per model/config. Multi-node
sweeps write one CSV per node/rank under a sweep/config directory:

```text
power_logs/<sweep>/<tp_pp_dp>/rank<rank>.csv
```

Rows typically include timestamped GPU power and, when available, training
step or configuration tags.

## Figures

`figures/experiment_plots/` contains generated plots from previous analyses.
These are useful for inspection, but the source of truth should be the
corresponding scripts, inputs, and run outputs.

## Layer Profiles

`results/layer_profiles/` contains outputs from `scripts/profile_per_layer.py`,
including:

- per-layer timing CSVs
- block timing CSVs
- summary CSV/JSON
- architecture-comparison plots

These outputs support analysis of why models such as GPT-2, Llama-style, and
Pythia may respond differently to TP/PP choices.

## Resharding Benchmarks

`results/resharding/reshard_times.csv` contains switching/resharding timing
measurements from the benchmark scripts under `scripts/benchmarks/`.

Switching cost is important because dynamic parallelism can only help when the
carbon savings from switching outweigh the lost time/energy.

## Logs

`results/logs/` contains historical `.out` logs. These appear to be local run
logs kept for provenance and debugging.

Slurm scripts write to `slurm_logs/` by default. That directory is ignored.

## Notebooks

`notebooks/` contains exploratory visualization notebooks. Some saved outputs
or cells may contain absolute paths from the original environment.

## Archive

`archive/` contains old notes or unclear one-off artifacts retained for
provenance rather than deleted.


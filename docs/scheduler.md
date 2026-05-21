# Scheduler and Simulator

The scheduler/simulator evaluates carbon-aware parallelism policies without
launching GPU training. It uses measured configuration profiles and hourly
carbon traces to simulate pretraining progress, energy, and carbon emissions.

## Main Entrypoint

```text
evaluate_carbon_policy.py
```

Core simulator logic lives in:

```text
src/carbon_aware_3d/carbon_aware_env.py
```

The simulator includes carbon trace loading, static rollouts, dynamic
parallelism rollouts, job shifting, plotting helpers, and an optional
Gymnasium wrapper for RL experiments.

## Inputs

The scheduler uses:

- token budget
- deadline or maximum episode duration
- hourly carbon-intensity trace
- profiled power and throughput table
- switching or resharding cost
- set of feasible `(TP, PP, DP)` configurations

Carbon trace CSVs are stored under `data/`. They are expected to contain:

```text
year, month, day, time, carbon
```

where `carbon` is carbon intensity in gCO2/kWh.

The paper uses Electricity Maps traces for AWS `us-west-1` in 2024. The
repository contains several trace files, including AWS region-specific files
for some months and broader 2024 traces. Verify the exact file before using a
command as a paper reproduction run.

## Outputs

`evaluate_carbon_policy.py` writes to the directory passed with `--out`.
Common outputs include:

- `results.csv`
- `aggregate.png`
- `aggregate_feasible_only.png`, when applicable
- `rollout_offset_<h>h.png`
- `superadditivity.csv`, when joint shifting/dynamic policy is evaluated
- `shift_switch_diagnostics.csv`, when diagnostics are enabled

See `docs/experiment_outputs.md` for more detail.

## Smoke Test

This small run does not launch GPU training:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_week.csv \
  --tokens 10000000 \
  --max-episode-hours 2 \
  --forecast-horizon-h 2 \
  --oracle-token-quantum 1000000 \
  --n-episodes 2 \
  --n-detail-rollouts 1 \
  --seed 1 \
  --out runs/smoke_eval
```

## Policy Comparison Template

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 4000000000 \
  --max-episode-hours 48 \
  --forecast-horizon-h 6 \
  --smart-fast-idx -1 \
  --smart-lookahead-h 16 \
  --smart-candidates 2,5 \
  --smart-margin-g 10 \
  --include-job-shift \
  --shift-window-h 24 \
  --shift-step-h 1 \
  --oracle-token-quantum 1000000 \
  --n-episodes 32 \
  --seed 123 \
  --out runs/eval_static_smart_oracle
```

This command exercises the main simulator path, but it should be treated as a
template unless the trace, model profile, token budget, deadline, and oracle
quantization are confirmed against the paper experiment.

## Important Arguments

- `--trace`: hourly carbon CSV.
- `--tokens`: training token budget for the simulated job.
- `--max-episode-hours`: deadline from job arrival to completion.
- `--include-job-shift`: include job shifting and joint
  shifting+dynamic-parallelism policies.
- `--shift-window-h`: maximum allowed start delay.
- `--oracle-token-quantum`: token discretization used by the offline dynamic
  programming oracle. Smaller values can be more accurate but slower.
- `--n-episodes`: number of sampled trace offsets.
- `--fixed-offset-h`: debug one specific start offset.
- `--filter-strategies`: filter configurations by a simple memory feasibility
  estimate.

## Feasibility Filtering

The simulator can remove configurations that appear infeasible for a model and
GPU-memory budget:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 4000000000 \
  --model-name llama_3b \
  --model-num-params 3000000000 \
  --filter-strategies \
  --gpu-memory-gb 48 \
  --usable-memory-frac 0.90 \
  --n-episodes 32 \
  --out runs/llama_3b_feasible_eval
```

This appears to be a coarse model-memory filter, not a replacement for an
actual Megatron fit test.

## Caveats

- Simulator accuracy depends on the measured profile table.
- If a new model, GPU, node count, or interconnect is used, rerun profiling.
- Some CLI names still refer to `smart` switching even when the paper-facing
  policy name is Greedy.
- Optional PPO/DQN code exists but appears exploratory and is not the default
  paper evaluation path.


# Reproducing and Inspecting Experiments

This repository contains both no-GPU scheduler simulations and GPU profiling
launchers. The simulator commands are safe to inspect on a CPU machine. The
profiling and training commands are cluster-specific and can be expensive.

Do not treat a command here as a paper-figure reproduction command unless the
trace, profile table, token budget, deadline, model, and hardware settings
have been checked against the paper experiment.

## 1. No-GPU Simulator Smoke Test

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

Expected output:

- `runs/smoke_eval/results.csv`
- `runs/smoke_eval/aggregate.png`
- one or more `rollout_offset_<h>h.png` detail plots

This command should not launch Megatron or use GPUs.

## 2. Scheduler / Simulator Policy Comparison

Template command:

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

This exercises the main simulator path and may compare:

- Best Static
- Greedy
- Fixed-Start Dynamic Parallelism
- Job Shifting
- Job Shifting + Dynamic Parallelism

Notes:

- The script still uses `smart-*` CLI names for the Greedy/smart-switch path.
- `--oracle-token-quantum` controls the dynamic-programming discretization.
  Smaller values can be slower.
- `--n-episodes` controls how many trace offsets are sampled.
- `--include-job-shift` enables the job-shifting policies.

## 3. Feasibility Filtering

Use feasibility filtering when a model size may not fit all TP/PP/DP
configurations:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 4000000000 \
  --model-name llama_3b \
  --model-num-params 3000000000 \
  --filter-strategies \
  --gpu-memory-gb 48 \
  --usable-memory-frac 0.90 \
  --include-job-shift \
  --n-episodes 32 \
  --out runs/llama_3b_feasible_eval
```

The current repository suggests this is a coarse memory estimate. It does not
replace an actual Megatron run on the target hardware.

## 4. Carbon Traces

Carbon traces live in `data/`. The paper uses Electricity Maps traces for AWS
`us-west-1` in 2024. The repository contains several 2024 trace files and some
AWS-region-specific examples.

Before reproducing a paper result, verify:

- region
- date range
- hourly formatting
- whether the trace should wrap or stop at the end

The simulator expects CSV columns:

```text
year, month, day, time, carbon
```

## 5. Inspecting Profiling Scripts

These commands print available arguments without launching long jobs:

```bash
python train_profile.py --help
python train.py --help
python train_elastic.py --help
python train_online_elastic.py --help
python train_naive_elastic.py --help
python compute_power_multinode.py --help
python scripts/profile_per_layer.py --help
```

To inspect the valid 4-GPU configurations for GPT-2-style settings:

```bash
python scripts/run_all_configs.py --model gpt2 --dry-run
```

## 6. Profiling Template

Profiling requires Megatron Core / Megatron-LM, PyTorch/CUDA, Apptainer, and
NVML access. The exact command depends on your cluster.

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

Review `docs/profiler.md` before running this. It can launch real GPU jobs.

## 7. Slurm Sweeps

Slurm scripts live in `batch_files/`. The multi-node submit helper is:

```bash
SMOKE_TEST=1 scripts/submit_gpt2m_node_sweeps.sh
```

Before submitting, check:

- Slurm account
- partition
- GPU type
- node count
- container path
- Megatron path
- output directories

## 8. Optional RL Baseline

RL experiments are optional and appear exploratory relative to the main paper
evaluation path.

Install optional dependencies:

```bash
python -m pip install -e ".[rl]"
```

Train:

```bash
python train_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 3000000000 \
  --total-timesteps 200000 \
  --algo ppo \
  --out runs/ppo_carbon
```

Evaluate a saved policy:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --include-rl \
  --model runs/ppo_carbon/model.zip \
  --algo ppo \
  --tokens 3000000000 \
  --n-episodes 32 \
  --out runs/eval_with_rl
```


# Carbon-Aware Dynamic Parallelism for Distributed Training

This repository supports the paper **"Carbon-Aware Dynamic Parallelism for
Distributed Training."** It studies large language model pretraining as a
carbon-aware scheduling problem: instead of choosing one distributed
parallelism configuration at launch and keeping it fixed, the scheduler can
treat the parallelism configuration itself as a control knob.

When grid carbon intensity is high, the system can switch to lower-power
parallel configurations. When carbon intensity is lower, it can switch back to
higher-throughput configurations to recover progress while still meeting the
training token budget and deadline.

## Research Problem

Distributed pretraining usually fixes its 3D parallelism layout for the full
job. That is simple, but it assumes the best configuration is independent of
time and carbon intensity. This project asks:

- How do DP/TP/PP configurations change power, throughput, energy per token,
  and switching overhead?
- Can dynamic parallelism reduce carbon emissions compared with the best
  deadline-feasible static configuration?
- How much additional benefit is available when the scheduler can also delay
  the job start time?

The paper reports that fixed-start dynamic parallelism gives modest but
meaningful carbon reductions over Best Static, up to about **3.4%**, with one
Greedy episode saving **4.7%**. Those savings depend on the measured
model/hardware/configuration profile and the carbon trace.

## Main Idea

The project has two main pieces:

1. **Profiler:** run short Megatron Core / Megatron-LM jobs for feasible
   parallel configurations and measure the cost/performance table.
2. **Scheduler / simulator:** combine the measured table with a token budget,
   deadline, carbon-intensity trace, and switching costs to choose a
   configuration over time.

The scheduler is intentionally profile-driven. Power and throughput depend on
the model, hardware, and parallel configuration, so the code should not assume
one universally best DP/TP/PP layout. In the paper's single 4-GPU L40 setup,
`TP=4` had the lowest measured average power, but also lower throughput. That
ranking should not be assumed to generalize to multi-node systems, where
cross-node communication can change both power and throughput.

## 3D Parallelism

Configurations are written as:

```text
(TP, PP, DP)
```

- **TP = tensor parallelism:** shards tensor operations within layers.
- **PP = pipeline parallelism:** partitions model layers into pipeline stages.
- **DP = data parallelism:** replicates the model state across data-parallel
  groups and splits batches.

A valid configuration must satisfy:

```text
TP * PP * DP = world size
```

For example, on 4 GPUs, the valid configurations are:

```text
(4, 1, 1)
(1, 4, 1)
(1, 1, 4)
(2, 2, 1)
(2, 1, 2)
(1, 2, 2)
```

## Profiler

The profiler runs short distributed training jobs for each feasible
configuration `pi = (TP, PP, DP)`. For each configuration, it measures or
derives:

- steady-state power
- throughput
- energy per step
- energy per token
- switching or resharding cost between configurations

Throughput comes from Megatron training logs or metrics. Power is sampled with
NVML and aggregated across the GPUs assigned to the job. The profile outputs
are stored as calibration JSON, CSV summaries, and power logs that the
simulator can consume.

Main profiler-related files:

- `train_profile.py`: launcher for Megatron profiling runs.
- `scripts/profile_per_layer.py`: exploratory per-layer timing profiler.
- `scripts/benchmarks/benchmark_transform.py`: resharding benchmark launcher.
- `compute_power_multinode.py`: multi-node NVML power-monitor sidecar.
- `calibrations/`: measured configuration/profile JSON files.
- `power_logs/`: measured GPU power traces.

See `docs/profiler.md` for details.

## Scheduler / Simulator

The scheduler uses:

- a token budget
- a deadline
- an hourly carbon-intensity trace
- a profiled power and throughput table
- switching cost between configurations

It then chooses which parallel configuration to use over time in order to
reduce carbon emissions while completing the training job before the deadline.
The main simulator entrypoint is:

```text
evaluate_carbon_policy.py
```

The simulator code lives in:

```text
src/carbon_aware_3d/carbon_aware_env.py
```

The paper uses Electricity Maps carbon-intensity traces for AWS `us-west-1` in
2024. The repository contains several carbon trace CSVs under `data/`,
including region-specific examples.

See `docs/scheduler.md` for simulator inputs, outputs, and caveats.

## Policies

The docs and scripts use the following paper-facing policy names:

- **Best Static:** the lowest-carbon static configuration that finishes before
  the deadline.
- **Greedy:** an online policy that uses the current hour's carbon intensity,
  not the future trace. Some script arguments still use older `smart-*` names.
- **Fixed-Start Dynamic Parallelism:** the job starts immediately, but the
  scheduler may switch TP/PP/DP configurations during the run.
- **Job Shifting + Dynamic Parallelism:** the scheduler may delay the start
  time and switch configurations during the run.

See `docs/policies.md` for the full definitions and naming notes.

## Repository Map

```text
README.md                      project overview and quickstart
pyproject.toml, requirements.txt, env.yaml
src/carbon_aware_3d/           reusable simulator and power-monitor modules
evaluate_carbon_policy.py      main scheduler/simulator entrypoint
train_profile.py               Megatron profile launcher
train.py                       fixed or scheduled Megatron training launcher
train_elastic.py               planned elastic schedule launcher
train_online_elastic.py        online elastic launcher
train_naive_elastic.py         naive carbon-bucket launcher
train_carbon_policy.py         optional PPO/DQN policy trainer
compute_power_multinode.py     multi-node NVML power monitor
make_static_schedule.py        helper for one-phase schedule JSON files
scripts/                       data, profiling, benchmark, and cluster helpers
scripts/benchmarks/            resharding benchmark scripts
batch_files/                   Slurm submission scripts
calibrations/                  measured strategy/profile configs
schedules/                     example static and phased schedules
data/                          carbon-intensity traces
power_logs/                    measured GPU power logs
runs/                          simulator/training run outputs
figures/experiment_plots/      generated plots from prior analyses
results/                       logs, layer profiles, and benchmark tables
notebooks/                     exploratory visualization notebooks
archive/                       old notes or one-off artifacts kept for safety
docs/                          additional documentation
```

See `docs/repository_structure.md` for a more detailed map.

## Setup

For simulator evaluation and plotting:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
```

Conda users can start from:

```bash
conda env create -f env.yaml
conda activate conda_env
```

Full profiling/training runs also require:

- Megatron Core / Megatron-LM with the project-specific profiling or elastic
  entrypoints available inside the container.
- PyTorch and CUDA versions compatible with the cluster.
- Apptainer for the existing launch scripts.
- Slurm for the batch files.
- NVML access for power measurement.

Cluster paths in the scripts reflect the original development environment;
override container, bind, Megatron, and work directories before using another
system.

## Smoke Test

This is a small no-GPU simulator check. It should create `results.csv` and a
few plots under `runs/smoke_eval/`:

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

This does not launch Megatron or GPU training.

## Running Scheduler / Simulation Experiments

A typical simulator command is:

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

This command inspects the scheduler/simulator path, but you should verify the
exact trace, token budget, deadline, and calibration settings against the paper
experiment before treating it as a figure-reproduction command.

More examples are in `docs/reproducing_experiments.md`.

## Running or Inspecting Profiling Scripts

Profiling launches Megatron jobs and should be run only on the intended GPU
cluster. Start with dry runs where available:

```bash
python train_profile.py --help
python train_online_elastic.py --help
python train.py --help
python scripts/run_all_configs.py --model gpt2 --dry-run
```

The Slurm scripts in `batch_files/` show how the original experiments were
submitted. Review paths, partitions, accounts, model settings, and output
directories before using them.

## Outputs, Figures, Logs, and Traces

- `runs/`: simulator and training run outputs.
- `data/`: hourly carbon-intensity traces.
- `calibrations/`: profiler output JSON and scheduler config files.
- `power_logs/`: NVML power traces.
- `figures/experiment_plots/`: generated analysis plots.
- `results/layer_profiles/`: per-layer profiling outputs.
- `results/resharding/`: resharding benchmark tables.
- `results/logs/`: historical local `.out` logs.

See `docs/experiment_outputs.md` for a file-by-file guide.

## Hardware Assumptions and Limitations

- The paper evaluates GPT-2, Llama-style, and Pythia models.
- The training/profiling path uses Megatron Core / Megatron-LM.
- The paper carbon traces come from Electricity Maps for AWS `us-west-1` in
  2024.
- The single-node measurements discussed in the paper include a 4-GPU L40
  setup. Multi-node behavior can differ because cross-node communication
  changes throughput and power.
- The simulator is only as accurate as the profiled power/throughput/switching
  table used as input.
- Some scripts are exploratory or cluster-specific. The current repository
  suggests several older notebooks and generated outputs were kept for
  provenance rather than as clean reproduction entrypoints.
  

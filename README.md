# Scheduling Carbon-Aware 3D Parallelism

Research code for studying carbon-aware scheduling of large-model
pretraining under elastic 3D parallelism. The project connects measured
DP/TP/PP strategy profiles with hourly carbon-intensity traces to ask when a
training job should run, which parallel configuration it should use, and how
much energy/carbon each choice costs.

The current paper-facing evaluation path is simulator based. GPU launchers and
profiling scripts are included for collecting the power, throughput, and
resharding measurements that parameterize the simulator.

## Main Idea

Large pretraining jobs can run under several valid 3D parallel configurations.
Those configurations have different throughput, power draw, energy per token,
and resharding costs. This repository compares:

- **Static:** choose one DP/TP/PP strategy and never switch.
- **Greedy / smart-switch:** an online heuristic that uses limited carbon
  lookahead to switch between faster and lower-power strategies.
- **Fixed Start-Time dynamic parallelism:** an offline dynamic-programming
  oracle that changes DP/TP/PP during a job, then replays the action sequence
  in the exact simulator.
- **Job Shifting:** delay the start of a job within an allowed slack window.
- **Job Shifting + Dynamic Parallelism:** jointly choose a delayed start time
  and a dynamic DP/TP/PP schedule.

Optional PPO/DQN experiments exist, but they are not the default evaluation
pipeline.

## Terminology

**DP (data parallelism)** replicates the model across workers and splits data
batches. In this codebase DP is inferred as `n_gpus / (tp * pp)` for a
strategy.

**TP (tensor parallelism)** shards tensor operations within a layer across
GPUs.

**PP (pipeline parallelism)** shards model layers into pipeline stages.

**Dynamic parallelism** means changing the active `(tp, pp, dp)` strategy over
the course of one pretraining job. The simulator charges measured warmup,
checkpoint/save, and resharding costs when a switch happens.

**Job shifting** means delaying a job's start time to run during cleaner carbon
intensity, while still respecting the user-specified deadline/slack.

## Repository Layout

```text
README.md                      project overview and quickstart
pyproject.toml, requirements.txt, env.yaml
src/carbon_aware_3d/           reusable simulator and power-monitor modules
evaluate_carbon_policy.py      main simulator evaluation entrypoint
train*.py                      Megatron/elastic training launchers
compute_power_multinode.py     multi-node NVML power monitor sidecar
make_static_schedule.py        helper for static schedule JSONs
scripts/                       data, profiling, benchmark, and cluster helpers
batch_files/                   Slurm submission scripts
calibrations/                  measured strategy/profile configs
schedules/                     example static and phased schedules
data/                          hourly carbon-intensity traces
power_logs/                    measured GPU power logs used for analysis
runs/                          simulator/training run outputs
figures/experiment_plots/      generated plots from prior analysis
results/                       logs, layer profiles, and benchmark tables
notebooks/                     exploratory visualization notebooks
archive/                       old or unclear loose root files kept for safety
docs/                          additional reproduction and output notes
```

See `docs/repository_structure.md` for more detail.

## Install

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
conda activate gpu_env
```

Full GPU training launchers also require a Megatron-LM checkout, a compatible
PyTorch/CUDA stack, Apptainer, and cluster-specific container/bind paths.
Those paths are exposed as command-line flags on the training scripts.

## Smoke Test

This runs the no-GPU simulator on a small token budget and writes plots/tables
under `runs/smoke_eval`:

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

## Main Simulator Evaluation

The main reusable evaluation script is `evaluate_carbon_policy.py`. A typical
paper-style run comparing static, Greedy, fixed-start dynamic parallelism, job
shifting, and joint shifting+dynamic parallelism is:

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

Useful variants:

- Add `--filter-strategies --model-num-params <N>` to remove infeasible
  strategies by a simple per-GPU memory estimate.
- Add `--fixed-offset-h <hours>` when debugging one trace start offset.
- Add `--include-rl --model <path.zip> --algo ppo|dqn` for an optional saved
  Stable-Baselines policy.

## GPU Measurement and Profiling

The GPU launchers were written for Slurm + Apptainer + Megatron-LM. They are
not required for the simulator smoke test.

- `train.py`: fixed or scheduled Megatron training with single-node power
  monitoring.
- `train_elastic.py`: planned elastic schedule using Megatron elastic
  pretraining.
- `train_online_elastic.py`: online scheduler using calibration JSON and a
  carbon forecast CSV.
- `train_naive_elastic.py`: carbon-bucket baseline launcher.
- `train_profile.py`: profile strategy menus and write calibration JSON.
- `scripts/profile_per_layer.py`: per-layer timing profiles, defaulting to
  `results/layer_profiles/`.
- `scripts/benchmarks/benchmark_transform.py`: resharding benchmark launcher;
  its CSV is written to `results/resharding/reshard_times.csv`.
- `batch_files/` and `scripts/submit_gpt2m_node_sweeps.sh`: Slurm sweeps for
  multi-node DP/TP/PP measurements.

See `docs/reproducing_experiments.md` for command examples.

## Outputs

Simulator outputs go under `runs/<name>/` and usually include `results.csv`,
aggregate plots, selected rollout plots, and diagnostic CSVs when debugging is
enabled. Measured GPU power traces are in `power_logs/`. Existing generated
figures were moved to `figures/experiment_plots/`. Historical local `.out`
logs were moved to `results/logs/`.

See `docs/experiment_outputs.md` for a file-by-file guide.

## Hardware Assumptions

The simulator default strategy table is an 8-GPU measured profile with
48 GB-per-GPU memory used by the feasibility filter unless overridden.
Several live-training scripts default to the original development cluster's
4-GPU-node Slurm/Apptainer setup and paths. For another cluster, pass
`--container-image`, repeated `--container-bind`, `--megatron-dir`,
`--num-gpus`, and `--work-dir` explicitly.

The code intentionally separates the no-GPU simulator evaluation from the
cluster launchers so another researcher can inspect and reproduce scheduling
logic before collecting new hardware measurements.


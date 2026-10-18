# Carbon-aware distributed training

Research code for profiling DP/TP/PP configurations, scheduling training against
carbon-intensity traces, and comparing carbon emissions under a token budget
and deadline. A valid layout satisfies `TP * PP * DP = world size`.

This is a fresh-start codebase. Historical measurements, experiment outputs,
notebooks, datasets, and cluster-specific submissions are not included. Generate
new traces and profiles for each study; do not interpret synthetic demo results
as hardware measurements.

## Install

Python 3.10+ is required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

`pyproject.toml` is the single dependency manifest. Optional extras:
`power` for NVML monitoring, `rl` for Gymnasium/Stable-Baselines3 policies,
and `profile` for PyTorch/Transformers layer profiling. Install only the extras
you use. GPU launchers additionally need Apptainer, compatible NVIDIA drivers,
and an external Megatron checkout/container. These are not installed by pip.

## CPU evaluation

```bash
python scripts/generate_synthetic_dataset.py --out runs/carbon.csv
python evaluate_carbon_policy.py --trace runs/carbon.csv \
  --tokens 10000000 --max-episode-hours 2 --forecast-horizon-h 2 \
  --oracle-token-quantum 1000000 --n-episodes 2 \
  --n-detail-rollouts 1 --seed 1 --out runs/smoke_eval
python -m pytest -q
```

The built-in strategy profile is **synthetic**, for smoke tests only. For research,
supply `--strategy-profile runs/calibration.json --n-gpus N`. The evaluator accepts
profiler calibration JSON (`{"strategies": [...]}`) or strategy-profile CSV.
Carbon CSVs must contain hourly `year,month,day,time,carbon` rows; `carbon` is
in gCO2/kWh. Supply your own measured regional trace for real experiments.

Evaluation compares static layouts, Greedy scheduling, and a discretized offline
oracle. `--include-job-shift` also tests delayed starts. Results and diagnostic
plots go under `--out`. Compare deadline-feasible completed jobs against the
same episode's best feasible static layout; incomplete jobs are not savings.
Use `--help` for token budgets, deadlines, switching costs, and oracle resolution.

Optional policy training:

```bash
python -m pip install -e '.[rl]'
python train_carbon_policy.py --trace runs/carbon.csv --out runs/policy
```

This trains a simulator policy, not a language model. The policy-training CLI
currently uses synthetic defaults only; measured-profile training still needs
configuration wiring. Evaluation of saved policies
uses `--include-rl --model PATH --algo ppo` (or `dqn`).

## GPU training and profiling

Run a repository Python script in the NeMo 26.08 container:

```bash
bash scripts/run_container.sh scripts/smoke_train.py
```

The wrapper uses NVIDIA GPUs, mounts this repository at `/workspace/project`,
and mounts `/vast/cs2731_2026f/wat29/Megatron-LM` over the container's
`/opt/Megatron-Bridge/3rdparty/Megatron-LM`. The default image is
`/vast/cs2731_2026f/wat29/containers/nemo_26.08.sif`.
Override host paths with `CONTAINER_IMAGE` and `MEGATRON_FORK`.
Relative script and override paths resolve from the repository root; additional
arguments pass through unchanged, with the container working directory set to
`/workspace/project`. From another directory, invoke the wrapper by its full
path. It loads Apptainer if unavailable and returns the training exit code.
It does not request a GPU allocation. Use it within an existing GPU Slurm job.


Run only inside an appropriate GPU allocation. Inspect `--help` and use
`--dry-run` where available before launching. The supplied launcher commands use mock data. Profiles and checkpoints must match
the model, batch size, sequence length, hardware, and parallel layouts being tested.

- `train.py`: fixed-layout / scheduled training launcher.
- `train_elastic.py`: checkpoint-based planned layout changes.
- `train_online_elastic.py`: online carbon-aware layout decisions.
- `train_naive_elastic.py`: carbon-bucket baseline using the online launcher.
- `train_profile.py`: measure configuration throughput, power, and switching costs.
- `compute_power_multinode.py`: per-node NVML sidecar.
- `make_static_schedule.py`: create a single-phase schedule.

```bash
python make_static_schedule.py --tp 1 --pp 1 --dp 4 --iters 100 \
  --out-dir runs/schedules
python train_profile.py --model gpt2 --num-gpus 4 \
  --profile-menu configs/profile_menu.example.json \
  --output runs/calibrations/gpt2.json \
  --container-image /path/to/training.sif \
  --container-bind /path/to/Megatron-LM:/opt/megatron-lm --dry-run
python train_online_elastic.py --help
```

Remove `--dry-run` only when ready to profile on GPUs. The example menu targets
four GPUs. Pass the resulting calibration as `--scheduler-config` and an hourly
trace as `--carbon-forecast` to `train_online_elastic.py`; also specify `--model`,
`--train-iters`, and the container options above. Set `--bucket-seconds 3600`
for hourly traces explicitly, plus `--deadline-seconds`, `--lookahead-steps`,
`--decide-every`, and a measured `--switch-power-w`. Calibration strategies need
`tp`, `pp`, `dp`, `step_time_s`, and `power_w`; switching time comes from
`average_reshard_wall_time_s` or `--switch-time-seconds`. The external Megatron checkout
must provide `profile_strategies.py`, `online_elastic_pretrain.py`, and
`elastic_pretrain.py` (plus `pretrain_gpt.py` for static training).

For single-node Slurm jobs, activate your environment and submit
`batch_files/train.sbatch` from the repository root with the training CLI
arguments. Set your site account/partition through `sbatch`; the template
requests four GPUs.

For multi-node static training, `run_inside_container.sh` is an in-container
`torchrun` wrapper. Launch one copy per node through your cluster scheduler;
provide the Slurm rank/node variables, `GPUS_PER_NODE`, `MASTER_ADDR`,
`MASTER_PORT`, and `MEGATRON_DIR`. Set NCCL networking variables for your cluster
rather than relying on the old cluster's interface settings. The wrapper uses
mock training data; it is a profiling workload, not corpus pretraining.

## GPU power time series

Within an existing GPU Slurm allocation, wrap any training command:

```bash
bash scripts/run_with_power.sh \
  bash scripts/run_container.sh scripts/smoke_train.py

POWER_INTERVAL=1 POWER_OUTPUT_DIR="$PWD/power_logs" \
  bash scripts/run_with_power.sh srun --overlap python train_custom.py --steps 100
```

Run the wrapper **once per allocation**, not once per training rank. The training
command is responsible for launching its own distributed workers. Its arguments
and working directory are unchanged. When training uses `srun`, use `--overlap`
so its step can share resources with the monitoring step.

Monitoring runs on the host, independently of NeMo and Megatron. Install the
existing optional dependency with `.venv/bin/python -m pip install -e '.[power]'`.
`POWER_PYTHON` selects the host Python executable; by default the wrapper uses
this repository's `.venv/bin/python`, or `python3` if no repository venv exists.
The repository, Python environment, and output directory must be accessible at
the same paths on every allocated node. No GPU allocation is requested.

The wrapper discovers allocation nodes and starts one overlapping Slurm sampler
task per node. Training starts only after every sampler has flushed its first
sample. `POWER_INTERVAL` defaults to **0.5 seconds**. `POWER_STARTUP_TIMEOUT`
defaults to **60 seconds**, and `POWER_STOP_TIMEOUT` to **10 seconds** for graceful
cleanup before termination. Normal completion, training failure, and catchable
termination signals stop monitoring. Training exit codes are preserved.

`POWER_OUTPUT_DIR` defaults to the repository's `power_logs/`. Each invocation
gets a unique run directory under `job_<SLURM_JOB_ID>/`, so repeated runs do not
overwrite each other. Each node writes its own `<hostname>.csv`.

CSV rows contain UTC `timestamp`, monotonic `elapsed_s` (also retained as
`wallclock_s`), `hostname`, `slurm_job_id`, `rank`, `slurm_node_id`, `config_tag`,
`time_step`, `n_gpus`, `gpu_uuids`, `total_power_W`, and `GPU_0_W`, `GPU_1_W`, etc.
The semicolon-separated UUID list maps in order to the GPU power columns.
Failed readings are `nan`; the node total is also `nan` if any GPU reading is
missing. Raw measurements are retained; no energy integration or carbon
accounting is performed. A sampler failure during training prints a warning
without replacing the training command's exit code.

GPU selection uses node-local Slurm allocation IDs, not remapped CUDA ordinals.
When job-level IDs are empty or absent (as on CRC), the wrapper falls back to
`SLURM_STEP_GPUS` from its allocation-wide monitoring step. NVML enumeration
index 0 may represent Slurm GPU 3: selection matches NVML's device minor number,
not its enumeration index, and records its UUID. A matching visible-device count
alone is not accepted as proof of ownership.
Numeric Slurm IDs are assumed to match `/dev/nvidia*` minor numbers, as in the
usual device-file GRES configuration; custom GRES numbering requires UUID IDs.
Missing or ambiguous allocation metadata fails startup rather than sampling
unallocated GPUs. MIG allocations are rejected because parent GPU power cannot
be attributed safely to a MIG instance.

The existing in-process `PowerMonitor` APIs and results tuple layouts remain
available, as do legacy `rank<N>.csv` sidecar filenames without `--ready-dir`.
Both monitors now share the NVML implementation, use allocation-aware selection,
and write `nan` instead of zero on read failures. CSV columns have been extended.
Legacy in-process monitors prefer step GPUs; this wrapper measures the whole
job allocation. Do not enable both monitoring paths for the same training run
unless duplicate sampling is intentional.

## Layout

```text
src/carbon_aware_3d/    simulator and reusable training/power utilities
train*.py              training, profiling, and policy entrypoints
evaluate_carbon_policy.py
compute_power_multinode.py
make_static_schedule.py
scripts/               synthetic inputs and profiling helpers
batch_files/           one portable single-node Slurm example
run_inside_container.sh  multi-node torchrun wrapper
configs/               minimal input examples, not measurements
tests/                 CPU regression tests
pyproject.toml         dependencies and package metadata
runs/                  generated inputs/results (ignored)
```

Keep generated inputs, calibrations, schedules, figures, logs, and checkpoints
under `runs/` or the other ignored artifact directories. Do not commit them.

## Limitations

GPU execution depends on external Megatron entrypoints, including the project's
elastic/profiling extensions; an arbitrary upstream checkout is not sufficient.
CPU tests do not validate distributed checkpoint conversion or multi-node GPU
execution. The simulator is only as accurate as its measured throughput, power,
and switching-cost inputs. Demo defaults use 8 GPUs, 2000 W / 100k tokens/s
and 1000 W / 40k tokens/s strategies, and 50 W idle power per GPU.
Synthetic defaults have zero switching/startup costs
and are not a physical model.

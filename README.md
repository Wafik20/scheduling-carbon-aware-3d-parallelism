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

For multi-node placement, see [rank ordering](docs/profile_rank_order.md),
[the successful 4-GPU reproduction](docs/profile_4gpu.md), and
[pre-NCCL diagnostics](docs/gpu_preflight.md).

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

Run the wrapper **once per allocation**, from the batch/allocation shell before
GPU-consuming steps, not inside a GPU-restricted training step or once per rank.
The training command is responsible for launching its own distributed workers. Its arguments
and working directory are unchanged. When training uses `srun`, use `--overlap`
so its step can share resources with the monitoring step.

Monitoring runs on the host, independently of NeMo and Megatron. Install the
existing optional dependency with `.venv/bin/python -m pip install -e '.[power]'`.
`POWER_PYTHON` selects the host Python executable; by default the wrapper uses
this repository's `.venv/bin/python`, or `python3` if no repository venv exists.
The repository, Python environment, and output directory must be accessible at
the same paths on every allocated node. No GPU allocation is requested.

The wrapper discovers allocation nodes and starts one Slurm sampler
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

GPU selection uses stable UUIDs, not numeric Slurm IDs or NVML minors.
For numeric allocations, the sampler verifies `task/cgroup`,
`ConstrainDevices=yes`, and membership in the current job's step cgroup on
that node. The CUDA driver resolves the step's `CUDA_VISIBLE_DEVICES` to UUIDs;
NVML opens those UUIDs. The UUID count must match node-local `SLURM_STEP_GPUS`
(or `SLURM_JOB_GPUS` if step metadata is absent). Whole-job monitoring also
rejects differing job/step counts when both are provided. NVML-visible devices
are diagnostic inventory only, never proof of ownership. Missing guarantees,
count mismatches, CUDA errors, and MIG devices fail closed.

Each sampler logs its node, Slurm/CUDA variables, NVML indices/minors/UUIDs,
and chosen UUIDs (or mapping failure). The host needs `libcuda.so.1` from the
NVIDIA driver; no Torch dependency is added. The sampler step gets fresh
node-local GPU variables from Slurm, not the launch node's inherited IDs.
This trusts the site's Slurm GRES device-file configuration and device-cgroup
enforcement, as other Slurm GPU jobs do. The check runs on each compute node;
it is not an independent audit of the kernel's cgroup-v2 BPF device policy.
If the site cannot guarantee enforcement, do not use this mapping until an
administrator verifies the GRES device assignments and effective device policy.

Inside an **existing** GPU allocation, from the repository root, diagnose all
nodes without training (does not request a new allocation):

```bash
env -u CUDA_VISIBLE_DEVICES -u SLURM_JOB_GPUS -u SLURM_STEP_GPUS \
  srun --exact --export=ALL --kill-on-bad-exit=1 \
  --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 --cpus-per-task=1 --gpus-per-node=1 --cpu-bind=none --gpu-bind=none \
  "$PWD/.venv/bin/python" "$PWD/compute_power_multinode.py" --diagnose
```

Run from the batch/allocation shell, not a GPU-restricted training task. For the
full two-node smoke test, submit manually with
`sbatch -M gpu scripts/slurm_multinode_power_smoke.sh`.

The existing in-process `PowerMonitor` APIs and results tuple layouts remain
available, as do legacy `rank<N>.csv` sidecar filenames without `--ready-dir`.
Both monitors now share the NVML implementation, use allocation-aware selection,
and write `nan` instead of zero on read failures. CSV columns have been extended.
Legacy in-process monitors prefer step GPUs; this wrapper measures the whole
job allocation. Do not enable both monitoring paths for the same training run
unless duplicate sampling is intentional.

### Opt-in built-in InfiniBand experiment

```bash
SMOKE_NCCL_BUILTIN_IB=1 SMOKE_NCCL_SOCKET=0 \
  sbatch -M gpu scripts/slurm_multinode_power_smoke.sh
```

This automatically enables unlimited soft memlock in each training task and
checks the inherited limits inside Apptainer. It selects `NCCL_NET=IB`,
`NCCL_NET_PLUGIN=none`, and `NCCL_IB_DISABLE=0`. NCCL automatically discovers
HCAs: inherited `NCCL_IB_HCA` and `APPTAINERENV_NCCL_IB_HCA` are cleared,
and container-default HCA restrictions are removed before NCCL imports.
To explicitly restrict adapters, set `SMOKE_NCCL_IB_HCA='=mlx5_3:1'` (or another
NCCL HCA expression valid on both nodes). Only that opt-in override is applied. Settings are exported with matching
`APPTAINERENV_*` variables and reapplied by the diagnostic Python entrypoint
inside the container, before importing Torch/NCCL, so container defaults such
as `NCCL_NET_PLUGIN=spcx` cannot replace them. CollNet is disabled for isolation;
INFO INIT/NET logging is enabled. Other caller settings, including DMA-BUF,
remain unchanged.

Each rank logs its effective NCCL settings and writes NCCL output to
`ranks/nccl-rank<N>.log` in the run artifact directory. After training, each
rank requires `Using network IB` and a `NET/IB : Using [...]` adapter-selection
line in its actual NCCL log, prints the selected adapters, and rejects a logged
Spectrum-X/SPCX plugin load or use. Failure returns nonzero. The log check is
performed during the real run; mocked tests do not demonstrate hardware success.
Socket mode cannot be combined with this experiment. Default and Socket
launches otherwise remain unchanged. No jobs are submitted by the scripts.

### Opt-in unlimited memlock test (normal NCCL transport)

```bash
SMOKE_UNLIMITED_MEMLOCK=1 SMOKE_NCCL_SOCKET=0 \
  sbatch -M gpu scripts/slurm_multinode_power_smoke.sh
```

Each Slurm training task runs `ulimit -Sl unlimited` before launching Apptainer.
Failure stops the task immediately; Slurm's kill-on-bad-exit stops the step.
Inside each container, the diagnostic entrypoint logs hostname, rank, and soft
and hard locked-memory limits, verifies unlimited soft memlock, then runs the
unchanged training script. The hard limit must already permit unlimited;
this test does not raise the hard limit or change NCCL environment settings.
Do not combine with Socket mode (the launcher rejects that combination).
Leave `SMOKE_UNLIMITED_MEMLOCK` unset for the original behavior. Success means
both ranks log unlimited limits and training completes past the first broadcast
through the full smoke test. No hardware result is implied by mocked tests.

### Opt-in NCCL Socket diagnosis

```bash
SMOKE_NCCL_SOCKET=1 sbatch -M gpu scripts/slurm_multinode_power_smoke.sh
```

This forces `NCCL_NET=Socket`, `NCCL_NET_PLUGIN=none`, `NCCL_IB_DISABLE=1`,
`NCCL_COLLNET_ENABLE=0`, `NCCL_DMABUF_ENABLE=0`, and `NCCL_DEBUG=INFO`.
The interface is exactly `ib0` (`NCCL_SOCKET_IFNAME==ib0`), used successfully
for bootstrap on both nodes in job 4449428. This means TCP over IPoIB, not
verbs/RDMA; it does not bypass the physical InfiniBand fabric. Override with
`SMOKE_SOCKET_IFNAME=<common-interface>` only after verifying that interface
and its cross-node IP reachability. Every rank checks the interface is up,
non-loopback, and has an IPv4 address inside the container. These checks do
not prove end-to-end connectivity on a future allocation.

Slurm exports the settings with `--export=ALL`; matching `APPTAINERENV_*`
variables override container defaults. The diagnostic entrypoint verifies
settings before importing training, prints the effective `NCCL_*` environment,
interface/address and locked-memory soft/hard limits (bytes, equivalent to
`ulimit -l` information). It then runs the unchanged training workload.
No transport settings are forced when `SMOKE_NCCL_SOCKET` is unset; caller
settings remain intact. The previous unconditional DMA-BUF/debug exports
are now diagnostic-only.

Success requires `Using network Socket` on both nodes, successful broadcasts
and training steps for 90 seconds, and `MULTINODE POWER SMOKE PASSED`. Initial
communicator setup alone is not success. `Using network SPCX` or `NET/IB` data
transport indicates the isolation was not effective.

After a passing TCP run, test built-in IB separately with caller settings
`NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0 NCCL_DEBUG=INFO`
(without Socket mode; ensure no stale `APPTAINERENV_NCCL_*` overrides).
Keep DMA-BUF disabled first. If built-in IB fails too, compare locked-memory
limits, NVIDIA/OFED versions and peer-memory registration with CRC support.
If built-in IB passes, re-enable the Spectrum-X plugin separately, then test
DMA-BUF separately. Change one variable at a time. Retain per-node logs and
the first registration error/errno. Do not treat disabling RDMA as a production
performance fix or raise locked-memory limits without site guidance.

### Two-node A100 power smoke test (manual submission)

From the repository root, with the host power dependency installed and the
container/fork paths available on both nodes:

```bash
sbatch -M gpu scripts/slurm_multinode_power_smoke.sh
```

This requests account `cs2731_2026f`, partition `a100_nvlink`, two nodes with
one A100 GPU, 4 CPUs and 64 GB RAM **per node**, and a one-hour limit.
It does not submit another allocation. The training step runs two Slurm ranks,
one GPU per rank, using the existing container wrapper. Each task uses CUDA
index 0 after Slurm GPU binding/remapping. Megatron uses DP=2, TP=1, PP=1,
BF16 synthetic GPT training, gradient averaging, and AdamW updates. Rank zero
coordinates stopping after every rank has trained for at least 90 seconds
(excludes container/model startup). `SMOKE_DURATION` overrides this duration;
use 60–120 seconds for this workload.

Training does not require UUID properties from PyTorch. With one GPU bound per
Slurm task it selects CUDA device 0, regardless of Slurm local rank. If the task
sees the shared node GPU set, it selects `SLURM_LOCALID`. Completion records
include local rank, CUDA device index, and visible-device count; only the NVML
monitor tracks GPU UUIDs.

The smoke launcher sets `POWER_GPUS_PER_NODE` to its expected GPU count (1 by
default). The power wrapper requests `--gpus-per-node=1` explicitly on its
one-task-per-node step, with `--exact --gpu-bind=none` (no `--overlap`). Each sampler
must resolve one GPU before publishing readiness. The training step also
requests one GPU per node and one GPU per task with `--gpu-bind=single:1`.
These step requests cannot grant GPUs outside the parent Slurm allocation.
For other uniform jobs set `POWER_GPUS_PER_NODE=COUNT` (or `TYPE:COUNT`);
otherwise the wrapper uses `SLURM_GPUS_PER_NODE` when present. Without either,
it warns and retains Slurm's default step inheritance; asymmetric allocations
must not be forced into a uniform count.

The existing power wrapper starts one sampler per node at 0.5-second intervals
before training and cleans both up afterward. Only the training step uses
`--overlap` to share the sampler resources. See
[the GPU visibility investigation](docs/gpu_troubleshooting.md). The sampler
has the full one-GPU allocation per node; only training tasks request one GPU each. Rendezvous uses the first allocated node and a
job-derived port (override `MASTER_PORT` if needed). The repository and host
Python environment must be shared across nodes; `POWER_PYTHON` overrides the
default repository `.venv/bin/python`.

Inspect `slurm-power-smoke-<jobid>.out` for the `Smoke artifacts:` path and
`MULTINODE POWER SMOKE PASSED`. For example:

```bash
cat slurm-power-smoke-<jobid>.out
find power_logs/multinode_<jobid>_* -name '*.csv'
.venv/bin/python scripts/validate_multinode_power_smoke.py \
  power_logs/multinode_<jobid>_<suffix> --duration 90
```

The artifact directory contains `ranks/rank0.json` through `rank1.json` and
`job_<jobid>/run_<suffix>/<node>.csv` for each of the two nodes. The validator
checks both completion records and their Slurm local-rank/CUDA mappings,
one monitor UUID per node, positive finite per-GPU readings, correct single-GPU totals,
and a time series spanning the training duration. Missing power readings fail
this smoke test (the raw CSV still retains them). Training failures retain their
exit code; successful training with invalid artifacts fails validation.
No real multi-node run is implied by passing the mocked tests.

The launcher derives node count from `SLURM_JOB_NODELIST`, world size from
`SLURM_NTASKS`, and GPUs per node from tasks divided by nodes. It saves these
expectations in `layout.json` for offline validation. Future layouts must keep
one task per GPU and an equal task/GPU count on every node; update the Slurm
resource requests together. TP=1, PP=1, and one CUDA-visible GPU per task remain
intentional assumptions. CPU threads follow `SLURM_CPUS_PER_TASK` (default 4).

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

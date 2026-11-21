# Placement profiling sweeps: four and eight GPUs

This sweep reuses `train_profile.py`, the four-GPU Slurm wrapper, built-in NCCL
InfiniBand setup, UUID/OOB preflights, and power monitoring. It does not download
weights or change model definitions. No GPU job is launched by generation or
CPU validation.

## Four-GPU experiment design

All cases request **2 nodes × 2 A100 GPUs**, partition `a100`, constraint `amd`,
with block ranks `[0,1]` on the first host and `[2,3]` on the second. No host names
are fixed. Global batch is 16; warmup is 30–60 seconds, tolerance 0.02, window 10;
measurement is 60 seconds. Sequence lengths come from `MODEL_CONFIGS`.
Each allocation requests 64 GiB memory per node and four CPUs per task, matching
the existing wrapper. The sweep allows two hours rather than its one-hour
default to accommodate sequential placement startup and checkpoint overhead.

The existing `configs/profile_menu_4gpu_microbatch.json` supplies microbatch
candidates **1 and 2**. Both pass batch divisibility and model dimension checks
for every requested configuration. They are candidates, not a claim of GPU
memory fit. OOMs remain recorded failures; no automatic batch-size changes are
made because that would confound comparisons.

| Model | Approximate size label | Sequence length | Microbatches | Cases |
|---|---|---:|---|---:|
| gpt2 | 124M | 1024 | 1, 2 | 18 |
| gpt2_medium | 355M | 1024 | 1, 2 | 18 |
| gpt2_large | 774M | 1024 | 1, 2 | 18 |
| pythia_410m | 410M | 1024 | 1, 2 | 18 |
| pythia_1_4b | 1.4B | 1024 | 1, 2 | 18 |
| pythia_2_8b | 2.8B | 1024 | 1, 2 | 18 |
| llama_1b | 1B | 256 | 1, 2 | 18 |
| llama_2b | 2B | 256 | 1, 2 | 18 |
| llama_3b | 3B | 256 | 1, 2 | 18 |

Size labels describe the existing configurations, not an independently measured
parameter count or a claim that these are pretrained checkpoints.

### Unique placements

Each row below is crossed with all nine models and both microbatches: **162
profiling cases**, grouped into **108 array tasks**. Each task keeps the one or
two placements for a fixed model/TP/PP/DP/microbatch in the same allocation and
runs them sequentially with separate profiler invocations. Rank order is static
within a profiler invocation. Singleton groups, omitted below, are
`[[0],[1],[2],[3]]`.

| TP | PP | DP | Representative order | Non-singleton groups |
|---:|---:|---:|---|---|
| 4 | 1 | 1 | tp-dp-pp | TP: `[0,1,2,3]` |
| 1 | 4 | 1 | tp-dp-pp | PP: `[0,1,2,3]` |
| 1 | 1 | 4 | tp-dp-pp | DP: `[0,1,2,3]` |
| 2 | 2 | 1 | tp-dp-pp | TP: `[0,1] [2,3]`; PP: `[0,2] [1,3]` |
| 2 | 2 | 1 | dp-pp-tp | PP: `[0,1] [2,3]`; TP: `[0,2] [1,3]` |
| 2 | 1 | 2 | tp-dp-pp | TP: `[0,1] [2,3]`; DP: `[0,2] [1,3]` |
| 2 | 1 | 2 | dp-tp-pp | DP: `[0,1] [2,3]`; TP: `[0,2] [1,3]` |
| 1 | 2 | 2 | tp-dp-pp | DP: `[0,1] [2,3]`; PP: `[0,2] [1,3]` |
| 1 | 2 | 2 | tp-pp-dp | PP: `[0,1] [2,3]`; DP: `[0,2] [1,3]` |

All six orders are equivalent for the first three configurations. For the
remaining configurations, equivalent orders are exactly those that preserve
the relative order of the two non-singleton axes (three orders per placement).
The generated metadata lists every equivalent order and every group explicitly.
Deduplication compares labeled TP/DP/PP group memberships, not just cross-node
communication counts.

## Interpretation and limitations

Average power is the profiler's sum of per-rank selected-GPU mean NVML power,
not whole-node electrical power. Energy per step is `power_w * step_time_s`;
energy per token comes from the profiler. Raw power traces are retained per
attempt. Warmup nonconvergence is retained as a warning, not silently excluded.
Checkpoint save timing is retained; single-case invocations do not measure
cross-placement reshard/load overhead, so that quantity is unavailable rather
than a measured zero.

Same-allocation sequential placements reduce hardware differences but do not
eliminate order effects or run-to-run noise. This first sweep has one observation
per case; repeat under a separate output root to assess statistical significance.
CPU validation checks the fork's rank-generation logic. Real GPU process groups,
UUIDs and block host mapping are checked again against calibration before a run
can be marked complete. GPU memory fit and runtime IB behavior require execution.

### Equivalent rank orderings

| TP/PP/DP | Representative | All equivalent orders |
|---|---|---|
| 4/1/1 | `tp-dp-pp` | `tp-dp-pp`, `tp-pp-dp`, `dp-tp-pp`, `dp-pp-tp`, `pp-tp-dp`, `pp-dp-tp` |
| 1/4/1 | `tp-dp-pp` | `tp-dp-pp`, `tp-pp-dp`, `dp-tp-pp`, `dp-pp-tp`, `pp-tp-dp`, `pp-dp-tp` |
| 1/1/4 | `tp-dp-pp` | `tp-dp-pp`, `tp-pp-dp`, `dp-tp-pp`, `dp-pp-tp`, `pp-tp-dp`, `pp-dp-tp` |
| 2/2/1 | `tp-dp-pp` | `tp-dp-pp`, `tp-pp-dp`, `dp-tp-pp` |
| 2/2/1 | `dp-pp-tp` | `dp-pp-tp`, `pp-tp-dp`, `pp-dp-tp` |
| 2/1/2 | `tp-dp-pp` | `tp-dp-pp`, `tp-pp-dp`, `pp-tp-dp` |
| 2/1/2 | `dp-tp-pp` | `dp-tp-pp`, `dp-pp-tp`, `pp-dp-tp` |
| 1/2/2 | `tp-dp-pp` | `tp-dp-pp`, `dp-tp-pp`, `dp-pp-tp` |
| 1/2/2 | `tp-pp-dp` | `tp-pp-dp`, `pp-tp-dp`, `pp-dp-tp` |

## Local validation

The pre-change CPU suite passed **234 tests**. The final suite passed **291 tests**.
All 162 generated launcher commands also passed CPU dry-run argument parsing;
all 162 generated menus passed the actual fork checks. The placement reference check
executes the actual local Megatron `RankGenerator` and group helper definitions
in isolation, without importing torch/CUDA. All 36 candidates match, including
pipeline member order. The menu checks likewise execute the fork's real pure
menu loader, model-shape validator and microbatch validator. Source hashes are
saved in the validation report. This is source-level validation, not a claim
that distributed GPU initialization has already run.

The project CPU environment does not include torch. Do not install CUDA training
dependencies into it to run this sweep controller; GPU execution uses the
existing container. No original launcher or model definitions were modified.

## Commands

Run from the repository root, using the existing CPU environment:

```bash
# Generate all 162 case definitions and the full matrix CSV; no submission.
.venv/bin/python experiments/placement_sweep/sweep.py generate

# Validate against the actual local Megatron fork, without CUDA.
.venv/bin/python experiments/placement_sweep/validate.py \
  --manifest runs/placement_sweep/manifest.json \
  --megatron-fork /vast/cs2731_2026f/wat29/Megatron-LM \
  --output runs/placement_sweep/validation/fork-validation.json

# Preview every profiler command, then the pending Slurm submission (no execution).
.venv/bin/python experiments/placement_sweep/sweep.py dry-run

# Preview pending array tasks and submission command (does not submit).
.venv/bin/python experiments/placement_sweep/sweep.py submit --concurrency 2

# SUBMISSION ONLY WHEN APPROVED: at most two allocations / eight GPUs at once.
.venv/bin/python experiments/placement_sweep/sweep.py submit --concurrency 2 --execute

# Monitor Slurm and per-case completion.
squeue -M gpu -u "$USER"
.venv/bin/python experiments/placement_sweep/sweep.py status

# After failed/interrupted jobs have ended, retry only outstanding cases.
.venv/bin/python experiments/placement_sweep/sweep.py submit --concurrency 2 --execute

# One row per validated completed case; default output runs/placement_sweep/results.csv.
.venv/bin/python experiments/placement_sweep/sweep.py aggregate

# Run the complete CPU test suite.
.venv/bin/python -m pytest -q
```

`submit` defaults to dry-run; only `--execute` calls Slurm. Resume uses the same
command as initial submission. Completed cases are revalidated against their
calibration before being skipped. A partially completed two-placement task
reruns only its unfinished case. Failed and interrupted attempts are preserved;
retry creates a new numbered attempt directory. Task locks prevent two sweep
workers from executing the same task concurrently. Do not delete results to
resume, and let old jobs exit or cancel them before resubmitting.

An interrupted case can retain `running` status after a hard kill; it is still
pending for resume. Use Slurm to distinguish a live job from this stale status.
Generated manifests cannot be changed in place: use `generate --output-dir
runs/placement_sweep_repeat1` for a new experiment, then pass `--manifest
runs/placement_sweep_repeat1/manifest.json` to subsequent commands.

## Eight GPUs: two nodes × four A100s

The same sweep supports `--world-size 8`; its default output is
`runs/placement_sweep_8gpu/`. Four-GPU defaults, manifests, and results are
unchanged. Eight-GPU submission requests partition `a100`, constraint `amd`,
2 nodes, 8 tasks, 4 tasks/GPUs per node, one GPU per task, and 64 GiB host memory
per node. It reuses `profile_multinode.sbatch` with the existing IB, TCP/OOB,
memlock, UUID validation, block rank distribution, and power-monitoring path.

All ten factor triples with TP×PP×DP=8 and all six orders are considered.
Megatron-equivalent rank-group layouts are deduplicated (21 unique placements
before model feasibility checks). The nine models and microbatches 1/2 produce
**364 runnable cases in 166 array tasks**. `manifest.json` also records 14
excluded model/configuration/microbatch combinations with reasons:
TP=8 for GPT-2, GPT-2 Large, and Llama 2B; PP=8 for GPT-2, GPT-2 Large,
Llama 2B, and Llama 3B. Their head/query-group or layer counts do not divide.
No model dimensions are changed. Other failures, including OOMs, are recorded
as failed attempts with return codes and logs; no batch-size fallback occurs.

Each task sequentially runs the distinct physical placements for one
model/TP/PP/DP/microbatch setting in the same allocation. Equivalent orders are
aliases, not duplicate measurements. Concurrency 2 means **up to 16 GPUs**
(two independent eight-GPU allocations), not eight GPUs total.

Settings remain global batch 16, warmup 30–60 seconds, tolerance 0.02,
window 10, measurement 60 seconds, existing per-model sequence lengths, and
checkpoint-free profiling. CPU validation cannot establish GPU memory fit or
runtime performance. Keep this directory separate from the four-GPU results.

```bash
# Generate and validate, without GPU work.
.venv/bin/python experiments/placement_sweep/sweep.py generate --world-size 8
.venv/bin/python experiments/placement_sweep/validate.py --manifest runs/placement_sweep_8gpu/manifest.json --megatron-fork /vast/cs2731_2026f/wat29/Megatron-LM --output runs/placement_sweep_8gpu/validation/fork-validation.json
.venv/bin/python experiments/placement_sweep/sweep.py dry-run --manifest runs/placement_sweep_8gpu/manifest.json
.venv/bin/python experiments/placement_sweep/sweep.py submit --manifest runs/placement_sweep_8gpu/manifest.json --concurrency 2

# USER submission or resume (only after old jobs have ended):
.venv/bin/python experiments/placement_sweep/sweep.py submit --manifest runs/placement_sweep_8gpu/manifest.json --concurrency 2 --execute

# Monitor and aggregate completed cases.
squeue -M gpu -u "$USER"
.venv/bin/python experiments/placement_sweep/sweep.py status --manifest runs/placement_sweep_8gpu/manifest.json
.venv/bin/python experiments/placement_sweep/sweep.py aggregate --manifest runs/placement_sweep_8gpu/manifest.json
```

Resume revalidates completed calibrations and skips them; failed/interrupted
cases receive new attempt directories with identical settings. Task locks
prevent overlapping launchers. All outputs, power CSVs, metadata, mappings,
status files, and Slurm logs stay under the eight-GPU root. Automatic checkpoint
cleanup and the guarded `cleanup-checkpoints --inactive` command apply here too;
pass `--manifest runs/placement_sweep_8gpu/manifest.json` to target this experiment.

## Checkpoint-free placement profiling

The sweep passes `--profile-no-checkpoints` for its one-case menus. This skips
only checkpoint saves, not optimizer updates, warmup, or measurement. Model,
batch sizes, sequence lengths, timing windows, and power sampling are unchanged.
The general profiler defaults still retain checkpoints for resharding/switching;
use a separate experiment without this flag for those measurements. Multi-case
menus are rejected with the flag because they require checkpoint transitions.
The mounted Megatron fork must include support for this option.

After completed or failed launches, and before resuming terminal attempts, the
sweep removes only `.distcp` and `.pt` payloads below each attempt's
`work/profile_ckpt`. All metadata (including checkpoint metadata), calibration
JSONs, CSVs, logs, rank/UUID records, and status files remain. Resume validates
calibration, not checkpoints. Cleanup holds the same task lock as execution.

For a hard interruption, cleanup deliberately skips running/unknown attempts:
GPU writers could outlive the launcher. Once Slurm confirms all old jobs and
writers have ended, run this idempotent CPU-only command:

```bash
.venv/bin/python experiments/placement_sweep/sweep.py cleanup-checkpoints --inactive
```

Without `--inactive`, only completed/failed attempts are cleaned. Active task
locks are always skipped. Do not run `--inactive` while old GPU writers remain.
New checkpoint-free attempts cannot accumulate checkpoint payloads even if killed.

## Output layout

- `manifest.json` and `matrix.csv`: complete generated experiment definitions.
- `cases/CASE_ID/menu.json`, `metadata.json`, `status.json`: case inputs/latest state.
- `cases/CASE_ID/attempts/NNNN/`: calibration JSON, launcher log, status,
  metadata, power logs and isolated work/checkpoint directory.
- `slurm/`: allocation/array logs.
- `validation/`: local verification reports.
- `results.csv`: completed-case metrics, model metadata, equivalent orders,
  predicted groups and observed per-rank mappings (nested fields encoded as JSON).

Keep the raw calibration JSONs: CSV energy per step is derived from rounded
reported mean power and step time, while energy per token is the profiler's
reported value. No measurements are fabricated for failed or unfinished cases.

## Files added

Source files: `README.md`, `placements.py`, `sweep.py`, `validate.py`, and
`array.sbatch` in this directory. Tests: `tests/test_placement_groups.py`,
`tests/test_placement_sweep.py`, and `tests/test_placement_validation.py`.
The launcher and mounted Megatron profiler also expose the opt-in checkpoint-free
mode. Generated outputs live under the already git-ignored `runs/placement_sweep/`.

# Workshop-method reproduction with new eight-GPU measurements

## Scope and interpretation

Use the existing evaluator and its static, Greedy, Fixed Start Time Dynamic
Parallelism, Job Shifting, and Job Shifting + Dynamic Parallelism policies.
Do not change the scheduler or the measurement pipeline. This is an extension
of the workshop methodology, not an exact paper replication: Chinchilla token
budgets, annual traces, eight GPUs, new placement measurements, and oracle
quantization differ from some historical workshop experiments. A larger saving
cannot be attributed solely to NVLink/InfiniBand without a matched old-profile
control with identical budgets, deadlines, trace offsets and policy settings.

The historical regional fixed-start driver did not enable job shifting or RL;
its fixed GPT-2 budgets ranged from 3.5B to 20B tokens and its Llama-1B workload
used 3B tokens. See `docs/workshop_historical_method.md` for exact source paths
and settings. The current requested Chinchilla + job-shifting study preserves
policy implementations but changes the experiment design.

## Measurement selection

Select standard Megatron `tp-pp-dp` placement using `equivalent_orders`, not only
literal `order`. Singleton dimensions can make another representative order
exactly equivalent. Keep one successful measurement per model/TP/PP/DP/microbatch.
Do not include a different topology just to fill a missing configuration.
Power is summed NVML GPU power, not node/CPU/network/facility power. Measurement
runs are about 60 seconds, with one observation per case; 32 trace episodes do
not quantify profiling uncertainty. Retain warmup convergence warnings.

Use canonical TP=1/PP=1 logged parameter counts, without multiplying by DP.
TP layouts may contain slightly different padding or replicated parameters;
use a single fixed workload per model for equal-work comparisons. These model
names are custom configurations, not necessarily published checkpoint sizes.

## Audit outcome

There are 364 planned placements, 348 successful measurements and 16 failures.
The standard-order subset has 166 planned configurations, 157 successes and
nine failures (52 literal `tp-pp-dp` rows plus 105 equivalent-order rows).
Twenty selected measurements report unconverged warmup and remain flagged.
No duplicate configurations are retained. Detailed failures and validation are
in `docs/workshop_profile_audit.md` and the preparation outputs.

## Workload and scheduling assumptions

Training tokens = 20 times canonical parameters. Deadline = ceil(1.25 times
fastest measured static completion hours), including initial warmup. Sequence
length is 1024 for GPT-2/Pythia and 256 for Llama; global batch is 16.
Use 32 paired episodes, seed 0, eight simulated GPUs and hourly decisions.
Cap at 16 CPU workers, reducing workers for large DP arrays within a
32-CPU/64-GB allocation. The wrapper reserves a conservative 24-GiB budget for
estimated DP memory (24 bytes per time/token/strategy state) and leaves the
remaining memory for other arrays and Python. This is an estimate, not a peak
memory guarantee. Limit native math libraries to one thread per worker.
The simulator is CPU-only.

Use the historical workload-tier quantum with a per-model cap below the slowest
measured hourly token production. A blanket 50M quantum is unsafe: even the
fastest Llama-2B and Llama-3B strategies produce fewer than 50M tokens per hour,
so the DP can round every full-hour transition to zero progress. Exact replay
cannot recover all schedules lost to this discretization. Smaller bins are
still approximate and require sensitivity checks. Long job-shifting searches
remain extremely costly; reduce worker counts according to per-worker DP memory
rather than exceeding the 64-GB allocation. No automatic workload reduction or
strategy pruning is justified to make the search faster.

Keep a 12-hour window for the static Job Shifting baseline. Existing joint
shifting-and-switching searches the entire deadline, not this 12-hour window.
Use a common safe maximum trace offset across workloads to pair arrivals.

## Switching and warmup

Missing `reshard_wall_time_s` falls back to zero switching seconds, not a measured
zero. First-switch cache miss, container startup and fresh build default to zero.
The profile's `build_s` is not charged. Checkpoint profiling was disabled; save_s
is zero and there is no separate per-switch checkpoint save/load charge.
Idle power defaults to 50 W/GPU (400 W total), but zero-duration idle phases use
no energy. Delayed job start itself is not modeled as allocated idle GPU energy.

Each initial start and every strategy-index change incurs the target profile's
warmup. Warmup produces useful tokens (`warmup_iters * global_batch * seq_length`)
at measured average GPU power. It reduces hourly token progress relative to
steady throughput; it is not pure idle time. Microbatch-only strategy changes
also trigger warmup. These are profiling warmups, not measured reshard recovery.
Keep these assumptions unchanged and report savings conditional on them.

## Regional trace blocker found during preparation

The current three annual 2024 CSVs have 8,784 hourly readings each but are
byte-identical (SHA256 a4f7fb2c448b753788f401892eaf7dbd7fec524d50236e4360f0b7f47ef8fdd3).
All have intensity range 266--523 gCO2/kWh. Hourly validation alone does not verify
regional provenance. Fetch logs claim three different requested regions, but
response-region metadata was not retained. Resolve the API mapping/provenance
before presenting regional comparisons. Do not infer which region these values
represent or silently substitute synthetic data. Execution must block identical
regional series. No new API fetch or evaluation was performed for this audit.

## Reporting

Use same-episode best *deadline-feasible* static carbon as the reference. Compare
energy and completion delay with that same static row, not separate energy/time
minima. Report feasible/comparable episode counts, feasibility rates, absolute
carbon, GPU energy and completion delay including start wait. Never count an
unfinished or late run as carbon savings. Keep negative savings visible.
One feasible static configuration still defines a valid baseline but indicates
a limited trade-off menu (not proof that switching cannot help).

Do not compare new headline percentages directly with earlier GPT-2 output:
placement, workloads and sampling may differ. Preserve existing output roots.

## Updated trace recovery and original fixed-start baseline

Historical May traces were recovered byte-for-byte from Git into
`data/recovered_workshop_may2024`. They are three distinct 744-hour series;
see `docs/trace_provenance.md` for hashes and the limits of historical provenance.
No valid regional annual set was recovered. The existing identical annual files
remain unchanged and blocked. Do not extrapolate or tile May into annual data.

The new `prepare --study workshop-fixed-start` mode preserves the historical
six-workload grid: GPT-2 3.5/10/12/15/20B tokens with deadlines12/32/39/48/64h,
and Llama1B3B tokens with deadline108h. Original quanta and32 episodes are retained.
This is18 runs across three regions. No job shifting or RL is enabled in this mode.
Budgets and absolute deadlines remain fixed: their effective slack under the new
measurements is not necessarily25%. Hardware and profiles are deliberately new.
The shared arrival window is0--575h in May; all six workloads have feasible static
references. A ready audited snapshot and dry-run commands were produced at
`runs/workshop_fixed_start_may2024_ready`; no evaluation was launched.

```bash
# Optional fresh preparation: use a new output directory, never overwrite snapshots.
.venv/bin/python experiments/workshop_reproduction.py prepare --study workshop-fixed-start --out runs/workshop_fixed_start_may2024_fresh
# Inspect the already prepared snapshot without launching:
.venv/bin/python experiments/workshop_reproduction.py run --prepared runs/workshop_fixed_start_may2024_ready --out runs/workshop_fixed_start_may2024_results
# Only when ready, inside the allocation:
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
.venv/bin/python -u experiments/workshop_reproduction.py run --prepared runs/workshop_fixed_start_may2024_ready --out runs/workshop_fixed_start_may2024_results --execute
.venv/bin/python experiments/workshop_reproduction.py summarize --results runs/workshop_fixed_start_may2024_results --out runs/workshop_fixed_start_may2024_figures
```

The Chinchilla annual extension remains separate. Before running it, resolve
annual trace provenance and inspect runtime/memory estimates. Fixed-start removes
the expensive repeated-start outer loop but does not redesign the scheduler.

## Commands (annual Chinchilla extension, blocked pending correct traces)

An audit snapshot has been generated at `runs/workshop_tp_pp_dp_audit`.
It is blocked by identical regional traces. After correcting trace provenance,
prepare a NEW immutable snapshot (old files are never overwritten):

```bash
.venv/bin/python experiments/workshop_reproduction.py prepare --out runs/workshop_tp_pp_dp_ready
.venv/bin/python experiments/workshop_reproduction.py run --prepared runs/workshop_tp_pp_dp_ready --out runs/workshop_tp_pp_dp_results > runs/workshop_tp_pp_dp_ready/commands.txt
```

Review the audit warnings and commands. Inside the 32-CPU/64-GB allocation:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
.venv/bin/python -u experiments/workshop_reproduction.py run --prepared runs/workshop_tp_pp_dp_ready --out runs/workshop_tp_pp_dp_results --execute
```

For the original fixed-start baseline set, add `--no-job-shift` and use a new
results root. The default joint extension searches every hourly start before
the deadline and can take extremely long for large Chinchilla workloads.
No ETA is claimed. Do not run both experiment sets concurrently in one allocation.

Monitor model/region status (separate terminal):

```bash
watch -n 30 'find runs/workshop_tp_pp_dp_results -name status.json -exec cat {} \;'
ps -u "$USER" -o pid,ppid,etime,%cpu,rss,args --sort=-rss
```

Each started model/region has `evaluation.log`, PID and command in `status.json`.
The evaluator buffers episode progress until all episodes finish; a quiet log
is not evidence of a hung process. No per-episode progress hook was added.

Generate final paired tables and figures after completion:

```bash
.venv/bin/python experiments/workshop_reproduction.py summarize --results runs/workshop_tp_pp_dp_results --out runs/workshop_tp_pp_dp_figures
```

Outputs: paired_results.csv, paired_summary.csv, coverage.csv, policy-specific
model-by-region carbon/energy/completion-delay savings heatmaps, and explicitly
labeled secondary pooled boxplots. The per-run evaluator also generates its
usual aggregate.png and results/summary CSVs. Tables report means of per-episode
percentages and absolute policy/reference metrics on identical paired subsets.
Zero-reshard fraction distinguishes switching benefit from configuration selection.
Missing runs are not imputed; verify 27 completed status files before final claims.

Validation only (no experiments):

```bash
.venv/bin/python -m pytest tests/test_workshop_reproduction.py tests/test_carbon_sim.py -q
```


# All measured eight-GPU configurations

See the [zero-savings investigation](measured_models_zero_savings_audit.md) for
the corrected trace provenance, independent accounting and quantum sensitivity.

Run from the repository root inside an existing CPU allocation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_measured_models.py --workers 16 --out runs/measured_models
```

No GPUs, Slurm submission, RL dependencies, or model training are used. Add
`--dry-run` to validate and prepare without running the oracle. The worker cap
uses a 24-GiB total memory estimate (including coordinator reserve). Models are
launched concurrently while their combined estimates fit, rather than limiting
every model by the largest model. Memory estimates are conservative planning
estimates, not a measured peak-memory guarantee.

All nine menus are validated against 348 measured configurations. Equivalent
rank-order aliases remain metadata; distinct placement/microbatch tuples remain
separate actions. Duplicate tuples are rejected, not silently averaged. Canonical
parameter counts come from unpartitioned TP=PP=1 profiler logs, not rounded model
names. Default tokens are 20 times that count. The deadline is the ceiling of
1.25 times the fastest fixed completion time, using the same zero-overhead model.

Both policies start at trace offset zero. The oracle jointly selects TP, PP, DP,
microbatch and placement; it does **not** optimize delayed starts. This is the
specialized zero-overhead fixed-start dynamic-programming evaluator, with token
discretization, exact replay and comparison against static candidates—not a proof
of a continuous global optimum. May 2024 us-west-1 repeats after its 744th hour.
The default is the recovered workshop trace under `data/recovered_workshop_may2024/`,
mean 142.728495 gCO2/kWh; see [trace provenance](trace_provenance.md). The top-level
regional files are identical suspect artifacts and must not be treated as regional
measurements. The runner prints and records the trace hash and mean.

With no switching overhead, the previous action is not needed in the DP state.
The optimized implementation uses rolling token-progress costs and vectorized
transitions, retaining compact parents for schedule reconstruction. It does not
run the old expensive single-action replay repair. Consequently, nonaligned
token bins can produce different schedules from the legacy repaired oracle;
this is not a claim of bit-for-bit equivalence. All actions remain available.

The idealized zero-switch-cost experiment sets reshard time and warmup time/iters
to zero, as well as simulator startup/build/switch/cache costs. Measured power,
throughput and save time are retained. All 348 current measurements have zero
save time; the fast oracle rejects nonzero overheads rather than silently ignoring
them. The general oracle remains available for those experiments. Original
measurement files are unchanged.
The automatic token quantum follows the workshop rule, increased when necessary
to fit memory; workloads record the change. Coarser bins can suppress progress
for slow actions, so numerical sensitivity should be checked before publication.
An explicitly supplied over-budget quantum is rejected, not silently changed.
The optimized state layout fits all current defaults without coarsening, including
Llama-3B at its original 2M-token quantum.

Validation on this checkout: the full default Llama-1B model evaluation took
about 3 seconds; Llama-3B at the original 2M quantum took about 17 seconds;
the GPT-2 3B-token regression took under 1 second (excluding startup/validation).
These are observed timings, not runtime guarantees.

Each model directory contains its normalized profile, action mapping, workload
and parameter provenance, per-policy results, schedules with tuple/index/timing
and cumulative/interval emissions, a comparison plot and a summary. Root
`summary.csv` includes zero savings and infeasible runs. Savings are blank when
there is no feasible fixed baseline or the oracle is infeasible. Wrap counts
count boundaries crossed before the end of a simulation; finishing exactly at
a boundary does not consume the next repetition.

Previous GPT-2 workload regression:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/evaluate_measured_models.py --models gpt2 --tokens 3000000000 --deadline-hours 72 --trace data/carbon_2024_05_hourly_aws_us-west-1.csv --workers 1 --out runs/measured_models_suspect_trace_regression
```

This explicitly reproduces the **suspect top-level trace**, not verified west-1:
expected static and oracle emissions are 2503.432565 g (2503.433 rounded), 0% savings.
Do not use that number as a west-1 validation target. Omit `--trace` for the
corrected historical west-1 experiment; use a new output directory.
`--tokens`, `--deadline-hours`, and `--oracle-token-quantum` override defaults for
selected models. Use separate output directories for different experiments.

Completed models are reused by default only when their saved input/algorithm
fingerprint matches and required output artifacts exist. Infeasible results are
also valid completed evaluations. Use `--no-resume` to force recomputation. Old
results without a matching fingerprint are recomputed, not silently mixed with
the new algorithm. Keep old experiment directories for comparison.

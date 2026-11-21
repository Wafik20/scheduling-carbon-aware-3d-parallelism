# Carbon evaluation cost estimates

Run dry estimates (no oracle is run by default):

```sh
.venv/bin/python scripts/estimate_carbon_eval_cost.py
.venv/bin/python scripts/estimate_carbon_eval_cost.py --include-job-shift
.venv/bin/python scripts/estimate_carbon_eval_cost.py --benchmark
.venv/bin/python -m pytest -q tests/test_carbon_eval_cost.py
```

The default describes the fixed-start 27-run grid: nine models,
three regions, 32 episodes each. It reads the prepared workloads and does not
validate trace provenance. `--include-job-shift` estimates the fixed oracle plus T joint hourly candidate
starts with shrinking deadline-relative horizons (T+1 calls). Assumes one-hour decisions and no startup deduction.

## Memory and work

For Q=ceil(tokens/quantum), T=ceil(deadline hours), K=strategies:

* Main dense arrays: (T+1)(Q+1)K *14 bytes (float64 cost, int32 bin parent,
  int16 strategy parent). Density does not reduce allocated memory.
* Nine float64 T*K*K caches plus a K*K switch-duration cache are counted
  separately. Small vectors, trace copies, Python, imports and process overhead
  are not included; this is not peak RSS.
* The existing wrapper estimates 24 bytes/state and caps workers at16 under
  a24GiB budget, but forces at least one worker. The estimator flags workloads
  exceeding that budget even with one worker.
* Every oracle scans T(Q+1)K states even when almost none are reachable.
  Finite unfinished states additionally branch to K actions. The JSON reports
  1%,10%,50%,100% density scenarios, not a measured reachability fraction.
* Repair can add O(T²K) replay work to each oracle. Summing across all possible
  hourly starts gives O(T²QK²) DP and O(T³K) repair; the evaluator includes another fixed-start call, giving
  summed horizons T+T(T+1)/2. Memory is not multiplied by sequential starts.

Current prepared workload estimates (GiB, per oracle):

| Model | q million | Q | T | Dense GiB | 24-byte GiB | Workers | Scans million |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | 5 | 498 | 10 | 0.001 | 0.002 | 16 | 0.08 |
| gpt2_large | 36 | 431 | 168 | 0.015 | 0.026 | 16 | 1.16 |
| gpt2_medium | 25 | 284 | 51 | 0.004 | 0.006 | 16 | 0.28 |
| llama_1b | 7 | 4282 | 498 | 0.557 | 0.955 | 16 | 42.66 |
| llama_2b | 3 | 17224 | 1403 | 4.730 | 8.108 | 2 | 362.50 |
| llama_3b | 2 | 36068 | 2072 | 16.573 | 28.411 | 1 | 1270.49 |
| pythia_1_4b | 20 | 1415 | 232 | 0.082 | 0.140 | 16 | 6.24 |
| pythia_2_8b | 12 | 4626 | 734 | 0.754 | 1.292 | 16 | 57.74 |
| pythia_410m | 25 | 325 | 43 | 0.003 | 0.006 | 16 | 0.25 |

**Llama3B exceeds the wrapper's conservative24GiB budget even at one worker.**
Llama2B two workers allocate about9.46GiB of main arrays; Llama3B one worker
allocates16.57GiB. Other overhead still matters. Llama1B16 workers allocate
about8.91GiB main arrays; Pythia2.8B16 workers about12.06GiB.
These caps are memory heuristics, not runtime promises.

## Recovered workshop fixed-start grid

The ready `runs/workshop_fixed_start_may2024_ready/workloads.json` contains six
workloads across GPT-2 and Llama1B, not six distinct models. It uses three May
regions, hence18 runs. Estimate it without launching:

```sh
.venv/bin/python scripts/estimate_carbon_eval_cost.py --workloads runs/workshop_fixed_start_may2024_ready/workloads.json --benchmark
```

Main DP arrays range from1.95 to17.49MiB per worker (before caches/process overhead),
far below the Chinchilla Llama arrays. The same tiny-benchmark extrapolation gives
roughly0.67--0.81 CPU-hours for all18 runs, but this is NOT a prediction: profile
strategy counts, finite-state density, terminal integrations, repair, plotting,
and process overhead differ. Do not promise a three-minute batch based on ideal
16-worker division. These small fixed-start workloads are the sensible first
experiment set after user approval.

Dry estimate reports are saved under `runs/carbon_eval_cost_estimates`.
No full evaluation was run to obtain them.

## Bounded timing calibration

Only two tiny synthetic calls to the actual `rollout_oracle_dp` (including
repair) were measured in the project `.venv`. Total command wall time was
0.86seconds, including imports. Cases: T6,Q40,K3 took0.0066seconds;
T12,Q100,K6 took0.0780seconds. Both completed. No annual evaluation ran.
The benchmark has fixed small dimensions and is not an adjustable production
run; it does not impose a hard wall-clock kill on an overloaded host.

These correspond to8.9–10.7microseconds per dense scanned state **including
all other work**. They are not scan-kernel rates. Blindly applying that ratio
to Llama3B gives3.1–3.8hours per fixed-start oracle, or roughly300–363CPU-hours
for96 episodes. This is only an illustrative scaling alarm, not a duration
prediction: K grows from6 to17, reachable density differs, caches grow, repair
scales differently, and completion transitions perform extra integration.
Concurrent worker contention also prevents assuming linear speedup. For
Llama3B the density-dependent transition scenarios range from216million
(1%) to21.6billion (100%) per oracle, on top of1.27billion state scans.
With `--benchmark`, JSON includes this illustrative CPU-time range and an
ideal wall-time range obtained by dividing by worker count. These are not
confidence bounds or guaranteed speedups. For a hypothetical18-run/two-region Chinchilla
subset, pass `--regions 2`; this gives200–242CPU-hours for64 Llama3B episodes.
That subset is not the original workshop grid (six workloads across two models, by three regions).
Use profiling on separately authorized bounded realistic cases before
promising a full-grid completion time.

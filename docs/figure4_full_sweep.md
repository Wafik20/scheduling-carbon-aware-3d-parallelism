# Full workshop Figure 4 token sweep

Source: `6c27ee4:scripts/run_region_model_token_sweep_experiments.py` and
`6c27ee4:notebooks/visualize_regions.ipynb`. This is the 1B--8B sweep, not the
older six-workload fixed-grid driver. Two models, eight budgets, three regions:
48 runs, each64 episodes. Original seed0, hourly decisions, six-hour forecast
and Greedy lookahead, sim switch-cost mode, no job shifting or RL, no wrapping.

Deadlines are ceil(1.25 * tokens / fastest measured steady throughput /3600),
matching the original formula. Exact warmup-inclusive completion is checked
separately. With new tp-pp-dp profiles:

|Tokens B|GPT-2 deadline h|Llama1B deadline h|Quantum M|
|---:|---:|---:|---:|
|1|4|17|5|
|2|8|34|5|
|3|12|50|5|
|4|15|67|10|
|5|19|83|10|
|6|23|100|10|
|7|26|117|10|
|8|30|133|25|

Every workload has a feasible static reference. GPT-2 has5--7 feasible static
strategies; Llama1B has2. Global batch16, sequence length1024/256 respectively;
all use8GPUs. Hardware/profiles are new, not original paper hardware replication.
The shared May arrival cap is537.5h =744-(1.5*133+1+6), preserving paired offsets.

The original Llama quanta10M/25M exceed the slowest configurations' hourly
production (~7.05M). The quantized DP may credit zero progress to these slow
strategies. Preserve the original quantum for this reproduction, flag it in the
manifest, and do not claim continuous-token global optimality. Do not silently
replace it with a different quantum; any sensitivity study needs a separate root.

Original worker count4 is changed to16 for the32CPU HTC allocation; this changes
parallel execution, not scientific settings. Limit native math threads to1.
The original CI is mean +/-1.96 sample SE. Plot script also supports requested
percentile bootstrap; keep that explicitly labeled rather than claiming identity.

Existing six-workload GPT-2 budgets3.5/10/12/15/20B do not overlap1--8B. Llama3B
budget overlaps, but its32 episodes,108h deadline and575h offset cap differ from
this sweep's64 episodes,50h deadline and537.5h cap; those rows cannot be reused.
Other existing `carbon_8gpu_32workers` and `carbon_8gpu_default` GPT-2 3B
results have32 episodes rather than64 and are also not reused. Existing result
directories must not be modified. Scientific matches require
same workload, deadline, seed/episode offsets, profiles, traces and policies.

Plot after all runs complete, preserving the earlier sparse figure:

```bash
.venv/bin/python scripts/plot_workshop_figure4.py --results runs/figure4_full_sweep/results --out figures/workshop_figure4_full_sweep --full-sweep --ci-method normal
# Optional separately labeled bootstrap variant:
.venv/bin/python scripts/plot_workshop_figure4.py --results runs/figure4_full_sweep/results --out figures/workshop_figure4_full_sweep_bootstrap --full-sweep --ci-method bootstrap
```

No experiments were launched while preparing this workflow.

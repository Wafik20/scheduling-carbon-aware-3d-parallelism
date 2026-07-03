# Dynamic Parallelism Savings Metric

The primary fixed-start dynamic parallelism metric is now:

```text
savings_vs_episode_best_static =
  (episode_best_feasible_static_carbon_g - policy_carbon_g)
  / episode_best_feasible_static_carbon_g
```

The baseline is selected separately for each evaluated episode/window. For a
given dynamic/oracle row, we collect static schedules from the same run and the
same `(episode, offset_h)` key, filter them to exact-deadline-feasible rows, and
choose the lowest-carbon static schedule. Savings are not computed if that same
episode has no feasible static baseline.

The previous global baseline metric is retained only as:

```text
savings_vs_best_single_static_policy
```

That diagnostic first picks one static policy with the lowest feasible mean
carbon across the run, then compares dynamic/oracle rows against that single
policy. It is no longer the paper metric.

## Episode Key Assumptions

Each `results.csv` is scoped by its run directory and manifest row, which fixes:

- region
- model
- token budget
- deadline / max episode hours
- carbon trace path
- strategy profile
- GPU count and sequence length

Within that scoped run, `(episode, offset_h)` identifies the sampled carbon
trace window. The aggregation validates that an episode-best static baseline is
unique for each matched episode key and that every computed savings row has at
least one feasible static candidate from the same key.

## Llama 1B / 7B Existing Output Reinterpretation

These values are recomputed from the existing
`runs/region_may2024_model_token_sweep_25slack_faithful_workers` outputs. The
old column is the previous best-single-static-policy ratio-of-means metric. The
new primary column is the mean per-episode savings against the episode-best
feasible static baseline.

| Region | Feasible DP runs | Feasible static candidates per episode | Old `savings_vs_best_single_static_policy` | New `savings_vs_episode_best_static` | New savings, reshards > 0 only | Reshards = 0 fraction | Dropped: no feasible static baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| us-west-1 | 43 | 3.0 | 2.417% | 1.891% | 3.128% | 39.5% | 0 |
| us-west-2 | 24 | 3.0 | 8.231% | 1.671% | 2.864% | 41.7% | 0 |

Under the corrected primary metric, the Llama 1B / 7B result no longer claims a
larger fixed-start dynamic-parallelism gain for `us-west-2` from the old static
baseline selection. The true dynamic-switching subset is also reported
separately from `reshards = 0` cases.


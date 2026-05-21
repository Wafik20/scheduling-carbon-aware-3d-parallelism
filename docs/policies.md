# Policies and Terminology

This document defines the scheduling policies used in the paper-facing
documentation. Some code paths still use older names such as `smart-*`; those
are noted below.

## Configuration

A parallel configuration is written as:

```text
(TP, PP, DP)
```

where:

- `TP` is tensor parallelism.
- `PP` is pipeline parallelism.
- `DP` is data parallelism.

The configuration must satisfy:

```text
TP * PP * DP = world size
```

For example, with 4 GPUs, valid configurations are `(4,1,1)`, `(1,4,1)`,
`(1,1,4)`, `(2,2,1)`, `(2,1,2)`, and `(1,2,2)`.

## Best Static

**Best Static** is the lowest-carbon static configuration that completes the
job before the deadline.

Static means the job chooses one `(TP, PP, DP)` configuration at launch and
never switches. Best Static is an important baseline because dynamic
parallelism should be compared against the best feasible static choice, not a
random fixed configuration.

In the code, static rollouts are produced by `rollout_static(...)` in
`src/carbon_aware_3d/carbon_aware_env.py`, and
`evaluate_carbon_policy.py` reports the best completed static rollout for an
episode.

## Greedy

**Greedy** is an online policy. In the paper terminology, it uses the current
hour's carbon intensity rather than the future trace.

Current code note: `evaluate_carbon_policy.py` labels this policy `Greedy`,
but several CLI arguments use older `smart-*` names, for example:

- `--smart-fast-idx`
- `--smart-candidates`
- `--smart-lookahead-h`
- `--smart-margin-g`

Those names come from an earlier "smart-switch" implementation. When
documenting paper results, use **Greedy** unless you are describing the exact
script arguments.

## Fixed-Start Dynamic Parallelism

**Fixed-Start Dynamic Parallelism** starts the job immediately at its arrival
time, but the scheduler can switch among `(TP, PP, DP)` configurations during
the run.

The simulator accounts for:

- power draw of the active configuration
- throughput of the active configuration
- warmup and save overheads
- switching or resharding time
- carbon intensity at the time energy is consumed

In the code, this is represented by the offline dynamic-programming oracle
`rollout_oracle_dp(...)`, and `evaluate_carbon_policy.py` reports it as
`Fixed Start Time Dynamic Parallelism`.

## Job Shifting

**Job Shifting** delays the start time within an allowed window, then runs a
chosen configuration. It captures the carbon benefit available from temporal
flexibility alone.

The current simulator includes a static-fast job-shifting helper:
`rollout_job_shifting_static_fast(...)`.

## Job Shifting + Dynamic Parallelism

**Job Shifting + Dynamic Parallelism** jointly chooses:

- when to start the job
- which `(TP, PP, DP)` configuration to use over time

This policy has more flexibility than fixed-start dynamic parallelism because
it can both wait for cleaner hours and adapt configuration during the run.

In the code, this is represented by `rollout_oracle_shift_switch(...)`, and
`evaluate_carbon_policy.py` reports it as `Job Shifting + Dynamic Parallelism`.

## Optional RL Policies

`train_carbon_policy.py` contains optional PPO/DQN policy training. These
experiments appear to be exploratory in the current repository and are not the
default paper-facing evaluation path.


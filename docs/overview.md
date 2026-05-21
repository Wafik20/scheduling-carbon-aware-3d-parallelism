# Project Overview

This repository supports the paper **"Carbon-Aware Dynamic Parallelism for
Distributed Training."** The paper studies whether large language model
pretraining can reduce carbon emissions by changing its distributed
parallelism configuration over time.

Most distributed training jobs choose a single 3D parallelism configuration at
launch. This project instead treats `(TP, PP, DP)` as a scheduling control:

- Use lower-power configurations when carbon intensity is high.
- Use higher-throughput configurations when carbon intensity is lower.
- Account for the cost of switching configurations.
- Still complete the token budget before the deadline.

## Two-Part System

The codebase has two main parts.

## Profiler

The profiler measures each feasible parallel configuration on the target
model and hardware. It runs short Megatron Core / Megatron-LM jobs and records
power, throughput, energy, and switching costs. These measurements are stored
as calibration tables that drive the simulator.

This matters because the best configuration is not universal. Power and
throughput depend on the model, hardware, and communication topology.

## Scheduler / Simulator

The scheduler combines a calibration table with:

- token budget
- deadline
- hourly carbon-intensity trace
- switching cost

It evaluates policies that choose a configuration over time. The simulator is
used for no-GPU scheduling experiments and policy comparisons.

## 3D Parallelism

The project uses:

- `TP`: tensor parallelism
- `PP`: pipeline parallelism
- `DP`: data parallelism

Configurations are written as `(TP, PP, DP)` and must satisfy:

```text
TP * PP * DP = world size
```

For a 4-GPU world size, the feasible factorizations are:

```text
(4, 1, 1)
(1, 4, 1)
(1, 1, 4)
(2, 2, 1)
(2, 1, 2)
(1, 2, 2)
```

## Paper Findings To Keep In Mind

- Power and throughput depend on model, hardware, and parallel configuration.
- The scheduler should not assume a fixed best configuration across all
  deployments.
- On the paper's single 4-GPU L40 setup, `TP=4` had the lowest measured
  average power, but also lower throughput.
- Multi-node rankings may differ because cross-node communication changes
  both power and throughput.
- The paper evaluates GPT-2, Llama-style, and Pythia models.
- The training stack uses Megatron Core / Megatron-LM.
- Carbon traces are from Electricity Maps for AWS `us-west-1` in 2024.
- The paper reports fixed-start dynamic parallelism reductions over Best
  Static of up to about 3.4%, and one Greedy episode saving 4.7%.

## Documentation Map

- `docs/profiler.md`: how profiling is represented in this repo.
- `docs/scheduler.md`: scheduler/simulator inputs, outputs, and commands.
- `docs/policies.md`: policy definitions and naming notes.
- `docs/repository_structure.md`: file and directory map.
- `docs/reproducing_experiments.md`: practical command examples.
- `docs/experiment_outputs.md`: where generated artifacts live.


# Scheduling Carbon-Aware 3D Parallelism:
This repository contains code for the paper "Scheduling Carbon-Aware 3D Parallelism."

Main evaluation baselines:

- Static: choose one DP/TP/PP strategy and never switch.
- Smart-switch: interpretable online heuristic using limited carbon lookahead.
- Offline DP oracle: perfect-foresight dynamic-programming upper bound, reported after exact simulator replay.

PPO/DQN scheduling experiments are currently experimental and deferred from the main paper evaluation pipeline.

For each model size, we filter the TP/PP/DP strategy set to configurations
that satisfy the memory constraints of the target hardware. If only one
feasible strategy remains, we treat that setting as a static-only feasibility
case and exclude it from dynamic-scheduling savings comparisons.

Default evaluation:

```bash
python evaluate_carbon_policy.py \
  --trace ./data/carbon_year.csv \
  --tokens 4000000000 \
  --max-episode-hours 48 \
  --forecast-horizon-h 6 \
  --smart-fast-idx -1 \
  --smart-lookahead-h 16 \
  --smart-candidates 2,5 \
  --smart-margin-g 10 \
  --oracle-token-quantum 1000000 \
  --n-episodes 32 \
  --seed 123 \
  --out runs/eval_static_smart_oracle
```

Optional RL baseline:

```bash
python evaluate_carbon_policy.py \
  --trace ./data/carbon_year.csv \
  --include-rl \
  --model runs/ppo_chunks_12h_seed0_retry/model.zip \
  --algo dqn \
  --tokens 4000000000 \
  --max-episode-hours 48 \
  --forecast-horizon-h 6 \
  --smart-fast-idx -1 \
  --smart-lookahead-h 16 \
  --smart-candidates 2,5 \
  --smart-margin-g 10 \
  --oracle-token-quantum 1000000 \
  --n-episodes 32 \
  --seed 123 \
  --out runs/eval_with_rl_optional
```

# Reproducing Experiments

This project has two layers of experiments:

1. No-GPU simulator evaluations using measured strategy profiles and carbon
   traces.
2. GPU measurement/profiling jobs that collect or refresh those strategy
   profiles.

The simulator layer is the easiest place to start.

## 1. Simulator Smoke Test

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_week.csv \
  --tokens 10000000 \
  --max-episode-hours 2 \
  --forecast-horizon-h 2 \
  --oracle-token-quantum 1000000 \
  --n-episodes 2 \
  --n-detail-rollouts 1 \
  --seed 1 \
  --out runs/smoke_eval
```

Expected output: `runs/smoke_eval/results.csv` plus aggregate and rollout
plots. This should not launch GPU training.

## 2. Static, Greedy, Oracle, and Job Shifting

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 4000000000 \
  --max-episode-hours 48 \
  --forecast-horizon-h 6 \
  --smart-fast-idx -1 \
  --smart-lookahead-h 16 \
  --smart-candidates 2,5 \
  --smart-margin-g 10 \
  --include-job-shift \
  --shift-window-h 24 \
  --shift-step-h 1 \
  --oracle-token-quantum 1000000 \
  --n-episodes 32 \
  --seed 123 \
  --out runs/eval_static_smart_oracle
```

This compares:

- Static strategies.
- Greedy / smart-switch.
- Fixed Start-Time dynamic parallelism.
- Job Shifting.
- Job Shifting + Dynamic Parallelism.

Use `--n-episodes` to control runtime. The offline oracle can become expensive
when the token budget is large and `--oracle-token-quantum` is small.

## 3. Feasibility Filtering

To evaluate model sizes where some DP/TP/PP choices do not fit memory:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 4000000000 \
  --model-name llama_3b \
  --model-num-params 3000000000 \
  --filter-strategies \
  --gpu-memory-gb 48 \
  --usable-memory-frac 0.90 \
  --include-job-shift \
  --n-episodes 32 \
  --out runs/llama_3b_feasible_eval
```

If only one strategy remains, the script treats that case as static-only
instead of reporting dynamic-scheduling savings.

## 4. Optional RL Baseline

RL is optional and requires:

```bash
python -m pip install -e ".[rl]"
```

Train:

```bash
python train_carbon_policy.py \
  --trace data/carbon_year.csv \
  --tokens 3000000000 \
  --total-timesteps 200000 \
  --algo ppo \
  --out runs/ppo_carbon
```

Evaluate a saved policy:

```bash
python evaluate_carbon_policy.py \
  --trace data/carbon_year.csv \
  --include-rl \
  --model runs/ppo_carbon/model.zip \
  --algo ppo \
  --tokens 3000000000 \
  --n-episodes 32 \
  --out runs/eval_with_rl
```

## 5. GPU Measurement Jobs

The live GPU scripts require Slurm, Apptainer, Megatron-LM, and matching CUDA
software. The default paths reflect the original development cluster; override
them for a new system.

Single static/scheduled launcher:

```bash
python train.py \
  --model gpt2_medium \
  --tp 1 \
  --pp 1 \
  --num-gpus 4 \
  --container-image /path/to/nemo.sif \
  --container-bind /path/to/Megatron-LM:/opt/megatron-lm \
  --megatron-dir /opt/megatron-lm \
  --work-dir runs/manual_static_test \
  --dry-run
```

Online elastic launcher:

```bash
python train_online_elastic.py \
  --model gpt2_medium \
  --scheduler-config calibrations/calibration_gpt2_medium_4gpu.json \
  --carbon-forecast data/carbon_synthetic_dataset.csv \
  --train-iters 1000 \
  --switch-power-w 200 \
  --deadline-seconds 3600 \
  --lookahead-steps 300 \
  --decide-every 100 \
  --num-gpus 4 \
  --work-dir runs/manual_online_test \
  --dry-run
```

Slurm sweep scripts live in `batch_files/`. The multi-node submit helper is:

```bash
SMOKE_TEST=1 scripts/submit_gpt2m_node_sweeps.sh
```

Review the paths and account/partition settings before submitting.

## 6. Profiling and Benchmarks

Layer timing:

```bash
python scripts/profile_per_layer.py --output-dir results/layer_profiles
```

Reshard benchmark:

```bash
python scripts/benchmarks/benchmark_transform.py
```

The reshard benchmark writes `results/resharding/reshard_times.csv`.


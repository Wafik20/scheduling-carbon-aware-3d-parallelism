#!/bin/bash
#SBATCH --job-name=naive-hybrid
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Naive scheduler run using the hybrid-only batch shape.
#
# Strategies are defined in this sbatch and passed into train_naive_elastic.py:
#   low:  TP=1 PP=1 DP=4   (fastest, highest power — when carbon is cheap)
#   med:  TP=2 PP=1 DP=2   (middle ground)
#   high: TP=4 PP=1 DP=1   (slowest, lowest power — when carbon is expensive)
#
# Carbon-trace mapping is now progress-based: the full CSV is replayed
# exactly once across `train_iters` steps, regardless of actual wall time.
# `min_dwell_steps` smooths the CI signal so a single-row spike or dip does
# not trigger resharding (which would cost more than it saves).
#
# Override knobs via the environment, no need to edit the file:
#   TRAIN_ITERS=750 DECIDE_EVERY=50 MIN_DWELL_STEPS=200 sbatch run_hybrid.sbatch
#
set -euo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
module purge
module load apptainer

cd "$PROJECT"
mkdir -p slurm_logs runs

CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON=/bgfs/stephenlee/wat29/Megatron-LM
COMMON_BINDS=(
  --container-image "$CONTAINER"
  --container-bind  "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29"
  --container-bind  "$MEGATRON:/opt/megatron-lm"
)

# ── Knobs (override via env) ─────────────────────────────────
MODEL=${MODEL:-gpt2_medium}
TRAIN_ITERS=${TRAIN_ITERS:-750}
DECIDE_EVERY=${DECIDE_EVERY:-50}

# Smoothing window: CI is averaged over [step, step+MIN_DWELL_STEPS]
# before classifying the bucket. Rule of thumb: ~2× reshard_cost / step_time.
# For the short hybrid run (reshard≈35s, step time 0.2–0.8s), 200 is a
# reasonable starting point. Set to 0 to disable smoothing entirely.
MIN_DWELL_STEPS=${MIN_DWELL_STEPS:-200}

LOW_TP=${LOW_TP:-1}
LOW_PP=${LOW_PP:-1}
LOW_DP=${LOW_DP:-4}
MED_TP=${MED_TP:-2}
MED_PP=${MED_PP:-1}
MED_DP=${MED_DP:-2}
HIGH_TP=${HIGH_TP:-4}
HIGH_PP=${HIGH_PP:-1}
HIGH_DP=${HIGH_DP:-1}

CARBON_CSV="$PROJECT/data/carbon_week.csv"
POWER_DIR="$PROJECT/power_logs/$MODEL"

[[ -f "$CARBON_CSV" ]] || { echo "[fatal] missing $CARBON_CSV"; exit 2; }
[[ -f "$CONTAINER"  ]] || { echo "[fatal] missing $CONTAINER"; exit 2; }
[[ -d "$MEGATRON"   ]] || { echo "[fatal] missing $MEGATRON"; exit 2; }

echo "[sweep] node=$(hostname)  start=$(date)"
echo "[sweep] PROJECT=$PROJECT"
echo "[sweep] MODEL=$MODEL  TRAIN_ITERS=$TRAIN_ITERS"
echo "[sweep] strategies: low=tp${LOW_TP}_pp${LOW_PP}_dp${LOW_DP}  med=tp${MED_TP}_pp${MED_PP}_dp${MED_DP}  high=tp${HIGH_TP}_pp${HIGH_PP}_dp${HIGH_DP}"
echo "[sweep] CARBON_CSV=$CARBON_CSV"
echo "[sweep] DECIDE_EVERY=$DECIDE_EVERY  MIN_DWELL_STEPS=$MIN_DWELL_STEPS"
echo "[sweep] mapping: progress-based (CSV stretched across $TRAIN_ITERS steps)"
echo

# ── NAIVE HYBRID PHASE ───────────────────────────────────────
RUN_DIR_HY="$PROJECT/runs/naive_hybrid_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR_HY"
echo "[sweep] === NAIVE HYBRID  start=$(date +%H:%M:%S)  RUN_DIR=$RUN_DIR_HY ==="

python train_naive_elastic.py \
  --model           "$MODEL" \
  --carbon-forecast "$CARBON_CSV" \
  --train-iters     "$TRAIN_ITERS" \
  --decide-every    "$DECIDE_EVERY" \
  --min-dwell-steps "$MIN_DWELL_STEPS" \
  --low-tp          "$LOW_TP" \
  --low-pp          "$LOW_PP" \
  --low-dp          "$LOW_DP" \
  --med-tp          "$MED_TP" \
  --med-pp          "$MED_PP" \
  --med-dp          "$MED_DP" \
  --high-tp         "$HIGH_TP" \
  --high-pp         "$HIGH_PP" \
  --high-dp         "$HIGH_DP" \
  --num-gpus        4 \
  --work-dir        "$RUN_DIR_HY" \
  "${COMMON_BINDS[@]}"

echo
echo "[sweep] done=$(date)"
echo "[sweep] phases:            $RUN_DIR_HY/phases.json"
echo "[sweep] inner summary:     $RUN_DIR_HY/elastic_results.json"
echo "[sweep] power dir:         $POWER_DIR"
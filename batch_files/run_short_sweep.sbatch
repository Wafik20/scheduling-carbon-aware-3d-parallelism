#!/bin/bash
#SBATCH --job-name=short-sweep
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Short sweep:
#   * 6 static parallelism strategies (one phase each), back-to-back
#   * 1 online (hybrid) run with the same calibration + the same CI CSV
#
# All 7 runs see the identical wall-clock CI trace, so the post-hoc plot
# can score them apples-to-apples. Total wall-time budget ≈ 40 min for
# gpt2_medium @ 1500 iters; tighten by lowering TRAIN_ITERS if needed.

set -euo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
module purge
module load apptainer

cd "$PROJECT"
mkdir -p slurm_logs runs schedules/short

CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON=/bgfs/stephenlee/wat29/Megatron-LM
COMMON_BINDS=(
  --container-image "$CONTAINER"
  --container-bind  "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29"
  --container-bind  "$MEGATRON:/opt/megatron-lm"
)

# ── Experiment knobs (edit these to retune) ───────────────────────
MODEL=gpt2_medium
TRAIN_ITERS=750             # ~60s for DP=4, ~370s for TP=2/PP=2
DEADLINE_S=400              # excludes TP=4 (would need ~630s) — by design
BUCKET_SEC=60               # carbon_short.csv is one row per minute
SWITCH_POWER_W=200          # GPU power during reshard
LOOKAHEAD_STEPS=300         # ≥ 1 CI bucket of work for any strategy
DECIDE_EVERY=100            # ~6 decisions per static-equivalent run

CARBON_CSV="$PROJECT/data/carbon_short.csv"
SCHED_DIR="$PROJECT/schedules/short"
SCHED_CFG="$PROJECT/calibrations/calibration_gpt2_medium_4gpu.json"

if [[ ! -f "$CARBON_CSV" ]]; then
  echo "[fatal] carbon CSV not found: $CARBON_CSV" >&2
  exit 1
fi
if [[ ! -f "$SCHED_CFG" ]]; then
  echo "[fatal] scheduler config not found: $SCHED_CFG" >&2
  exit 1
fi

echo "[sweep] node=$(hostname)  start=$(date)"
echo "[sweep] PROJECT=$PROJECT"
echo "[sweep] MODEL=$MODEL  TRAIN_ITERS=$TRAIN_ITERS  DEADLINE_S=$DEADLINE_S"
echo "[sweep] CARBON_CSV=$CARBON_CSV (bucket=${BUCKET_SEC}s)"
echo

# ── Static baselines (single-phase schedules) ──────────────
# Order: fastest → slowest, so faster ones complete early and a tail
# failure on a slow run doesn't block the others.
STRATEGIES=(
  "1 1 4"
  "1 2 2"
  "1 4 1"
  "2 1 2"
  "2 2 1"
  "4 1 1"
)

echo "[sweep] generating ${#STRATEGIES[@]} static schedule JSONs..."
for s in "${STRATEGIES[@]}"; do
  read -r tp pp dp <<< "$s"
  python "$PROJECT/make_static_schedule.py" \
    --tp "$tp" --pp "$pp" --dp "$dp" \
    --iters "$TRAIN_ITERS" \
    --out-dir "$SCHED_DIR"
done
echo

# ── STATIC PHASE ───────────────────────────────────────────
echo "[sweep] === STATIC PHASE ==="
for s in "${STRATEGIES[@]}"; do
  read -r tp pp dp <<< "$s"
  TAG="tp${tp}_pp${pp}_dp${dp}"
  echo
  echo "[sweep] static $TAG  start=$(date +%H:%M:%S)"
  RUN_DIR="$PROJECT/runs/short_${SLURM_JOB_ID}/static_$TAG"
  mkdir -p "$RUN_DIR"

  # `|| true` so one failed config doesn't kill the rest of the sweep.
  python train_elastic.py \
    --model "$MODEL" \
    --schedule "$SCHED_DIR/${TAG}.json" \
    --num-gpus 4 \
    --work-dir "$RUN_DIR" \
    "${COMMON_BINDS[@]}" \
    || echo "[sweep] WARN: static $TAG exited nonzero — continuing"
done

# ── HYBRID PHASE ───────────────────────────────────────────
echo
echo "[sweep] === HYBRID PHASE  start=$(date +%H:%M:%S) ==="
RUN_DIR_HY="$PROJECT/runs/short_${SLURM_JOB_ID}/hybrid"
mkdir -p "$RUN_DIR_HY"
python train_online_elastic.py \
  --model              "$MODEL" \
  --scheduler-config   "$SCHED_CFG" \
  --carbon-forecast    "$CARBON_CSV" \
  --train-iters        "$TRAIN_ITERS" \
  --switch-power-w     "$SWITCH_POWER_W" \
  --deadline-seconds   "$DEADLINE_S" \
  --lookahead-steps    "$LOOKAHEAD_STEPS" \
  --decide-every       "$DECIDE_EVERY" \
  --bucket-seconds     "$BUCKET_SEC" \
  --num-gpus           4 \
  --work-dir           "$RUN_DIR_HY" \
  "${COMMON_BINDS[@]}" \
  || echo "[sweep] WARN: hybrid exited nonzero"

echo
echo "[sweep] done=$(date)"
echo "[sweep] static logs:  $PROJECT/power_logs/$MODEL/elastic_*_power.csv"
echo "[sweep] hybrid log:   $PROJECT/power_logs/$MODEL/online_elastic_calibration_${MODEL}_4gpu_power.csv"
echo "[sweep] next:         python visualize_carbon_short.py"
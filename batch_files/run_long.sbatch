#!/bin/bash
#SBATCH --job-name=long-run
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# 20k-iter long-run carbon-aware experiment: 2 statics + hybrid.
# Training time budget: ~70 min DP=4, ~150-168 min TP=2,DP=2, ~95 min hybrid.
# Reshard overhead drops to <1% of hybrid wall — production-realistic.
#
# Override via env: TRAIN_ITERS=10000 sbatch run_long.sbatch
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
MODEL=${MODEL:-llama_2b}
TRAIN_ITERS=${TRAIN_ITERS:-20000}
DEADLINE_S=${DEADLINE_S:-11000}        # 183 min: TP=2,DP=2 fits even if 12% slow
BUCKET_SEC=${BUCKET_SEC:-60}
SWITCH_POWER_W=${SWITCH_POWER_W:-200}
SWITCH_TIME_S=${SWITCH_TIME_S:-50}
LOOKAHEAD_STEPS=${LOOKAHEAD_STEPS:-15000}   # spans HIGH→LOW transition at any strategy
DECIDE_EVERY=${DECIDE_EVERY:-500}           # ~40 decisions over full run, no flapping

CARBON_CSV="$PROJECT/data/carbon_synthetic_dataset.csv"
SCHED_CFG="$PROJECT/calibrations/calibration_gpt2_medium_4gpu.json"
POWER_DIR="$PROJECT/power_logs/$MODEL"
mkdir -p "$POWER_DIR"

[[ -f "$CARBON_CSV" ]] || { echo "[fatal] missing $CARBON_CSV"; exit 2; }
[[ -f "$SCHED_CFG"  ]] || { echo "[fatal] missing $SCHED_CFG"; exit 2; }

# ── Cleanup stale static CSVs ────────────────────────────────
echo "[long] cleaning stale statics so the plot stays apples-to-apples"
for tag in tp1_pp2_dp2 tp1_pp4_dp1 tp2_pp2_dp1 tp4_pp1_dp1; do
  rm -fv "$POWER_DIR/elastic_${tag}_power.csv" || true
done
echo

echo "[long] node=$(hostname)  start=$(date)"
echo "[long] MODEL=$MODEL  TRAIN_ITERS=$TRAIN_ITERS  DEADLINE_S=${DEADLINE_S}s ($(( DEADLINE_S/60 )) min)"
echo "[long] CARBON_CSV=$CARBON_CSV (bucket=${BUCKET_SEC}s, "
echo "       horizon=$(awk -F, 'NR>1{n++}END{print n*60}' "$CARBON_CSV")s = $(( $(awk -F, 'NR>1{n++}END{print n}' "$CARBON_CSV") )) min)"
echo "[long] LOOKAHEAD=$LOOKAHEAD_STEPS  DECIDE_EVERY=$DECIDE_EVERY  SWITCH_TIME_S=$SWITCH_TIME_S"
echo

# ── Two static phases (in size order — fastest first) ────────
STATIC_TAGS=(
  "tp1_pp1_dp4"      # DP=4   — fastest, energy-per-iter winner (~70 min @ 20k iters)
  "tp1_pp2_dp2"      # TP=2,DP=2 — best static under varying CI (~150-168 min)
)

for tag in "${STATIC_TAGS[@]}"; do
  RUN_DIR="$PROJECT/runs/long_${SLURM_JOB_ID}/static_${tag}"
  mkdir -p "$RUN_DIR"

  TP=$(echo "$tag" | sed -n 's/^tp\([0-9]*\).*/\1/p')
  PP=$(echo "$tag" | sed -n 's/^tp[0-9]*_pp\([0-9]*\).*/\1/p')
  DP=$(echo "$tag" | sed -n 's/.*_dp\([0-9]*\)$/\1/p')

  echo "[long] === STATIC $tag (TP=$TP PP=$PP DP=$DP)  start=$(date +%H:%M:%S) ==="

  # make_static_schedule.py writes <out-dir>/tp{TP}_pp{PP}_dp{DP}.json
  # and prints that path on stdout — capture it for --schedule.
  SCHED=$(python make_static_schedule.py \
            --tp "$TP" --pp "$PP" --dp "$DP" \
            --iters "$TRAIN_ITERS" \
            --out-dir "$RUN_DIR")

  if [[ -z "$SCHED" || ! -f "$SCHED" ]]; then
    echo "[fatal] make_static_schedule.py produced no schedule (got: '$SCHED')" >&2
    exit 3
  fi

  python train_elastic.py \
    --model    "$MODEL" \
    --schedule "$SCHED" \
    --num-gpus 4 \
    --work-dir "$RUN_DIR" \
    "${COMMON_BINDS[@]}"

  echo "[long] === STATIC $tag done  $(date +%H:%M:%S) ==="
  echo
done

# ── Hybrid phase ─────────────────────────────────────────────
RUN_DIR_HY="$PROJECT/runs/long_${SLURM_JOB_ID}/hybrid"
mkdir -p "$RUN_DIR_HY"
echo "[long] === HYBRID  start=$(date +%H:%M:%S) ==="

python train_online_elastic.py \
  --model              "$MODEL" \
  --scheduler-config   "$SCHED_CFG" \
  --carbon-forecast    "$CARBON_CSV" \
  --train-iters        "$TRAIN_ITERS" \
  --switch-power-w     "$SWITCH_POWER_W" \
  --switch-time-seconds "$SWITCH_TIME_S" \
  --deadline-seconds   "$DEADLINE_S" \
  --lookahead-steps    "$LOOKAHEAD_STEPS" \
  --decide-every       "$DECIDE_EVERY" \
  --bucket-seconds     "$BUCKET_SEC" \
  --num-gpus           4 \
  --work-dir           "$RUN_DIR_HY" \
  "${COMMON_BINDS[@]}"

echo
echo "[long] all phases complete  $(date)"
echo "[long] runs in:    $PROJECT/runs/long_${SLURM_JOB_ID}/"
echo "[long] power logs: $POWER_DIR"
echo "[long] next:       python visualize_carbon_long.py"
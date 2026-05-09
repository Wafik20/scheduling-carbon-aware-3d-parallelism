#!/bin/bash
#SBATCH --job-name=naive-elastic
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# 4-GPU, 1-node naive carbon-bucket elastic run.
#
# Strategies are defined in this sbatch and passed into train_naive_elastic.py:
#   low:  TP=1 PP=1 DP=4
#   med:  TP=1 PP=1 DP=4
#   high: TP=4 PP=1 DP=1
#
# The carbon CSV is compressed onto the expected run horizon by setting:
#   bucket_seconds = HORIZON_S / number_of_csv_rows
#
# Override via env, for example:
#   TRAIN_ITERS=10000 HORIZON_S=3600 sbatch batch_files/run_naive_elastic.sbatch
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
TRAIN_ITERS=${TRAIN_ITERS:-15000}
HORIZON_S=${HORIZON_S:-3600}
DECIDE_EVERY=${DECIDE_EVERY:-500}

LOW_TP=${LOW_TP:-1}
LOW_PP=${LOW_PP:-1}
LOW_DP=${LOW_DP:-4}
MED_TP=${MED_TP:-1}
MED_PP=${MED_PP:-1}
MED_DP=${MED_DP:-4}
HIGH_TP=${HIGH_TP:-4}
HIGH_PP=${HIGH_PP:-1}
HIGH_DP=${HIGH_DP:-1}

CARBON_CSV="$PROJECT/data/carbon_week.csv"
RUN_DIR="$PROJECT/runs/naive_${SLURM_JOB_ID}"

[[ -f "$CARBON_CSV" ]] || { echo "[fatal] missing $CARBON_CSV"; exit 2; }
[[ -f "$CONTAINER"  ]] || { echo "[fatal] missing $CONTAINER"; exit 2; }
[[ -d "$MEGATRON"   ]] || { echo "[fatal] missing $MEGATRON"; exit 2; }

CARBON_ROWS=$(awk -F, 'NR>1{n++} END{print n+0}' "$CARBON_CSV")
if [[ "$CARBON_ROWS" -le 0 ]]; then
  echo "[fatal] $CARBON_CSV has no carbon rows" >&2
  exit 2
fi

BUCKET_SEC=$(python -c "print(${HORIZON_S} / ${CARBON_ROWS})")
mkdir -p "$RUN_DIR"

echo "[naive] node=$(hostname)  start=$(date)"
echo "[naive] MODEL=$MODEL  TRAIN_ITERS=$TRAIN_ITERS"
echo "[naive] strategies: low=tp${LOW_TP}_pp${LOW_PP}_dp${LOW_DP}  med=tp${MED_TP}_pp${MED_PP}_dp${MED_DP}  high=tp${HIGH_TP}_pp${HIGH_PP}_dp${HIGH_DP}"
echo "[naive] CARBON_CSV=$CARBON_CSV"
echo "[naive] rows=$CARBON_ROWS  horizon=${HORIZON_S}s  bucket=${BUCKET_SEC}s"
echo "[naive] DECIDE_EVERY=$DECIDE_EVERY"
echo "[naive] RUN_DIR=$RUN_DIR"
echo

python train_naive_elastic.py \
  --model           "$MODEL" \
  --carbon-forecast "$CARBON_CSV" \
  --train-iters     "$TRAIN_ITERS" \
  --bucket-seconds  "$BUCKET_SEC" \
  --decide-every    "$DECIDE_EVERY" \
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
  --work-dir        "$RUN_DIR" \
  "${COMMON_BINDS[@]}"

echo
echo "[naive] complete  $(date)"
echo "[naive] run dir: $RUN_DIR"
echo "[naive] phases:  $RUN_DIR/phases.json"
echo "[naive] summary: $RUN_DIR/elastic_results.json"

#!/bin/bash
#SBATCH --job-name=hybrid-only
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#
# Hybrid-only sweep — skip the static baselines because their power CSVs
# are already on disk from a previous full sweep. Use this for fast
# iteration on the scheduler (warm-start, decide-every, lookahead, CI shape).
# When the static baselines need refreshing — e.g., after a calibration
# change or a model change — go back to run_short_sweep.sbatch.
#
# Override knobs via the environment, no need to edit the file:
#   TRAIN_ITERS=750 DECIDE_EVERY=100 sbatch run_hybrid.sbatch
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
DEADLINE_S=${DEADLINE_S:-400}
BUCKET_SEC=${BUCKET_SEC:-60}
SWITCH_POWER_W=${SWITCH_POWER_W:-200}
LOOKAHEAD_STEPS=${LOOKAHEAD_STEPS:-600}
DECIDE_EVERY=${DECIDE_EVERY:-100}

CARBON_CSV="$PROJECT/data/carbon_short.csv"
SCHED_CFG="$PROJECT/calibrations/calibration_gpt2_medium_4gpu.json"
POWER_DIR="$PROJECT/power_logs/$MODEL"

# ── Sanity: confirm static CSVs exist on disk ────────────────
# The visualizer reads these alongside the hybrid run to build the
# comparison plot. If any are missing, the plot will show fewer
# baselines and the verdict line may be misleading.
EXPECTED_STATICS=(
  "elastic_tp1_pp1_dp4_power.csv"
  "elastic_tp1_pp2_dp2_power.csv"
  "elastic_tp1_pp4_dp1_power.csv"
  "elastic_tp2_pp1_dp2_power.csv"
  "elastic_tp2_pp2_dp1_power.csv"
  "elastic_tp4_pp1_dp1_power.csv"
)

missing=()
for f in "${EXPECTED_STATICS[@]}"; do
  if [[ ! -f "$POWER_DIR/$f" ]] || [[ ! -s "$POWER_DIR/$f" ]]; then
    missing+=("$f")
  fi
done

if (( ${#missing[@]} > 0 )); then
  echo "[fatal] missing static baselines under $POWER_DIR :"
  printf '         %s\n' "${missing[@]}"
  echo
  echo "        Run a full sweep first to generate them:"
  echo "          sbatch run_short_sweep.sbatch"
  echo "        Then come back to run_hybrid.sbatch for fast iteration."
  exit 2
fi

# Also check the inputs the hybrid actually needs.
[[ -f "$CARBON_CSV" ]] || { echo "[fatal] missing $CARBON_CSV"; exit 2; }
[[ -f "$SCHED_CFG"  ]] || { echo "[fatal] missing $SCHED_CFG"; exit 2; }

echo "[sweep] node=$(hostname)  start=$(date)"
echo "[sweep] PROJECT=$PROJECT"
echo "[sweep] MODEL=$MODEL  TRAIN_ITERS=$TRAIN_ITERS  DEADLINE_S=$DEADLINE_S"
echo "[sweep] CARBON_CSV=$CARBON_CSV (bucket=${BUCKET_SEC}s)"
echo "[sweep] SCHED_CFG=$SCHED_CFG"
echo "[sweep] LOOKAHEAD_STEPS=$LOOKAHEAD_STEPS  DECIDE_EVERY=$DECIDE_EVERY"
echo "[sweep] reusing static CSVs in $POWER_DIR (${#EXPECTED_STATICS[@]} found)"
echo

# ── HYBRID PHASE ─────────────────────────────────────────────
RUN_DIR_HY="$PROJECT/runs/hybrid_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR_HY"
echo "[sweep] === HYBRID  start=$(date +%H:%M:%S)  RUN_DIR=$RUN_DIR_HY ==="

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
  "${COMMON_BINDS[@]}"

echo
echo "[sweep] done=$(date)"
echo "[sweep] hybrid power log:  $POWER_DIR/online_elastic_$(basename "${SCHED_CFG%.json}")_power.csv"
echo "[sweep] phases:            $RUN_DIR_HY/phases.json"
echo "[sweep] inner summary:     $RUN_DIR_HY/elastic_results.json"
echo "[sweep] next:              python visualize_carbon_short.py"
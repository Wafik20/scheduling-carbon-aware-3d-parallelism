#!/bin/bash
set -uo pipefail

# Submit GPT-2 medium multinode sweeps.
#
# Default behavior:
#   one Slurm job per valid TP/PP/DP config.
#
# Full run:
#   ./scripts/submit_gpt2m_node_sweeps.sh
#
# Smoke test:
#   ./scripts/submit_gpt2m_node_sweeps.sh smoke
#
# All per-config jobs from one invocation share a single SWEEP_ID, so
# their power_logs go under ONE folder (one subdir per config).
# Override the sweep id with SWEEP_ID=my_tag ./submit_gpt2m_node_sweeps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}" || exit 1
mkdir -p slurm_logs

SBATCH_FILE=${SBATCH_FILE:-"${PROJECT_ROOT}/batch_files/run_gpt2m_multinode_sweep_dynamic.sbatch"}

if [ ! -f "$SBATCH_FILE" ]; then
    echo "ERROR: Could not find batch file:"
    echo "  $SBATCH_FILE"
    exit 1
fi

MODE=${1:-full}

case "$MODE" in
    full)
        SMOKE_TEST=0
        MODE_TAG="full"
        ;;
    smoke|--smoke)
        SMOKE_TEST=1
        MODE_TAG="smoke"
        ;;
    *)
        echo "ERROR: Unknown mode: $MODE"
        echo "Use one of:"
        echo "  ./scripts/submit_gpt2m_node_sweeps.sh"
        echo "  ./scripts/submit_gpt2m_node_sweeps.sh smoke"
        exit 1
        ;;
esac

# Default to one job per config so a slow pure-DP config does not block all others.
SUBMIT_MODE=${SUBMIT_MODE:-per_config}

case "$SUBMIT_MODE" in
    per_config|sequential)
        ;;
    *)
        echo "ERROR: Unknown SUBMIT_MODE=$SUBMIT_MODE"
        echo "Use one of:"
        echo "  SUBMIT_MODE=per_config"
        echo "  SUBMIT_MODE=sequential"
        exit 1
        ;;
esac

# Sweep ID groups all per-config jobs from this invocation into a single
# power_logs subfolder. Override with SWEEP_ID=... to share or resume.
SWEEP_ID="${SWEEP_ID:-$(date +%Y%m%d_%H%M%S)}"

NODE_LIST=${NODE_LIST:-"2 4 6 8"}

SBATCH_PARTITION=${SBATCH_PARTITION:-"a100_multi"}
SBATCH_CONSTRAINT=${SBATCH_CONSTRAINT:-"a100&40g&amd"}

FULL_TIME=${FULL_TIME:-"04:00:00"}
PURE_DP_TIME=${PURE_DP_TIME:-"04:00:00"}
SMOKE_TIME=${SMOKE_TIME:-"00:25:00"}

MODEL_NAME=${MODEL_NAME:-gpt2_medium}
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
HEADS=${HEADS:-16}
SEQ=${SEQ:-1024}
VOCAB=${VOCAB:-50257}
MICRO_BATCH=${MICRO_BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-192}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

TRAIN_ITERS=${TRAIN_ITERS:-80}
LOG_INTERVAL=${LOG_INTERVAL:-1}

SMOKE_TRAIN_ITERS=${SMOKE_TRAIN_ITERS:-5}
SMOKE_CONFIG_LIMIT=${SMOKE_CONFIG_LIMIT:-2}

if [ "$SMOKE_TEST" = "1" ]; then
    WALL_TIME="$SMOKE_TIME"
else
    WALL_TIME="$FULL_TIME"
fi

valid_configs() {
    local W="$1"

    for TP in $(seq 1 "$W"); do
        if [ $((W % TP)) -ne 0 ]; then
            continue
        fi

        if [ $((HEADS % TP)) -ne 0 ]; then
            continue
        fi

        for PP in $(seq 1 "$W"); do
            local TP_PP=$((TP * PP))

            if [ "$TP_PP" -gt "$W" ]; then
                continue
            fi

            if [ $((W % TP_PP)) -ne 0 ]; then
                continue
            fi

            if [ $((NUM_LAYERS % PP)) -ne 0 ]; then
                continue
            fi

            local DP=$((W / TP_PP))

            if [ $((GLOBAL_BATCH % (MICRO_BATCH * DP))) -ne 0 ]; then
                continue
            fi

            echo "tp${TP}_pp${PP}_dp${DP}"
        done
    done
}

echo "============================================================"
echo "  Submitting GPT-2 medium multinode sweeps"
echo "============================================================"
echo "Mode:               $MODE_TAG"
echo "Submit mode:        $SUBMIT_MODE"
echo "Sweep ID:           $SWEEP_ID"
echo "Node list:          $NODE_LIST"
echo "Batch file:         $SBATCH_FILE"
echo "Partition:          $SBATCH_PARTITION"
echo "Constraint:         ${SBATCH_CONSTRAINT:-<none>}"
echo "Wall time:          $WALL_TIME"
echo "Pure-DP wall time:  $PURE_DP_TIME"
echo "MODEL_NAME:         $MODEL_NAME"
echo "NUM_LAYERS:         $NUM_LAYERS"
echo "HIDDEN:             $HIDDEN"
echo "HEADS:              $HEADS"
echo "SEQ:                $SEQ"
echo "MICRO_BATCH:        $MICRO_BATCH"
echo "GLOBAL_BATCH:       $GLOBAL_BATCH"
echo "TRAIN_ITERS:        $TRAIN_ITERS"
echo "LOG_INTERVAL:       $LOG_INTERVAL"
echo "SMOKE_TRAIN_ITERS:  $SMOKE_TRAIN_ITERS"
echo "SMOKE_CONFIG_LIMIT: $SMOKE_CONFIG_LIMIT"
echo

SUBMITTED=0
FAILED=0

submit_one() {
    local NODES="$1"
    local JOB_NAME="$2"
    local OUT_FILE="$3"
    local THIS_WALL_TIME="$4"
    local ONLY_CONFIG_VALUE="${5:-}"

    if [ -n "$SBATCH_CONSTRAINT" ]; then
        CONSTRAINT_ARGS=(--constraint="$SBATCH_CONSTRAINT")
    else
        CONSTRAINT_ARGS=()
    fi

    EXPORTS="ALL"
    EXPORTS="${EXPORTS},SWEEP_ID=${SWEEP_ID}"
    EXPORTS="${EXPORTS},SMOKE_TEST=${SMOKE_TEST}"
    EXPORTS="${EXPORTS},SMOKE_TRAIN_ITERS=${SMOKE_TRAIN_ITERS}"
    EXPORTS="${EXPORTS},SMOKE_CONFIG_LIMIT=${SMOKE_CONFIG_LIMIT}"
    EXPORTS="${EXPORTS},MODEL_NAME=${MODEL_NAME}"
    EXPORTS="${EXPORTS},NUM_LAYERS=${NUM_LAYERS}"
    EXPORTS="${EXPORTS},HIDDEN=${HIDDEN}"
    EXPORTS="${EXPORTS},HEADS=${HEADS}"
    EXPORTS="${EXPORTS},SEQ=${SEQ}"
    EXPORTS="${EXPORTS},VOCAB=${VOCAB}"
    EXPORTS="${EXPORTS},MICRO_BATCH=${MICRO_BATCH}"
    EXPORTS="${EXPORTS},GLOBAL_BATCH=${GLOBAL_BATCH}"
    EXPORTS="${EXPORTS},TRAIN_ITERS=${TRAIN_ITERS}"
    EXPORTS="${EXPORTS},LOG_INTERVAL=${LOG_INTERVAL}"

    if [ -n "$ONLY_CONFIG_VALUE" ]; then
        EXPORTS="${EXPORTS},ONLY_CONFIG=${ONLY_CONFIG_VALUE}"
    fi

    if sbatch \
        --nodes="$NODES" \
        --partition="$SBATCH_PARTITION" \
        "${CONSTRAINT_ARGS[@]}" \
        --time="$THIS_WALL_TIME" \
        --job-name="$JOB_NAME" \
        --output="$OUT_FILE" \
        --export="$EXPORTS" \
        "$SBATCH_FILE"; then

        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "WARNING: Failed to submit ${JOB_NAME}"
        FAILED=$((FAILED + 1))
    fi
}

for NODES in $NODE_LIST; do
    WORLD_SIZE=$((NODES * GPUS_PER_NODE))
    mapfile -t CONFIGS < <(valid_configs "$WORLD_SIZE")

    if [ "$SMOKE_TEST" = "1" ]; then
        CONFIGS=("${CONFIGS[@]:0:$SMOKE_CONFIG_LIMIT}")
    fi

    echo "------------------------------------------------------------"
    echo "n=${NODES}, world_size=${WORLD_SIZE}, configs=${#CONFIGS[@]}"
    printf '  %s\n' "${CONFIGS[@]}"
    echo

    if [ "$SUBMIT_MODE" = "sequential" ]; then
        JOB_NAME="gpt2m-${MODE_TAG}-n${NODES}"
        OUT_FILE="slurm_logs/${JOB_NAME}-%j.out"

        echo "Submitting sequential ${JOB_NAME}..."
        submit_one "$NODES" "$JOB_NAME" "$OUT_FILE" "$WALL_TIME" ""
        echo
        continue
    fi

    for CONFIG in "${CONFIGS[@]}"; do
        JOB_NAME="gpt2m-${MODE_TAG}-n${NODES}-${CONFIG}"
        OUT_FILE="slurm_logs/${JOB_NAME}-%j.out"

        if [ "$SMOKE_TEST" = "1" ]; then
            THIS_WALL_TIME="$SMOKE_TIME"
        elif [[ "$CONFIG" == tp1_pp1_dp* ]]; then
            THIS_WALL_TIME="$PURE_DP_TIME"
        else
            THIS_WALL_TIME="$FULL_TIME"
        fi

        echo "Submitting ${JOB_NAME} time=${THIS_WALL_TIME}..."
        submit_one "$NODES" "$JOB_NAME" "$OUT_FILE" "$THIS_WALL_TIME" "$CONFIG"
    done

    echo
done

echo "============================================================"
echo "  Submission complete"
echo "============================================================"
echo "Sweep ID:  $SWEEP_ID"
echo "Submitted: $SUBMITTED"
echo "Failed:    $FAILED"
echo
echo "All power logs will land under:"
echo "  power_logs/${MODEL_NAME}_n<N>_${SWEEP_ID}/"
echo "  (smoke runs use ${MODEL_NAME}_smoke_n<N>_${SWEEP_ID}/)"
echo
echo "Check jobs:"
echo "  squeue -M gpu -u \$USER"
echo
echo "Check accounting:"
echo "  sacct -M gpu -u \$USER -S today --format=JobID,JobName%45,State,ExitCode,Elapsed,AllocNodes"
echo
echo "Tail logs:"
echo "  ls -lh slurm_logs/"
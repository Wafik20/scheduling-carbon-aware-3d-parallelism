#!/bin/bash
#SBATCH --job-name=gpt2m-sweep
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_multi
#SBATCH --constraint=a100&40g&amd
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# Do not exit the whole sweep on one failed config.
# We record per-config status and exit nonzero at the end if any failed.
set -uo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism}
CONTAINER=${CONTAINER:-/bgfs/stephenlee/wat29/nemo_2504.sif}
MEGATRON_HOST=${MEGATRON_HOST:-/bgfs/stephenlee/wat29/Megatron-LM}
RUN_INSIDE=${RUN_INSIDE:-"${PROJECT_DIR}/run_inside_container.sh"}

# ── Model config ─────────────────────────────────────────────
MODEL_NAME=${MODEL_NAME:-gpt2_medium}
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
HEADS=${HEADS:-16}
SEQ=${SEQ:-1024}
VOCAB=${VOCAB:-50257}
MICRO_BATCH=${MICRO_BATCH:-2}

# Important:
# GLOBAL_BATCH=192 works cleanly for DP values from 2, 4, 6, and 8 node runs.
# With 4 GPUs/node:
#   n2 -> W=8,  max DP=8,  MBS*DP=16
#   n4 -> W=16, max DP=16, MBS*DP=32
#   n6 -> W=24, max DP=24, MBS*DP=48
#   n8 -> W=32, max DP=32, MBS*DP=64
# 192 is divisible by all of these.
GLOBAL_BATCH=${GLOBAL_BATCH:-192}

TRAIN_ITERS=${TRAIN_ITERS:-80}
LOG_INTERVAL=${LOG_INTERVAL:-1}

# ── Smoke test mode ──────────────────────────────────────────
SMOKE_TEST=${SMOKE_TEST:-0}
SMOKE_TRAIN_ITERS=${SMOKE_TRAIN_ITERS:-5}
SMOKE_CONFIG_LIMIT=${SMOKE_CONFIG_LIMIT:-2}

if [ "$SMOKE_TEST" = "1" ]; then
    TRAIN_ITERS="$SMOKE_TRAIN_ITERS"
fi

# ── Derived ──────────────────────────────────────────────────
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

if [ "$SMOKE_TEST" = "1" ]; then
    RUN_NAME="${MODEL_NAME}_smoke_n${SLURM_NNODES}_${SLURM_JOB_ID}"
else
    RUN_NAME="${MODEL_NAME}_n${SLURM_NNODES}_${SLURM_JOB_ID}"
fi

RUN_BASE_DIR="${PROJECT_DIR}/power_logs/${RUN_NAME}"

mkdir -p "${RUN_BASE_DIR}" "${PROJECT_DIR}/slurm_logs"
cd "${PROJECT_DIR}" || exit 1

if [ ! -f "$RUN_INSIDE" ]; then
    echo "ERROR: Could not find run_inside_container.sh at:"
    echo "  $RUN_INSIDE"
    exit 1
fi

chmod +x "$RUN_INSIDE"

# ── Modules ──────────────────────────────────────────────────
module purge
module load apptainer

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)

# Job-dependent port to reduce collisions.
export MASTER_PORT=${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
export GPUS_PER_NODE

# ── Generate every valid TP/PP/DP factorization ──────────────
# No TP/PP cap.
# We enumerate every integer TP and PP from 1..TOTAL_GPUS, compute DP,
# and keep only configs satisfying:
#   TP * PP * DP = TOTAL_GPUS
#   HEADS % TP = 0
#   NUM_LAYERS % PP = 0
#   GLOBAL_BATCH % (MICRO_BATCH * DP) = 0
CONFIGS=()

for TP in $(seq 1 "$TOTAL_GPUS"); do
    if [ $((TOTAL_GPUS % TP)) -ne 0 ]; then
        continue
    fi

    if [ $((HEADS % TP)) -ne 0 ]; then
        continue
    fi

    for PP in $(seq 1 "$TOTAL_GPUS"); do
        TP_PP=$((TP * PP))

        if [ "$TP_PP" -gt "$TOTAL_GPUS" ]; then
            continue
        fi

        if [ $((TOTAL_GPUS % TP_PP)) -ne 0 ]; then
            continue
        fi

        if [ $((NUM_LAYERS % PP)) -ne 0 ]; then
            continue
        fi

        DP=$((TOTAL_GPUS / TP_PP))

        if [ $((GLOBAL_BATCH % (MICRO_BATCH * DP))) -ne 0 ]; then
            continue
        fi

        CONFIGS+=("${TP} ${PP} ${DP}")
    done
done

# Smoke mode is allowed to cap only for quick validation.
# Full mode never caps.
if [ "$SMOKE_TEST" = "1" ]; then
    CONFIGS=("${CONFIGS[@]:0:$SMOKE_CONFIG_LIMIT}")
fi

# ── Metadata ─────────────────────────────────────────────────
{
    echo "MODEL_NAME=${MODEL_NAME}"
    echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    echo "SLURM_JOB_NAME=${SLURM_JOB_NAME}"
    echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unknown}"
    echo "SLURM_NNODES=${SLURM_NNODES}"
    echo "SLURM_NODELIST=${SLURM_NODELIST}"
    echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
    echo "TOTAL_GPUS=${TOTAL_GPUS}"
    echo "MASTER_ADDR=${MASTER_ADDR}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "NUM_LAYERS=${NUM_LAYERS}"
    echo "HIDDEN=${HIDDEN}"
    echo "HEADS=${HEADS}"
    echo "SEQ=${SEQ}"
    echo "VOCAB=${VOCAB}"
    echo "MICRO_BATCH=${MICRO_BATCH}"
    echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
    echo "TRAIN_ITERS=${TRAIN_ITERS}"
    echo "LOG_INTERVAL=${LOG_INTERVAL}"
    echo "SMOKE_TEST=${SMOKE_TEST}"
    echo "SMOKE_TRAIN_ITERS=${SMOKE_TRAIN_ITERS}"
    echo "SMOKE_CONFIG_LIMIT=${SMOKE_CONFIG_LIMIT}"
    echo "CONFIG_GENERATION=all valid integer factorizations"
    echo "CONFIGS:"
    printf '  %s\n' "${CONFIGS[@]}"
} > "${RUN_BASE_DIR}/sweep_metadata.txt"

echo "============================================================"
echo "  ${MODEL_NAME} all-valid parallel-config sweep"
echo "============================================================"
echo "Job:        $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "Nodes:      $SLURM_NNODES, $SLURM_NODELIST"
echo "GPUs:       ${TOTAL_GPUS} (${GPUS_PER_NODE}/node)"
echo "World size: ${TOTAL_GPUS}"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "GBS/MBS:    ${GLOBAL_BATCH}/${MICRO_BATCH}"
echo "Iters:      $TRAIN_ITERS per config"
echo "Log int:    $LOG_INTERVAL"
echo "Smoke test: $SMOKE_TEST"
echo "Configs:    ${#CONFIGS[@]}"
printf '  - TP PP DP: %s\n' "${CONFIGS[@]}"
echo "Logs:       $RUN_BASE_DIR"
echo

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "ERROR: No valid configs generated."
    exit 1
fi

# ── Sidecar cleanup helper ───────────────────────────────────
SIDECAR_PID=""

cleanup_sidecar() {
    if [ -n "${SIDECAR_PID:-}" ]; then
        kill -TERM "$SIDECAR_PID" 2>/dev/null || true
        wait "$SIDECAR_PID" 2>/dev/null || true
        SIDECAR_PID=""
    fi
}

trap cleanup_sidecar EXIT TERM INT

VALID_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

for cfg in "${CONFIGS[@]}"; do
    read -r TP PP DP <<< "$cfg"
    CONFIG_NAME="tp${TP}_pp${PP}_dp${DP}"

    echo
    echo "============================================================"
    echo "  CONFIG: ${CONFIG_NAME}  (TP=$TP PP=$PP DP=$DP)"
    echo "============================================================"

    POWER_DIR="${RUN_BASE_DIR}/${CONFIG_NAME}"
    mkdir -p "${POWER_DIR}"
    VALID_RUNS=$((VALID_RUNS + 1))

    {
        echo "CONFIG_NAME=${CONFIG_NAME}"
        echo "TP=${TP}"
        echo "PP=${PP}"
        echo "DP=${DP}"
        echo "TOTAL_GPUS=${TOTAL_GPUS}"
        echo "SLURM_NNODES=${SLURM_NNODES}"
        echo "SLURM_NODELIST=${SLURM_NODELIST}"
        echo "GLOBAL_BATCH=${GLOBAL_BATCH}"
        echo "MICRO_BATCH=${MICRO_BATCH}"
        echo "TRAIN_ITERS=${TRAIN_ITERS}"
        echo "LOG_INTERVAL=${LOG_INTERVAL}"
    } > "${POWER_DIR}/config_metadata.txt"

    echo "  Logs: ${POWER_DIR}"

    # ── Sidecar power logging ────────────────────────────────
    srun --overlap \
         --nodes="$SLURM_NNODES" \
         --ntasks-per-node=1 \
         --output="${POWER_DIR}/sidecar_%n.log" \
         python "${PROJECT_DIR}/compute_power_multinode.py" \
                --log-dir "${POWER_DIR}" \
                --interval 0.5 \
                --config-tag "${CONFIG_NAME}" &

    SIDECAR_PID=$!
    sleep 3

    # ── Training ─────────────────────────────────────────────
    srun --nodes="$SLURM_NNODES" \
         --ntasks-per-node=1 \
         bash -lc "
            module purge
            module load apptainer

            unset NCCL_SOCKET_IFNAME
            unset NCCL_NET
            unset NCCL_IB_DISABLE
            unset APPTAINERENV_NCCL_SOCKET_IFNAME
            unset APPTAINERENV_NCCL_NET
            unset APPTAINERENV_NCCL_IB_DISABLE

            apptainer exec --nv \
                --bind ${PROJECT_DIR}:${PROJECT_DIR} \
                --bind /bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29 \
                --bind ${MEGATRON_HOST}:/opt/megatron-lm \
                --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
                --env PYTHONUNBUFFERED=1 \
                --env NCCL_IB_DISABLE=1 \
                --env NCCL_NET=Socket \
                --env NCCL_SOCKET_IFNAME='^lo,docker0,virbr0,veth' \
                --env NCCL_DEBUG=INFO \
                --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
                --env SLURM_NNODES=${SLURM_NNODES} \
                --env SLURM_NODEID=\${SLURM_NODEID} \
                --env GPUS_PER_NODE=${GPUS_PER_NODE} \
                --env MASTER_ADDR=${MASTER_ADDR} \
                --env MASTER_PORT=${MASTER_PORT} \
                --env TP=${TP} \
                --env PP=${PP} \
                --env NUM_LAYERS=${NUM_LAYERS} \
                --env HIDDEN=${HIDDEN} \
                --env HEADS=${HEADS} \
                --env SEQ=${SEQ} \
                --env VOCAB=${VOCAB} \
                --env MICRO_BATCH=${MICRO_BATCH} \
                --env GLOBAL_BATCH=${GLOBAL_BATCH} \
                --env TRAIN_ITERS=${TRAIN_ITERS} \
                --env LOG_INTERVAL=${LOG_INTERVAL} \
                ${CONTAINER} \
                ${RUN_INSIDE}
         "

    TRAIN_RC=$?

    cleanup_sidecar

    if [ "$TRAIN_RC" -eq 0 ]; then
        echo "  ${CONFIG_NAME} OK"
        echo "OK" > "${POWER_DIR}/status.txt"
        SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
    else
        echo "  ${CONFIG_NAME} FAILED (rc=$TRAIN_RC)"
        echo "FAILED rc=${TRAIN_RC}" > "${POWER_DIR}/status.txt"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi

    sleep 5
done

echo
echo "============================================================"
echo "  SWEEP COMPLETE"
echo "============================================================"
echo "Valid configs attempted: $VALID_RUNS / ${#CONFIGS[@]}"
echo "Successful configs:      $SUCCESS_RUNS"
echo "Failed configs:          $FAILED_RUNS"
echo "All logs:                $RUN_BASE_DIR"
ls -la "${RUN_BASE_DIR}"

{
    echo "VALID_RUNS=${VALID_RUNS}"
    echo "SUCCESS_RUNS=${SUCCESS_RUNS}"
    echo "FAILED_RUNS=${FAILED_RUNS}"
    echo "RUN_BASE_DIR=${RUN_BASE_DIR}"
} >> "${RUN_BASE_DIR}/sweep_metadata.txt"

if [ "$VALID_RUNS" -eq 0 ]; then
    echo "ERROR: No valid configs ran."
    exit 1
fi

if [ "$FAILED_RUNS" -gt 0 ]; then
    echo "ERROR: One or more configs failed."
    exit 2
fi
#!/bin/bash
#SBATCH --job-name=gpt2m-sweep
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# Don't exit on single failure
set -uo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON_HOST=/bgfs/stephenlee/wat29/Megatron-LM

# ── Model config ─────────────────────────────────────────────
MODEL_NAME=gpt2_medium
NUM_LAYERS=24
HIDDEN=1024
HEADS=16
SEQ=1024
VOCAB=50257
MICRO_BATCH=2
GLOBAL_BATCH=64
TRAIN_ITERS=80

# ── Derived ──────────────────────────────────────────────────
GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

RUN_NAME="${MODEL_NAME}_n${SLURM_NNODES}_${SLURM_JOB_ID}"
RUN_BASE_DIR="${PROJECT_DIR}/power_logs/${RUN_NAME}"

mkdir -p "${RUN_BASE_DIR}" slurm_logs
cd "${PROJECT_DIR}"

# ── Modules ──────────────────────────────────────────────────
module purge
module load apptainer

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE

# ── Valid configs for 16 GPUs ────────────────────────────────
CONFIGS=(
    "1 1 16"
    "2 1 8"
    "4 1 4"
    "8 1 2"
    "16 1 1"

    "1 2 8"
    "1 4 4"
    "1 8 2"
    "1 16 1"

    "2 2 4"
    "2 4 2"
    "4 2 2"
)

echo "============================================================"
echo "  ${MODEL_NAME} parallel-config sweep"
echo "============================================================"
echo "Job:        $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Nodes:      $SLURM_NNODES — $SLURM_NODELIST"
echo "GPUs:       ${TOTAL_GPUS} (${GPUS_PER_NODE}/node)"
echo "World size: ${TOTAL_GPUS}"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "Configs:    ${#CONFIGS[@]}"
echo "Iters:      $TRAIN_ITERS per config"
echo "Logs:       $RUN_BASE_DIR"
echo

VALID_RUNS=0

for cfg in "${CONFIGS[@]}"; do
    read -r TP PP DP <<< "$cfg"
    CONFIG_NAME="tp${TP}_pp${PP}_dp${DP}"

    echo
    echo "============================================================"
    echo "  CONFIG: ${CONFIG_NAME}  (TP=$TP PP=$PP DP=$DP)"
    echo "============================================================"

    # ── Sanity checks BEFORE creating folder ──
    if [ $((TP * PP * DP)) -ne $TOTAL_GPUS ]; then
        echo "  SKIP: TP*PP*DP=$((TP*PP*DP)) != $TOTAL_GPUS"
        continue
    fi
    if [ $((NUM_LAYERS % PP)) -ne 0 ]; then
        echo "  SKIP: $NUM_LAYERS layers not divisible by PP=$PP"
        continue
    fi
    if [ $((HEADS % TP)) -ne 0 ]; then
        echo "  SKIP: $HEADS heads not divisible by TP=$TP"
        continue
    fi
    if [ $((GLOBAL_BATCH % (MICRO_BATCH * DP))) -ne 0 ]; then
        echo "  SKIP: GBS=$GLOBAL_BATCH not divisible by MBS*DP=$((MICRO_BATCH*DP))"
        continue
    fi

    # ── Create directory ONLY if valid ──
    POWER_DIR="${RUN_BASE_DIR}/${CONFIG_NAME}"
    mkdir -p "${POWER_DIR}"
    VALID_RUNS=$((VALID_RUNS+1))

    echo "  Logs: ${POWER_DIR}"

    # ── Sidecar (power logging) ──────────────────────────────
    srun --overlap \
         --nodes=$SLURM_NNODES --ntasks-per-node=1 \
         --output="${POWER_DIR}/sidecar_%n.log" \
         python "${PROJECT_DIR}/compute_power_multinode.py" \
                --log-dir "${POWER_DIR}" \
                --interval 0.5 \
                --config-tag "${CONFIG_NAME}" &
    SIDECAR_PID=$!

    sleep 3

    # ── Training ─────────────────────────────────────────────
    srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
         bash -c '
            module purge
            module load apptainer

            export APPTAINERENV_NCCL_IB_DISABLE=1
            export APPTAINERENV_NCCL_NET=Socket
            export APPTAINERENV_NCCL_SOCKET_IFNAME=ib0
            export APPTAINERENV_NCCL_DEBUG=WARN

            apptainer exec --nv \
                --bind /bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29 \
                --bind '"${MEGATRON_HOST}"':/opt/megatron-lm \
                --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
                --env SLURM_NNODES=$SLURM_NNODES \
                --env SLURM_NODEID=$SLURM_NODEID \
                --env GPUS_PER_NODE='"${GPUS_PER_NODE}"' \
                --env MASTER_ADDR='"${MASTER_ADDR}"' \
                --env MASTER_PORT='"${MASTER_PORT}"' \
                --env TP='"${TP}"' \
                --env PP='"${PP}"' \
                --env NUM_LAYERS='"${NUM_LAYERS}"' \
                --env HIDDEN='"${HIDDEN}"' \
                --env HEADS='"${HEADS}"' \
                --env SEQ='"${SEQ}"' \
                --env VOCAB='"${VOCAB}"' \
                --env MICRO_BATCH='"${MICRO_BATCH}"' \
                --env GLOBAL_BATCH='"${GLOBAL_BATCH}"' \
                --env TRAIN_ITERS='"${TRAIN_ITERS}"' \
                '"${CONTAINER}"' \
                '"${PROJECT_DIR}"'/run_inside_container.sh
         '

    TRAIN_RC=$?

    # ── Stop sidecar ─────────────────────────────────────────
    kill -TERM $SIDECAR_PID 2>/dev/null || true
    wait $SIDECAR_PID 2>/dev/null || true

    if [ $TRAIN_RC -eq 0 ]; then
        echo "  ${CONFIG_NAME} OK"
    else
        echo "  ${CONFIG_NAME} FAILED (rc=$TRAIN_RC)"
    fi

    sleep 5
done

echo
echo "============================================================"
echo "  SWEEP COMPLETE"
echo "============================================================"
echo "Valid configs executed: $VALID_RUNS / ${#CONFIGS[@]}"
echo "All logs: $RUN_BASE_DIR"
ls -la "${RUN_BASE_DIR}"

if [ $VALID_RUNS -eq 0 ]; then
    echo "ERROR: No valid configs ran"
    exit 1
fi

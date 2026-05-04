#!/bin/bash
#SBATCH --job-name=gpt2m-sweep
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_multi
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# NOT -e — one config failing shouldn't abort the whole sweep
set -uo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON_HOST=/bgfs/stephenlee/wat29/Megatron-LM
RUN_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
RUN_BASE_DIR="${PROJECT_DIR}/power_logs/${RUN_NAME}"
GPUS_PER_NODE=4
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
mkdir -p "${RUN_BASE_DIR}" slurm_logs
cd "${PROJECT_DIR}"

# ── Modules ──────────────────────────────────────────────────
module purge
module load apptainer

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE

# ── gpt2_medium model config ─────────────────────────────────
MODEL_NAME=gpt2_medium
NUM_LAYERS=24
HIDDEN=1024
HEADS=16
SEQ=1024
VOCAB=50257
MICRO_BATCH=2
GLOBAL_BATCH=64
TRAIN_ITERS=80

# ── All TP×PP×DP=8 configs valid for gpt2_medium ─────────────
CONFIGS=(
    "1 1 8"   # pure DP
    "2 1 4"   # TP only
    "4 1 2"   # more TP
    "8 1 1"   # max TP (cross-node)
    "1 2 4"   # PP only
    "1 4 2"   # more PP
    "1 8 1"   # max PP
    "2 2 2"   # balanced
    "4 2 1"   # TP+PP, no DP
    "2 4 1"   # PP+TP, no DP
)

echo "============================================================"
echo "  ${MODEL_NAME} parallel-config sweep  (Socket-over-IPoIB)"
echo "============================================================"
echo "Job:      $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Nodes:    $SLURM_NNODES — $SLURM_NODELIST"
echo "GPUs:     ${TOTAL_GPUS} (${GPUS_PER_NODE}/node)"
echo "Master:   $MASTER_ADDR:$MASTER_PORT"
echo "Configs:  ${#CONFIGS[@]}"
echo "Iters:    $TRAIN_ITERS per config"
echo "Logs:     $RUN_BASE_DIR"
echo "Network:  NCCL Socket transport over ib0 (IPoIB)"
echo

for cfg in "${CONFIGS[@]}"; do
    read -r TP PP DP <<< "$cfg"
    CONFIG_NAME="tp${TP}_pp${PP}_dp${DP}"
    POWER_DIR="${RUN_BASE_DIR}/${CONFIG_NAME}"
    mkdir -p "${POWER_DIR}"

    echo
    echo "============================================================"
    echo "  CONFIG: ${CONFIG_NAME}  (TP=$TP PP=$PP DP=$DP)"
    echo "  Logs:   ${POWER_DIR}"
    echo "============================================================"

    # Sanity-check geometry
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

    # ── Sidecar (one per node) ───────────────────────────────
    srun --overlap \
         --nodes=$SLURM_NNODES --ntasks-per-node=1 \
         --output="${POWER_DIR}/sidecar_%n.log" \
         python "${PROJECT_DIR}/compute_power_multinode.py" \
                --log-dir "${POWER_DIR}" \
                --interval 0.5 \
                --config-tag "${CONFIG_NAME}" &
    SIDECAR_PID=$!

    sleep 3   # baseline idle samples

    # ── Training ─────────────────────────────────────────────
    # Network: NCCL Socket transport pinned to ib0 (IPoIB).
    # Native verbs/RDMA failed inside this container (ibv_open_device denied
    # by Apptainer namespace); IPoIB still uses the HDR200 IB fabric, just
    # via TCP/IP rather than RDMA. Documented as the network condition.
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
                --env CUDA_HOME=/usr/local/cuda \
                --env HOME=/bgfs/stephenlee/wat29 \
                --env XDG_CACHE_HOME=/bgfs/stephenlee/wat29/.cache \
                --env HF_HOME=/bgfs/stephenlee/wat29/.cache/huggingface \
                --env FLASHINFER_WORKSPACE_BASE=/bgfs/stephenlee/wat29/.cache/flashinfer \
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
        echo "  ${CONFIG_NAME} FAILED (rc=$TRAIN_RC) — continuing sweep"
    fi

    sleep 5
done

echo
echo "============================================================"
echo "  SWEEP COMPLETE"
echo "============================================================"
echo "All logs: $RUN_BASE_DIR"
ls -la "${RUN_BASE_DIR}"
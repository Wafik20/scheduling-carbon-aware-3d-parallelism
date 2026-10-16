#!/bin/bash
set -euo pipefail

# Called once per node inside a container with torchrun and Megatron installed.
# Leave NCCL transport/interface selection to the site or caller.
: "${MEGATRON_DIR:?Set MEGATRON_DIR to the in-container Megatron-LM directory}"
: "${SLURM_NNODES:?Set SLURM_NNODES to the number of nodes}"
: "${SLURM_NODEID:?Set SLURM_NODEID to the node rank}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous host}"
: "${MASTER_PORT:?Set MASTER_PORT to the rendezvous port}"
export PYTHONUNBUFFERED=1

TP=${TP:-2}
PP=${PP:-2}
NUM_LAYERS=${NUM_LAYERS:-24}
HIDDEN=${HIDDEN:-1024}
HEADS=${HEADS:-16}
SEQ=${SEQ:-1024}
VOCAB=${VOCAB:-50257}
MICRO_BATCH=${MICRO_BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-192}
TRAIN_ITERS=${TRAIN_ITERS:-80}
LOG_INTERVAL=${LOG_INTERVAL:-1}

echo "===== INSIDE CONTAINER ($(hostname)) ====="
echo "Time: $(date)"
echo "TP=$TP PP=$PP layers=$NUM_LAYERS hidden=$HIDDEN heads=$HEADS"
echo "MBS=$MICRO_BATCH GBS=$GLOBAL_BATCH iters=$TRAIN_ITERS log_interval=$LOG_INTERVAL"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT node_rank=$SLURM_NODEID"
echo "Starting torchrun at $(date)"

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$MEGATRON_DIR/pretrain_gpt.py" \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size "$PP" \
    --num-layers "$NUM_LAYERS" \
    --hidden-size "$HIDDEN" \
    --num-attention-heads "$HEADS" \
    --seq-length "$SEQ" \
    --max-position-embeddings "$SEQ" \
    --micro-batch-size "$MICRO_BATCH" \
    --global-batch-size "$GLOBAL_BATCH" \
    --train-iters "$TRAIN_ITERS" \
    --lr 1.5e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size "$VOCAB" \
    --split 99,1,0 \
    --log-interval "$LOG_INTERVAL" \
    --eval-iters 0 \
    --eval-interval 1000 \
    --bf16

echo "=========================================="
echo "torchrun finished at $(date)"
echo "=========================================="
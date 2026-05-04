#!/bin/bash
set -e

# Network: NCCL Socket transport pinned to ib0 (IPoIB).
# Native IB verbs unavailable inside this Apptainer container
# (ibv_open_device denied by namespace; libs and devices visible but
# the kernel refuses to open them). Falling back to TCP-over-IB.
# All NCCL_* values are also exported via APPTAINERENV_* in the sbatch —
# duplicating here as defense-in-depth in case the env passing changes.
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

# Defaults if env vars not set
TP=${TP:-2}
PP=${PP:-2}
NUM_LAYERS=${NUM_LAYERS:-12}
HIDDEN=${HIDDEN:-768}
HEADS=${HEADS:-12}
SEQ=${SEQ:-1024}
VOCAB=${VOCAB:-50257}
MICRO_BATCH=${MICRO_BATCH:-2}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
TRAIN_ITERS=${TRAIN_ITERS:-200}

echo "===== INSIDE CONTAINER ($(hostname)) ====="
echo "TP=$TP PP=$PP  layers=$NUM_LAYERS hidden=$HIDDEN heads=$HEADS"
echo "MBS=$MICRO_BATCH GBS=$GLOBAL_BATCH iters=$TRAIN_ITERS"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT  node_rank=$SLURM_NODEID"
echo "NCCL: NET=$NCCL_NET SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "=========================================="

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /opt/megatron-lm/pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS --hidden-size $HIDDEN --num-attention-heads $HEADS \
    --seq-length $SEQ --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITERS \
    --lr 1.5e-4 --min-lr 1e-5 --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 --clip-grad 1.0 --weight-decay 0.1 \
    --mock-data --tokenizer-type NullTokenizer --vocab-size $VOCAB \
    --split 99,1,0 \
    --log-interval 10 --eval-iters 0 --eval-interval 1000 \
    --bf16
#!/bin/bash
set -euo pipefail

# NCCL setup.
# Pin to eth0 explicitly. The 10.201.x.x subnet is uniform across all
# a100_multi nodes; ib0 (10.203.x.x) is not exposed on every node in
# this partition, so exclusion-only patterns produce heterogeneous
# interface lists that deadlock at the first cross-rank connect.

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET="${NCCL_NET:-Socket}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GRAPH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# Clear deprecated variable if inherited from old runs.
unset NCCL_ASYNC_ERROR_HANDLING || true

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
echo "NCCL: IB_DISABLE=$NCCL_IB_DISABLE NET=$NCCL_NET SOCKET_IFNAME=$NCCL_SOCKET_IFNAME DEBUG=$NCCL_DEBUG SUBSYS=$NCCL_DEBUG_SUBSYS"
echo "----- Network diagnostics -----"
echo "Hostname:        $(hostname)"
echo "Hostname (FQDN): $(hostname -f 2>/dev/null || echo unknown)"
echo "Resolved IPs:"
(getent hosts "$(hostname)" 2>/dev/null | sed 's/^/  /' || echo "  (getent failed)")
echo "Master resolution:"
(getent hosts "$MASTER_ADDR" 2>/dev/null | sed 's/^/  /' || echo "  (could not resolve $MASTER_ADDR)")
echo "Addresses on the pinned interface ($NCCL_SOCKET_IFNAME):"
(cat /sys/class/net/$NCCL_SOCKET_IFNAME/address 2>/dev/null | sed 's/^/  MAC: /' || true)
(awk -v iface="$NCCL_SOCKET_IFNAME" '$2 == iface {print "  v4 hex: " $3}' /proc/net/route 2>/dev/null || true)
echo "=========================================="
echo "Starting torchrun at $(date)"
echo "=========================================="

torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    /opt/megatron-lm/pretrain_gpt.py \
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
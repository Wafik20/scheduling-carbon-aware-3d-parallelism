#!/bin/bash
# convert_checkpoints_1b_bridge.sh
# Converts HF weights to Megatron format for all TP/PP configs
# Uses Megatron-Bridge's multi-GPU roundtrip script
# Usage: bash convert_checkpoints_1b_bridge.sh

MODEL_ID=/u/wat29/llama-3.2-1B
SAVE_BASE=/u/wat29/llama-3.2-1B-megatron
CONTAINER=/u/wat29/nemo.sif
SCRIPT=/opt/Megatron-Bridge/examples/conversion/hf_megatron_roundtrip_multi_gpu.py

for config in "1 1" "2 2" "4 1" "1 4" "2 1" "1 2"; do
    TP=$(echo $config | cut -d' ' -f1)
    PP=$(echo $config | cut -d' ' -f2)
    NPROC=$((TP * PP))

    echo "=================================================="
    echo "  Converting: TP=${TP} PP=${PP} (${NPROC} GPUs)"
    echo "=================================================="

    apptainer exec --nv \
        --bind /u/wat29:/u/wat29 \
        ${CONTAINER} \
        torchrun --nproc_per_node=${NPROC} ${SCRIPT} \
            --hf-model-id ${MODEL_ID} \
            --tp ${TP} \
            --pp ${PP} \
            --megatron-save-path ${SAVE_BASE}-tp${TP}pp${PP}

    echo "  Done: TP=${TP} PP=${PP} -> ${SAVE_BASE}-tp${TP}pp${PP}"
done

echo "All conversions complete."
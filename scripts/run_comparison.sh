#!/bin/bash
set -uo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON=/bgfs/stephenlee/wat29/Megatron-LM
cd "$PROJECT"

# Common args for all three runs
COMMON_ARGS=(
    --model gpt2_medium
    --carbon-forecast  "$PROJECT/data/carbon_synthetic_dataset.csv"
    --num-gpus         4
    --container-image  "$CONTAINER"
    --container-bind   "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29"
    --container-bind   "$MEGATRON:/opt/megatron-lm"
    --train-iters      100000
    --switch-power-w   200
    --deadline-seconds 14400
    --lookahead-steps  8000
    --decide-every     1000
    --bucket-seconds   600
)

# Run each variant. Format: tag:calibration_filename
for spec in \
    "greedy_v1:calibration_gpt2_medium_4gpu" \
    "fast_v1:static_fast_gpt2_medium_4gpu" \
    "slow_v1:static_slow_gpt2_medium_4gpu" ; do

    tag="${spec%:*}"
    cal="${spec#*:}"
    RUN_DIR="$PROJECT/runs/$tag"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "============================================================"
    echo "  STARTING $tag  ($cal)  at $(date)"
    echo "============================================================"

    python train_online_elastic.py \
        --scheduler-config "$PROJECT/calibrations/${cal}.json" \
        --work-dir         "$RUN_DIR" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "$RUN_DIR/run.log"

    rc=${PIPESTATUS[0]}
    echo ""
    echo "============================================================"
    echo "  $tag finished at $(date) (rc=$rc)"
    echo "============================================================"

    if [ $rc -ne 0 ]; then
        echo "  $tag FAILED, stopping the comparison."
        exit $rc
    fi
done

echo ""
echo "============================================================"
echo "  ALL THREE RUNS COMPLETE"
echo "============================================================"
echo "  Greedy:      $PROJECT/runs/greedy_v1/"
echo "  Static fast: $PROJECT/runs/fast_v1/"
echo "  Static slow: $PROJECT/runs/slow_v1/"
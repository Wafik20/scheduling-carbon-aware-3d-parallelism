#!/bin/bash
# run_large_benchmarks.sh — full baseline + hybrid sweep for GPT-2 Large

MODEL=gpt2_large
SEQ=2048
MBS=1
GBS=16

SCHEDULES=(
  "tp1_pp1_dp4"
  "tp4_pp1_dp1"
  "tp1_pp4_dp1"
  "tp2_pp2_dp1"
  "tp2_pp1_dp2"
  "tp1_pp2_dp2"
  "example_schedule"
)

for sched in "${SCHEDULES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Running: $sched  (model=$MODEL, seq=$SEQ)"
    echo "=========================================="
    python train_elastic.py \
        --model $MODEL \
        --schedule schedules/${sched}.json \
        --seq-length $SEQ \
        --micro-batch-size $MBS \
        --global-batch-size $GBS
    sleep 10  # let GPU state settle between runs
done

echo ""
echo "All runs complete. CSVs in power_logs/${MODEL}/"
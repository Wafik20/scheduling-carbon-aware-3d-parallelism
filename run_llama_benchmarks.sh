#!/bin/bash
# run_llama_benchmarks.sh — full baseline + hybrid sweep across LLaMA family
#
# Runs each model × each schedule = 3 × 7 = 21 total runs. Between every
# run (and before the first), calls check_cluster_users.py to log who
# else is on the GPUs. Contamination is detect-only — we never abort.
#
# Contamination definition (enforced in check_cluster_users.py):
#   another user with a process on a GPU whose utilization is >5%.
#   This filters out idle CUDA contexts and our own monitoring handles.
#
# Expected wall time: ~6-7 hours (overnight-safe).

set -u

MODELS=(
    "llama_1b"
    "llama_2b"
    "llama_3b"
)

SCHEDULES=(
    "tp1_pp1_dp4"
    "tp4_pp1_dp1"
    "tp1_pp4_dp1"
    "tp2_pp2_dp1"
    "tp2_pp1_dp2"
    "tp1_pp2_dp2"
    "example_schedule"
)

SEQ=256
MBS=1
GBS=16

CONTAM_LOG="contamination_log.csv"
RUN_MANIFEST="run_manifest.csv"   # one row per run with start/end + pre-check/post-check refs

# Reset logs at the start of a fresh sweep so we don't mix with prior runs.
# Comment out if you want to preserve across sweep invocations.
: > "$CONTAM_LOG"
echo "model,schedule,start_ts,end_ts,duration_s,exit_code,pre_check_contam,post_check_contam" > "$RUN_MANIFEST"

# Helper: run one contamination check with a label, and return whether
# it flagged contamination ("yes"/"no") for the manifest.
# We call this BETWEEN runs, not during — so it's cheap and never races.
run_check() {
    local tag="$1"
    python check_cluster_users.py \
        --my-user "$USER" \
        --log "$CONTAM_LOG" \
        --tag "$tag"
    local rc=$?
    if [ $rc -eq 1 ]; then
        echo "yes"
    else
        echo "no"
    fi
}

# ── Sweep ───────────────────────────────────────────────────
PASSED=()
FAILED=()
CONTAM_RUNS=()   # runs with contamination in either pre or post check
SWEEP_START=$(date +%s)

echo "[sweep] Starting at $(date). Initial cluster check..."
INITIAL_CONTAM=$(run_check "sweep_start")

for model in "${MODELS[@]}"; do
    for sched in "${SCHEDULES[@]}"; do
        echo ""
        echo "=================================================="
        echo "  [$(date '+%H:%M:%S')]  $model / $sched"
        echo "=================================================="

        rm -rf /u/data3/wat29/carbon_training/elastic/elastic_ckpt

        # Pre-run check: log cluster state right before we launch.
        # If this is "yes", the run's baseline power will be inflated
        # from the moment it starts.
        pre_contam=$(run_check "before_${model}_${sched}")

        start_ts=$(date -u +"%Y-%m-%dT%H:%M:%S")
        run_start=$(date +%s)

        python train_elastic.py \
            --model "$model" \
            --schedule "schedules/${sched}.json" \
            --seq-length $SEQ \
            --micro-batch-size $MBS \
            --global-batch-size $GBS
        rc=$?

        run_end=$(date +%s)
        end_ts=$(date -u +"%Y-%m-%dT%H:%M:%S")
        run_duration=$((run_end - run_start))

        # Post-run check: did contamination appear DURING the run?
        # If pre=no but post=yes, someone joined mid-run and the power
        # CSV for this run is partially polluted. If pre=yes and
        # post=no, they left mid-run (partially polluted). If both yes,
        # the whole run was polluted. Both no = probably clean.
        post_contam=$(run_check "after_${model}_${sched}")

        # Write manifest row. Power-CSV analysis joins on this to flag
        # runs that need an asterisk in the paper.
        echo "${model},${sched},${start_ts},${end_ts},${run_duration},${rc},${pre_contam},${post_contam}" >> "$RUN_MANIFEST"

        tag="${model}/${sched}"
        if [ $rc -eq 0 ]; then
            PASSED+=("$tag (${run_duration}s)")
            echo "  [ok] $tag finished in ${run_duration}s"
        else
            FAILED+=("$tag (exit=$rc, ${run_duration}s)")
            echo "  [FAIL] $tag exited with code $rc after ${run_duration}s"
        fi

        if [ "$pre_contam" = "yes" ] || [ "$post_contam" = "yes" ]; then
            CONTAM_RUNS+=("$tag (pre=${pre_contam}, post=${post_contam})")
            echo "  [⚠ contam] $tag has contamination in pre or post check"
        fi

        sleep 10
    done
done

SWEEP_END=$(date +%s)
TOTAL_MIN=$(( (SWEEP_END - SWEEP_START) / 60 ))

echo ""
echo "[sweep] Final cluster check..."
FINAL_CONTAM=$(run_check "sweep_end")

# ── Summary ─────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  SWEEP COMPLETE"
echo "=================================================="
echo "  Total wall time:     ${TOTAL_MIN} min"
echo "  Runs passed:         ${#PASSED[@]}/$((${#MODELS[@]} * ${#SCHEDULES[@]}))"
echo "  Runs failed:         ${#FAILED[@]}"
echo "  Runs w/ contamination: ${#CONTAM_RUNS[@]}  (pre or post check flagged)"
echo ""

if [ ${#CONTAM_RUNS[@]} -gt 0 ]; then
    echo "  Contaminated runs — treat power CSVs with caution:"
    for x in "${CONTAM_RUNS[@]}"; do echo "    ⚠ $x"; done
    echo ""
fi

if [ ${#PASSED[@]} -gt 0 ]; then
    echo "  Succeeded:"
    for x in "${PASSED[@]}"; do echo "    ✓ $x"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "  Failed:"
    for x in "${FAILED[@]}"; do echo "    ✗ $x"; done
fi

echo ""
echo "  Artifacts:"
echo "    Run manifest:      $RUN_MANIFEST"
echo "    Contamination log: $CONTAM_LOG"
echo "    Power CSVs:        power_logs/{llama_1b,llama_2b,llama_3b}/"
echo "=================================================="

exit 0
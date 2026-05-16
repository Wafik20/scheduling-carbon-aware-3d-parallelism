#!/bin/bash
#
# Naive scheduler hybrid + 2 static baselines, LOCAL variant for tintin.
#
# Runs three configurations sequentially against the same carbon trace
# and the same TRAIN_ITERS, so total CO2 numbers are directly comparable:
#
#   1. naive_hybrid : LOW=DP4/TP1/PP1, MED=DP2/TP2/PP1, HIGH=DP1/TP4/PP1
#   2. static_a     : DP=4/TP=1/PP=1   (same as naive's LOW config)
#   3. static_b     : DP=2/TP=1/PP=2   (mixed pipeline + data parallel)
#
# Static baselines invoke train_naive_elastic.py with all three bucket
# configs set to the same (TP, PP, DP). The scheduler runs every
# decide_every steps but `next_strat.key == current.key` always holds,
# so it never switches — yielding a clean single-phase run with output
# format identical to the hybrid run, which makes the comparison plot
# easy to write.
#
# All three runs share a COMPARISON_TAG so downstream visualization can
# group them. Override knobs via env, e.g.:
#   TRAIN_ITERS=3000 MIN_DWELL_STEPS=150 ./scripts/run_hybrid_local.sh
#
set -euo pipefail

PROJECT="${HOME}/scheduling-carbon-aware-3d-parallelism"
MEGATRON=/u/data3/wat29/Megatron-LM
CONTAINER=/u/data3/wat29/nemo.sif

command -v apptainer >/dev/null || {
  echo "[fatal] apptainer not in PATH"; exit 2;
}

cd "$PROJECT"
mkdir -p runs logs

COMMON_BINDS=(
  --container-image "$CONTAINER"
  --container-bind  "$PROJECT:$PROJECT"
  --container-bind  "/u/data3/wat29:/u/data3/wat29"
  --container-bind  "$MEGATRON:/opt/megatron-lm"
)

# ── Knobs (override via env) ─────────────────────────────────
MODEL=${MODEL:-gpt2_medium}
# 5000 steps × ~0.4s/step ≈ 33 min training/run. Three runs = ~100 min.
# Bump up if you've fetched even more carbon data and want more variation.
TRAIN_ITERS=${TRAIN_ITERS:-5000}
DECIDE_EVERY=${DECIDE_EVERY:-50}
# With ~720 CSV rows over 5000 steps (1 row ≈ 7 steps), 200-step dwell
# smooths over ~28 rows ≈ 1.2 days of carbon — captures the daily cycle
# without flipping on every diurnal swing.
MIN_DWELL_STEPS=${MIN_DWELL_STEPS:-200}

# Naive scheduler bucket configs. All three must satisfy TP*PP*DP == NUM_GPUS.
LOW_TP=${LOW_TP:-1};   LOW_PP=${LOW_PP:-1};   LOW_DP=${LOW_DP:-4}
MED_TP=${MED_TP:-2};   MED_PP=${MED_PP:-1};   MED_DP=${MED_DP:-2}
HIGH_TP=${HIGH_TP:-4}; HIGH_PP=${HIGH_PP:-1}; HIGH_DP=${HIGH_DP:-1}

# Static baseline configs.
STATIC_A_TP=${STATIC_A_TP:-1}; STATIC_A_PP=${STATIC_A_PP:-1}; STATIC_A_DP=${STATIC_A_DP:-4}
STATIC_B_TP=${STATIC_B_TP:-1}; STATIC_B_PP=${STATIC_B_PP:-2}; STATIC_B_DP=${STATIC_B_DP:-2}

NUM_GPUS=${NUM_GPUS:-4}

# Checkpoint scratch dir on fast local storage. Wiped before each run.
ELASTIC_CKPT_DIR=${ELASTIC_CKPT_DIR:-/tmp/wat29_elastic_ckpt_scratch}

CARBON_CSV="$PROJECT/data/carbon_week.csv"

[[ -f "$CARBON_CSV" ]] || { echo "[fatal] missing $CARBON_CSV"; exit 2; }
[[ -f "$CONTAINER"  ]] || { echo "[fatal] missing $CONTAINER"; exit 2; }
[[ -d "$MEGATRON"   ]] || { echo "[fatal] missing $MEGATRON"; exit 2; }

# Quick CSV row count for the banner.
CARBON_ROWS=$(($(wc -l < "$CARBON_CSV") - 1))

# Shared identifier so visualization can group the three runs.
COMPARISON_TAG="local_$(date +%Y%m%d_%H%M%S)"

echo "════════════════════════════════════════════════════════════════"
echo "  3-CONFIG COMPARISON RUN"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:           $MODEL"
echo "  Train iters:     $TRAIN_ITERS"
echo "  Decide every:    $DECIDE_EVERY  steps"
echo "  Min dwell:       $MIN_DWELL_STEPS  steps"
echo "  Carbon CSV:      $CARBON_CSV  ($CARBON_ROWS hourly rows)"
echo "  Comparison tag:  $COMPARISON_TAG"
echo "  Ckpt scratch:    $ELASTIC_CKPT_DIR  (wiped per-run)"
echo "  Configs:"
echo "    naive   LOW=tp${LOW_TP}/pp${LOW_PP}/dp${LOW_DP}  MED=tp${MED_TP}/pp${MED_PP}/dp${MED_DP}  HIGH=tp${HIGH_TP}/pp${HIGH_PP}/dp${HIGH_DP}"
echo "    static_a  tp${STATIC_A_TP}/pp${STATIC_A_PP}/dp${STATIC_A_DP}"
echo "    static_b  tp${STATIC_B_TP}/pp${STATIC_B_PP}/dp${STATIC_B_DP}"
echo "════════════════════════════════════════════════════════════════"
echo

run_one() {
  local kind="$1"
  local lt="$2" lp="$3" ld="$4"
  local mt="$5" mp="$6" md="$7"
  local ht="$8" hp="$9" hd="${10}"

  local run_dir="$PROJECT/runs/${kind}_${COMPARISON_TAG}"
  local logfile="$PROJECT/logs/${kind}-${COMPARISON_TAG}.out"
  mkdir -p "$run_dir"

  rm -rf "$ELASTIC_CKPT_DIR"
  mkdir -p "$ELASTIC_CKPT_DIR"

  local t0
  t0=$(date +%s)

  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  RUN: $kind   start=$(date +%H:%M:%S)"
  echo "════════════════════════════════════════════════════════════════"
  echo "  RUN_DIR=$run_dir"
  echo "  LOGFILE=$logfile"
  echo

  # tee both stdout+stderr to the run-specific logfile so each run has
  # its own clean log without per-line overhead from a global redirect.
  python train_naive_elastic.py \
    --model            "$MODEL" \
    --carbon-forecast  "$CARBON_CSV" \
    --train-iters      "$TRAIN_ITERS" \
    --decide-every     "$DECIDE_EVERY" \
    --min-dwell-steps  "$MIN_DWELL_STEPS" \
    --elastic-ckpt-dir "$ELASTIC_CKPT_DIR" \
    --low-tp           "$lt" --low-pp  "$lp" --low-dp  "$ld" \
    --med-tp           "$mt" --med-pp  "$mp" --med-dp  "$md" \
    --high-tp          "$ht" --high-pp "$hp" --high-dp "$hd" \
    --num-gpus         "$NUM_GPUS" \
    --work-dir         "$run_dir" \
    "${COMMON_BINDS[@]}" \
    2>&1 | tee "$logfile"

  local elapsed=$(( $(date +%s) - t0 ))
  echo
  echo "[$kind] DONE   wall=${elapsed}s   end=$(date +%H:%M:%S)"
}

# ── Run 1: naive scheduler (real LOW/MED/HIGH) ───────────────
run_one "naive_hybrid" \
  "$LOW_TP"  "$LOW_PP"  "$LOW_DP"  \
  "$MED_TP"  "$MED_PP"  "$MED_DP"  \
  "$HIGH_TP" "$HIGH_PP" "$HIGH_DP"

# ── Run 2: static A ─ all three buckets = STATIC_A config ────
run_one "static_a" \
  "$STATIC_A_TP" "$STATIC_A_PP" "$STATIC_A_DP" \
  "$STATIC_A_TP" "$STATIC_A_PP" "$STATIC_A_DP" \
  "$STATIC_A_TP" "$STATIC_A_PP" "$STATIC_A_DP"

# ── Run 3: static B ─ all three buckets = STATIC_B config ────
run_one "static_b" \
  "$STATIC_B_TP" "$STATIC_B_PP" "$STATIC_B_DP" \
  "$STATIC_B_TP" "$STATIC_B_PP" "$STATIC_B_DP" \
  "$STATIC_B_TP" "$STATIC_B_PP" "$STATIC_B_DP"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  ALL THREE RUNS COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  tag:        $COMPARISON_TAG"
echo "  naive:      $PROJECT/runs/naive_hybrid_${COMPARISON_TAG}/"
echo "  static_a:   $PROJECT/runs/static_a_${COMPARISON_TAG}/"
echo "  static_b:   $PROJECT/runs/static_b_${COMPARISON_TAG}/"
echo "  logs:       $PROJECT/logs/*-${COMPARISON_TAG}.out"
echo "════════════════════════════════════════════════════════════════"
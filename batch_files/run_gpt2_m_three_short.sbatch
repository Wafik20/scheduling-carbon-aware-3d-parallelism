#!/bin/bash
#SBATCH --job-name=static_comparison_gpt2_medium
#SBATCH -M gpu
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
cd "$PROJECT"
mkdir -p slurm_logs

module load apptainer

# Run three static strategies sequentially. Each gets its own work-dir
# so checkpoints from one run can't accidentally be loaded by the next
# (which would skip the fresh-build path and break the comparison with
# the simulator). Each also gets its own MASTER_PORT — paranoia in case
# the default 29500 is held in TIME_WAIT from the previous call.
#
# 200 iters per strategy. At global_batch=16, seq_length=1024 that's
# 3.28M tokens — short enough to finish quickly, long enough to dwarf
# the per-iter warmup (~21-37 iters) so steady-state dominates.
#
# Expected wall per strategy on 4× A100 NVLink:
#   (1,1):  ~38s training +  ~30s overhead ≈ 1.1 min
#   (2,1):  ~83s training +  ~30s overhead ≈ 1.9 min
#   (4,1): ~163s training +  ~30s overhead ≈ 3.2 min
# Total ≈ 6-7 min.

N_ITERS=200
BASE_PORT=$(( 25000 + SLURM_JOB_ID % 5000 ))

run_strategy() {
    local TP=$1
    local PP=$2
    local PORT_OFFSET=$3

    local TAG="tp${TP}_pp${PP}"
    local PORT=$(( BASE_PORT + PORT_OFFSET ))
    local SCHEDULE="$PROJECT/_static_${TAG}.json"
    local WORKDIR="/bgfs/stephenlee/wat29/static_comparison_workdir_${TAG}"

    # Write the single-phase schedule inline, then clean it up after.
    cat > "$SCHEDULE" <<EOF
[{"tp": ${TP}, "pp": ${PP}, "start_step": 0, "end_step": ${N_ITERS}}]
EOF

    rm -rf "$WORKDIR"
    mkdir -p "$WORKDIR"

    echo "=========================================================="
    echo "  STATIC RUN: $TAG  (port $PORT, ${N_ITERS} iters)"
    echo "=========================================================="

    MASTER_PORT=$PORT python train_elastic.py \
        --model gpt2_medium \
        --schedule "$SCHEDULE" \
        --num-gpus 4 \
        --micro-batch-size 2 \
        --global-batch-size 16 \
        --container-image /bgfs/stephenlee/wat29/nemo_2504.sif \
        --container-bind /bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29 \
        --container-bind /bgfs/stephenlee/wat29/Megatron-LM:/opt/megatron-lm \
        --megatron-dir /opt/megatron-lm \
        --work-dir "$WORKDIR"

    rm -f "$SCHEDULE"
    echo "[done] $TAG"
    echo
}

run_strategy 1 1 0
run_strategy 2 1 1
run_strategy 4 1 2

echo "All three static runs complete."
echo "Power CSVs in:  $PROJECT/power_logs/gpt2_medium/"
echo "Training logs:  /bgfs/stephenlee/wat29/static_comparison_workdir_*/elastic_*.log"
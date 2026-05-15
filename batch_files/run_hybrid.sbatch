#!/bin/bash
#SBATCH --job-name=hybrid_run_gpt2_medium
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

# Hybrid elastic run: 80 iters @ (1,1) → 60 @ (2,1) → 60 @ (4,1).
# Matches HYBRID_SCHEDULE in the simulator notebook exactly. Same
# hyperparams (mbs=2, gbs=16, seq=1024, mock-data) as the three static
# comparison runs so actuals are directly diffable against:
#   - the simulator's predicted hybrid trace
#   - the three static actuals (elastic__static_tp{1,2,4}_pp1_power.csv)

MASTER_PORT=$(( 25000 + SLURM_JOB_ID % 5000 ))

TAG="hybrid_v1"
SCHEDULE="$PROJECT/_${TAG}.json"
WORKDIR="/bgfs/stephenlee/wat29/hybrid_run_workdir_${TAG}"

cat > "$SCHEDULE" <<'EOF'
[
  {"tp": 1, "pp": 1, "start_step":   0, "end_step":  80},
  {"tp": 2, "pp": 1, "start_step":  80, "end_step": 140},
  {"tp": 4, "pp": 1, "start_step": 140, "end_step": 200}
]
EOF

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

echo "=========================================================="
echo "  HYBRID RUN: $TAG  (port $MASTER_PORT)"
echo "=========================================================="
echo "  Schedule: $SCHEDULE"
echo "  Workdir:  $WORKDIR"
cat "$SCHEDULE"
echo

# elastic_pretrain.py writes checkpoints to /scratch/slurm-$SLURM_JOB_ID/...
# inside the container, which is SLURM's per-job scratch dir on the compute
# node. That path is writable on the host but invisible to the container
# unless explicitly bound — apptainer doesn't pass /scratch through by
# default. Adding /scratch:/scratch to the bind list exposes it, the
# default checkpoint path then resolves correctly, and reshards work.
MASTER_PORT=$MASTER_PORT python train_elastic.py \
    --model gpt2_medium \
    --schedule "$SCHEDULE" \
    --num-gpus 4 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --container-image /bgfs/stephenlee/wat29/nemo_2504.sif \
    --container-bind /bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29 \
    --container-bind /bgfs/stephenlee/wat29/Megatron-LM:/opt/megatron-lm \
    --container-bind /scratch:/scratch \
    --megatron-dir /opt/megatron-lm \
    --work-dir "$WORKDIR"

rm -f "$SCHEDULE"

echo
echo "Done. Power CSV:"
echo "  $PROJECT/power_logs/gpt2_medium/elastic__${TAG}_power.csv"
echo "Training log:"
echo "  $WORKDIR/elastic__${TAG}.log"
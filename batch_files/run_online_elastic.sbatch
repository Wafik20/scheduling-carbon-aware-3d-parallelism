#!/bin/bash
#SBATCH --job-name=online-elastic
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100_nvlink
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out

# End-to-end test of the online (greedy carbon-aware) elastic scheduler on
# a single L40 node. Drives train_online_elastic.py, which in turn launches
# online_elastic_pretrain.py via torchrun inside the Apptainer container.
#
# Single-node: no MASTER_ADDR / NCCL networking env needed — torchrun
# handles intra-node collectives over NVLink/PCIe.
set -euo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
module purge
module load apptainer

cd "$PROJECT"
mkdir -p slurm_logs runs

CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON=/bgfs/stephenlee/wat29/Megatron-LM

# Per-run work dir under the project so each invocation gets its own
# checkpoints / phases.json / elastic_results.json.
RUN_TAG="${RUN_TAG:-online_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"
RUN_DIR="$PROJECT/runs/$RUN_TAG"
mkdir -p "$RUN_DIR"

echo "[sbatch] PROJECT=$PROJECT"
echo "[sbatch] CONTAINER=$CONTAINER"
echo "[sbatch] MEGATRON=$MEGATRON"
echo "[sbatch] RUN_DIR=$RUN_DIR"
echo "[sbatch] node=$(hostname)  date=$(date)"
echo "[sbatch] forwarding args: $@"

# All positional args after the sbatch script name flow through to the
# launcher via "$@". Typical usage:
#   sbatch run_online_elastic.sbatch \
#       --train-iters 200 \
#       --switch-power-w 200 \
#       --deadline-seconds 180 \
#       --lookahead-steps 100 \
#       --decide-every 50
python train_online_elastic.py \
    --model gpt2_medium \
    --scheduler-config "$PROJECT/calibrations/calibration_gpt2_medium_4gpu.json" \
    --carbon-forecast  "$PROJECT/data/carbon_synthetic_dataset.csv" \
    --num-gpus         4 \
    --container-image  "$CONTAINER" \
    --container-bind   "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29" \
    --container-bind   "$MEGATRON:/opt/megatron-lm" \
    --work-dir         "$RUN_DIR" \
    "$@"

echo "[sbatch] done.  RUN_DIR=$RUN_DIR"
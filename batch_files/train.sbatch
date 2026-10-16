#!/bin/bash
#SBATCH --job-name=carbon-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail

# Submit from the repository root. Supply site account/partition with sbatch.
# Example: sbatch batch_files/train.sbatch --model gpt2 --tp 1 --pp 1 \
#   --container-image /path/to/nemo.sif \
#   --container-bind /path/to/Megatron-LM:/opt/megatron-lm
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
python train.py --num-gpus "${NUM_GPUS:-4}" "$@"

#!/bin/bash
#SBATCH --job-name=profile_parallel_configs
#SBATCH -M gpu
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



set -euo pipefail

PROJECT=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
cd "$PROJECT"
mkdir -p slurm_logs

module load apptainer

python train_profile.py \
    --model gpt2 \
    --profile-menu "$PROJECT/profile_menu.json" \
    --output       "$PROJECT/calibrations/calibration_gpt2_medium_4gpu.json" \
    --num-gpus 4 \
    --warmup-seconds 30 \
    --measure-seconds 60 \
    --container-image /bgfs/stephenlee/wat29/nemo_2504.sif

echo "Profile job complete (rc=$?)."
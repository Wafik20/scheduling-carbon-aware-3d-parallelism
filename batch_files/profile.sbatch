#!/usr/bin/env bash
#SBATCH --job-name=carbon-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-profile-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the scheduler repository root}"
export POWER_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
# Clear step memory options; the existing job memory allocation is unchanged.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
exec bash scripts/run_with_power.sh   srun --overlap --exact --ntasks=1 --cpu-bind=none --gpu-bind=none   bash scripts/run_container.sh train_profile.py --in-container "$@"

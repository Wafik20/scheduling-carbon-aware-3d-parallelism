#!/usr/bin/env bash
#SBATCH --job-name=carbon-profile-4gpu
#SBATCH --clusters=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --account=cs2731_2026f
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-profile-multinode-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the scheduler repository root}"
[[ ${SLURM_NTASKS:?} == 4 ]] || { echo "This wrapper requires four tasks" >&2; exit 1; }
# Bash ignores the shared launcher's SBATCH defaults; this allocation stays 2x2.
exec bash batch_files/profile_multinode.sbatch "$@"

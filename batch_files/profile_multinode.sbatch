#!/usr/bin/env bash
#SBATCH --job-name=carbon-profile-multinode
#SBATCH --clusters=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --account=cs2731_2026f
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-profile-multinode-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the scheduler repository root}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if (( ${#nodes[@]} != 2 )) || [[ ${SLURM_NTASKS:?} != 2 && ${SLURM_NTASKS} != 4 && ${SLURM_NTASKS} != 8 ]]; then
    echo "This profiler requires two nodes and two, four, or eight tasks (one per GPU)" >&2
    exit 1
fi
world_size=$SLURM_NTASKS
gpus_per_node=$((world_size / 2))
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export POWER_GPUS_PER_NODE=$gpus_per_node
export PROFILE_TASK_GPUS=1
# One namespace per invocation, shared by all tasks; never consume stale mappings.
export PROFILE_MAPPING_RUN=$(cat /proc/sys/kernel/random/uuid)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Clear conflicting step memory options, not the job's memory allocation.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# Match the tested built-in IB smoke path; never inherit a stale HCA filter.
unset NCCL_IB_HCA APPTAINERENV_NCCL_IB_HCA
export APPTAINERENV_SMOKE_NCCL_IB_HCA="${SMOKE_NCCL_IB_HCA:-}"
export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_COLLNET_ENABLE=0
for name in NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_COLLNET_ENABLE; do
    export "APPTAINERENV_${name}=${!name}"
done

# Bind one CUDA-visible GPU per task, but allow NCCL to access allocated peer
# devices through the cgroup. Do not replace task binding with --gpu-bind=none.
# Each rank raises memlock before Apptainer; --slurm also checks it inside.
exec bash scripts/run_with_power.sh \
    srun --overlap --exact --export=ALL --kill-on-bad-exit=1 --cpu-bind=none --distribution=block \
    --nodes=2 --ntasks="$world_size" --ntasks-per-node="$gpus_per_node" --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    --gpus-per-node="$gpus_per_node" --gpus-per-task=1 --gpu-bind=single:1 --gres-flags=allow-task-sharing \
    bash -c 'ulimit -Sl unlimited || { echo "Cannot raise soft memlock on $(hostname)" >&2; exit 1; }; exec "$@"' \
    memlock-task bash scripts/run_container.sh train_profile.py \
    --in-container --slurm --num-gpus "$world_size" "$@"

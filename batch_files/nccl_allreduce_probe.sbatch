#!/usr/bin/env bash
#SBATCH --job-name=nccl-allreduce-probe
#SBATCH --clusters=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --account=cs2731_2026f
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-nccl-allreduce-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
[[ ${#nodes[@]} == 2 && ${SLURM_NTASKS:?} == 2 ]] || {
    echo "This probe requires two nodes and two tasks" >&2; exit 1;
}
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export POWER_GPUS_PER_NODE=1 PROFILE_TASK_GPUS=1
export PROFILE_MAPPING_RUN=$(cat /proc/sys/kernel/random/uuid)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
unset NCCL_IB_HCA APPTAINERENV_NCCL_IB_HCA
export APPTAINERENV_SMOKE_NCCL_IB_HCA="${SMOKE_NCCL_IB_HCA:-}"
# Same transport settings as the profiler; configure() reapplies inside Apptainer.
export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_COLLNET_ENABLE=0
for name in NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_COLLNET_ENABLE; do
    export "APPTAINERENV_${name}=${!name}"
done
mkdir -p runs
export PROBE_HOST_DIR=$(mktemp -d "$PWD/runs/nccl_allreduce_${SLURM_JOB_ID}_XXXXXX")
echo "All-reduce artifacts: $PROBE_HOST_DIR"
# Direct Slurm files also retain failures before the Python supervisor starts.
# Unlike profiling, let the peer finish its timeout and save its own status.
exec bash scripts/run_with_power.sh \
    srun --overlap --exact --export=ALL --kill-on-bad-exit=0 --cpu-bind=none --distribution=block \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    --gpus-per-node=1 --gpus-per-task=1 --gpu-bind=single:1 --gres-flags=allow-task-sharing \
    --output="$PROBE_HOST_DIR/launch-rank%t.out" --error="$PROBE_HOST_DIR/launch-rank%t.err" \
    bash -c 'scontrol show job -dd "$SLURM_JOB_ID" > "$PROBE_HOST_DIR/job-rank$SLURM_PROCID.txt" 2>&1; scontrol show step -dd "$SLURM_JOB_ID.$SLURM_STEP_ID" > "$PROBE_HOST_DIR/step-rank$SLURM_PROCID.txt" 2>&1; scontrol show config > "$PROBE_HOST_DIR/config-rank$SLURM_PROCID.txt" 2>&1; ulimit -Sl unlimited || exit 1; exec "$@"' \
    probe-task bash scripts/run_container.sh scripts/nccl_allreduce_probe.py \
    --result-dir "/workspace/project/${PROBE_HOST_DIR#"$PWD/"}" "$@"

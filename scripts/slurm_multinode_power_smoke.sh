#!/usr/bin/env bash
#SBATCH --job-name=multinode-power-smoke
#SBATCH --clusters=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --account=cs2731_2026f
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-power-smoke-%j.out
set -euo pipefail

# Slurm spools this script elsewhere; submit from the repository root.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
repo=$PWD
[[ -f scripts/run_with_power.sh && -f scripts/run_container.sh ]] || {
    echo "Submit from the repository root" >&2; exit 1;
}
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
world_size=${SLURM_NTASKS:?}
node_count=${#nodes[@]}
[[ $world_size =~ ^[1-9][0-9]*$ ]] && (( node_count > 0 && world_size % node_count == 0 )) || {
    echo "This smoke test requires a positive, uniform task count per node" >&2; exit 1;
}
gpus_per_node=$((world_size / node_count))
cpus_per_task=${SLURM_CPUS_PER_TASK:-4}
export MASTER_ADDR=${nodes[0]}
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export OMP_NUM_THREADS=$cpus_per_task
export POWER_INTERVAL=0.5
export POWER_GPUS_PER_NODE=$gpus_per_node
# Unique parent isolates this invocation from other runs in the same job.
mkdir -p "$repo/power_logs"
run=$(mktemp -d "$repo/power_logs/multinode_${SLURM_JOB_ID}_XXXXXX")
export POWER_OUTPUT_DIR="$run"
mkdir -p "$run/ranks"
duration=${SMOKE_DURATION:-90}
python=${POWER_PYTHON:-$repo/.venv/bin/python}
export POWER_PYTHON="$python"
echo "Smoke artifacts: $run"
# Persist expected topology independently of completion files for offline checks.
"$python" -c 'import json, sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(world_size=int(sys.argv[2]), gpus_per_node=int(sys.argv[3]), nodes=sys.argv[4:])))' \
    "$run/layout.json" "$world_size" "$gpus_per_node" "${nodes[@]}"


# The sampler step inherits all allocated GPUs/node. Only the training
# step requests one GPU/task; Slurm remaps each task's CUDA device to index 0.
if [[ ${SMOKE_NCCL_BUILTIN_IB:-0} == 1 ]]; then
    if [[ ${SMOKE_NCCL_SOCKET:-0} == 1 ]]; then
        echo "Built-in IB and Socket modes are mutually exclusive" >&2
        exit 1
    fi
    export SMOKE_UNLIMITED_MEMLOCK=1
fi
if [[ ${SMOKE_UNLIMITED_MEMLOCK:-0} == 1 && ${SMOKE_NCCL_SOCKET:-0} == 1 ]]; then
    echo "Memlock isolation test cannot be combined with Socket mode" >&2
    exit 1
fi
training_script=scripts/smoke_train_multinode.py
if [[ ${SMOKE_NCCL_SOCKET:-0} == 1 ]]; then
    # ib0 carried successful TCP bootstrap on both nodes in job 4449428.
    # This is TCP over IPoIB, not verbs/RDMA. Validate again on each rank.
    export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=1
    export NCCL_COLLNET_ENABLE=0 NCCL_DMABUF_ENABLE=0 NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME="=${SMOKE_SOCKET_IFNAME:-ib0}"
    for name in NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_COLLNET_ENABLE NCCL_DMABUF_ENABLE NCCL_DEBUG NCCL_SOCKET_IFNAME; do
        export "APPTAINERENV_${name}=${!name}"
    done
    training_script=scripts/nccl_socket_smoke.py
fi

training_command=(bash "$repo/scripts/run_container.sh" "$training_script")
if [[ ${SMOKE_UNLIMITED_MEMLOCK:-0} == 1 ]]; then
    # Executed by srun on EVERY training task, not in the batch shell.
    training_command=(bash -c 'ulimit -Sl unlimited || { echo "Cannot raise soft memlock on $(hostname)" >&2; exit 1; }; exec "$@"'
        memlock-task bash "$repo/scripts/run_container.sh" scripts/memlock_smoke.py)
fi

if [[ ${SMOKE_NCCL_BUILTIN_IB:-0} == 1 ]]; then
    export NCCL_NET=IB NCCL_NET_PLUGIN=none NCCL_IB_DISABLE=0 NCCL_IB_HCA="=mlx5_0:1"
    export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_COLLNET_ENABLE=0
    for name in NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_COLLNET_ENABLE; do
        export "APPTAINERENV_${name}=${!name}"
    done
    # Preserve the task-level memlock setup; replace only the container entrypoint.
    training_command[${#training_command[@]}-1]=scripts/nccl_builtin_ib_smoke.py
fi

bash scripts/run_with_power.sh \
    srun --overlap --exact --export=ALL --kill-on-bad-exit=1 --cpu-bind=none \
    --nodes="$node_count" --ntasks="$world_size" --ntasks-per-node="$gpus_per_node" --cpus-per-task="$cpus_per_task" \
    --gpus-per-node="$gpus_per_node" --gpus-per-task=1 --gpu-bind=single:1 \
    "${training_command[@]}" \
    --duration "$duration" \
    --result-dir "/workspace/project/${run#"$repo/"}/ranks"

"$python" scripts/validate_multinode_power_smoke.py "$run" --duration "$duration"

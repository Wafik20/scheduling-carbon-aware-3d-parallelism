#!/bin/bash
#SBATCH --job-name=online-elastic
#SBATCH --clusters=gpu
#SBATCH --account=cs2770_2026s
#SBATCH --partition=a100
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

set -uo pipefail

# ── Paths ────────────────────────────────────────────────────
PROJECT_DIR=/bgfs/stephenlee/wat29/scheduling-carbon-aware-3d-parallelism
CONTAINER=/bgfs/stephenlee/wat29/nemo_2504.sif
MEGATRON_HOST=/bgfs/stephenlee/wat29/Megatron-LM

# Inputs to the scheduler. Both files must already exist; the launcher
# fails fast if either is missing.
SCHEDULER_CONFIG="${PROJECT_DIR}/scheduler_config.json"
CARBON_FORECAST="${PROJECT_DIR}/data/carbon_synthetic_dataset.csv"

# ── Run config ───────────────────────────────────────────────
MODEL=gpt2_medium    # any key from MODEL_CONFIGS in train_online_elastic.py
TRAIN_ITERS=1000     # short for a smoke test; bump for real runs
GPUS_PER_NODE=4

RUN_NAME="online_${MODEL}_${SLURM_JOB_ID}"
WORK_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"

mkdir -p "${WORK_DIR}" slurm_logs
cd "${PROJECT_DIR}"

# ── Modules ──────────────────────────────────────────────────
module purge
module load apptainer

# ── Pre-flight checks ────────────────────────────────────────
# Cheap to verify here; the launcher also checks but failing in SLURM's
# stderr stream is more visible than failing five layers deep.
for f in "${SCHEDULER_CONFIG}" "${CARBON_FORECAST}" "${CONTAINER}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: missing required path: ${f}" >&2
        exit 2
    fi
done

# ── Banner ───────────────────────────────────────────────────
echo "============================================================"
echo "  Online elastic scheduler — single-node test"
echo "============================================================"
echo "Job:        $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "Node:       $SLURM_NODELIST"
echo "GPUs:       ${GPUS_PER_NODE}"
echo "Model:      ${MODEL}"
echo "Train iters: ${TRAIN_ITERS}"
echo "Scheduler:  ${SCHEDULER_CONFIG}"
echo "Forecast:   ${CARBON_FORECAST}"
echo "Container:  ${CONTAINER}"
echo "Megatron:   ${MEGATRON_HOST}"
echo "Work dir:   ${WORK_DIR}"
echo "============================================================"
echo

# ── Run the launcher ─────────────────────────────────────────
# train_online_elastic.py spawns apptainer internally with the binds we
# pass below. Override container-bind and megatron-dir to point at the
# /bgfs/stephenlee/wat29 layout this cluster uses (the launcher's
# defaults assume /u/data3/wat29 from the original site).
python "${PROJECT_DIR}/train_online_elastic.py" \
    --model "${MODEL}" \
    --scheduler-config "${SCHEDULER_CONFIG}" \
    --carbon-forecast "${CARBON_FORECAST}" \
    --train-iters "${TRAIN_ITERS}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --container-image "${CONTAINER}" \
    --container-bind "/bgfs/stephenlee/wat29:/bgfs/stephenlee/wat29" \
    --container-bind "${MEGATRON_HOST}:/opt/megatron-lm" \
    --megatron-dir /opt/megatron-lm \
    --work-dir "${WORK_DIR}"

RC=$?

# ── Footer ───────────────────────────────────────────────────
echo
echo "============================================================"
if [ $RC -eq 0 ]; then
    echo "  ONLINE ELASTIC TEST OK (rc=0)"
else
    echo "  ONLINE ELASTIC TEST FAILED (rc=$RC)"
fi
echo "============================================================"
echo "Phases:        ${WORK_DIR}/phases.json"
echo "Inner result:  ${WORK_DIR}/elastic_results.json"
echo "Power CSV:     ${PROJECT_DIR}/power_logs/${MODEL}/online_elastic_$(basename ${SCHEDULER_CONFIG} .json)_power.csv"
echo "Run log:       ${WORK_DIR}/online_elastic_$(basename ${SCHEDULER_CONFIG} .json).log"
echo

exit $RC
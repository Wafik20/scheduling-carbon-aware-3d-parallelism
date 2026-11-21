#!/usr/bin/env bash
# Resources are supplied by sweep.py submit; this entrypoint never submits jobs.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit through sweep.py from repository root}"
export SLURM_SUBMIT_DIR="$PWD"
exec "${SWEEP_PYTHON:-.venv/bin/python}" experiments/placement_sweep/sweep.py run-task \
    --manifest "${1:?manifest path required}" --task-index "${SLURM_ARRAY_TASK_ID:?array task required}"

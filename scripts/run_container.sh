#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_container.sh scripts/train.py [training arguments...]
# Override CONTAINER_IMAGE and MEGATRON_FORK with host paths if needed.
error() { printf 'run_container.sh: %s\n' "$*" >&2; exit 1; }
[[ $# -ge 1 ]] || error "Usage: $0 <repository Python script> [arguments...]"

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$REPO_ROOT"
CONTAINER_IMAGE=${CONTAINER_IMAGE:-/vast/cs2731_2026f/wat29/containers/nemo_26.08.sif}
MEGATRON_FORK=${MEGATRON_FORK:-/vast/cs2731_2026f/wat29/Megatron-LM}
CONTAINER_IMAGE=$(realpath -e -- "$CONTAINER_IMAGE") || error "Container path does not exist"
MEGATRON_FORK=$(realpath -e -- "$MEGATRON_FORK") || error "Megatron fork path does not exist"
TRAIN_SCRIPT=$(realpath -e -- "$1") || error "Training script does not exist: $1"
shift
[[ -f "$CONTAINER_IMAGE" && -r "$CONTAINER_IMAGE" ]] || error "Container is not a readable file: $CONTAINER_IMAGE"
[[ -d "$MEGATRON_FORK/megatron" ]] || error "Fork must contain a megatron directory: $MEGATRON_FORK"
[[ -f "$TRAIN_SCRIPT" && -r "$TRAIN_SCRIPT" && "$TRAIN_SCRIPT" == *.py ]] || error "Expected a readable Python script: $TRAIN_SCRIPT"
[[ "$TRAIN_SCRIPT" == "$REPO_ROOT/"* ]] || error "Training script must be inside $REPO_ROOT"
# Apptainer's bind syntax reserves these characters.
for path in "$REPO_ROOT" "$MEGATRON_FORK"; do
    [[ "$path" != *:* && "$path" != *,* ]] || error "Bind path cannot contain ':' or ',': $path"
done

if ! command -v apptainer >/dev/null 2>&1; then
    if ! type module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    fi
    type module >/dev/null 2>&1 || error "Apptainer is unavailable and environment modules could not be initialized"
    module load apptainer || error "Could not load the apptainer module"
fi
command -v apptainer >/dev/null 2>&1 || error "Apptainer is unavailable after module load"

# Prefer the mounted fork and project over host Python packages.
export APPTAINERENV_PYTHONPATH=/opt/Megatron-Bridge/3rdparty/Megatron-LM:/workspace/project:/workspace/project/src
export APPTAINERENV_PYTHONUNBUFFERED=1
exec apptainer exec --nv \
    --bind "$MEGATRON_FORK:/opt/Megatron-Bridge/3rdparty/Megatron-LM" \
    --bind "$REPO_ROOT:/workspace/project" \
    --pwd /workspace/project \
    "$CONTAINER_IMAGE" python \
    "/workspace/project/${TRAIN_SCRIPT#"$REPO_ROOT/"}" "$@"

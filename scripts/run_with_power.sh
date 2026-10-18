#!/usr/bin/env bash
# Run any command unchanged, with one power sampler per allocation node.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${POWER_PYTHON:-}" ]]; then
    python="$POWER_PYTHON"
elif [[ -x "$repo_dir/.venv/bin/python" ]]; then
    python="$repo_dir/.venv/bin/python"
else
    python=python3
fi
exec "$python" "$repo_dir/scripts/run_with_power.py" "$@"

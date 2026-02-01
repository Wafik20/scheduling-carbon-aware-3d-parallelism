#!/usr/bin/env bash
set -e

ENV_NAME="nx-dag"
YAML_FILE="networkx-dag.yaml"

echo "🔄 Reinstalling conda environment: $ENV_NAME"

# Make sure conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Activate conda first."
    exit 1
fi

# Remove env if it exists
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "🗑 Removing existing environment: $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y
fi

# Create environment
echo "📦 Creating environment from $YAML_FILE"
conda env create -f "$YAML_FILE"

echo "✅ Environment '$ENV_NAME' successfully reinstalled"
echo "👉 Activate with: conda activate $ENV_NAME"

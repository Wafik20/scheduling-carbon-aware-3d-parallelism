#!/bin/bash
# spin_env.sh — Set up gpu_env from scratch for MegatronLM training
#
# Prerequisites:
#   1. conda env create -f env.yaml
#   2. conda activate gpu_env
#   3. bash spin_env.sh
#
# Installation order matters:
#   1- PyTorch CUDA 12.8 via pip (matches system drivers + TE wheel)
#   2- pybind11 before MegatronLM source install
#   3- MegatronLM from source (no deps to avoid PyTorch upgrade)
#   4- Set CUDA 12.4 + NCCL paths (needed for Transformer Engine build)
#   5- Transformer Engine (prebuilt wheel for torch2.10.0+cu128)
#   6- Upgrade PyTorch to 2.10.0+cu128 to match TE wheel
#   7- Everything else last
#   8- Patch fused_softmax.py (APEX not available on this cluster)

set -e  # stop on first error

# ── Step 1: PyTorch ──────────────────────────────────────────────────
echo "==> Installing PyTorch with CUDA 12.8 via pip..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Verifying PyTorch + CUDA..."
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# ── Step 2: pybind11 ─────────────────────────────────────────────────
echo "==> Installing pybind11 (needed for MegatronLM build)..."
pip install pybind11

# ── Step 3: MegatronLM from source ───────────────────────────────────
echo "==> Installing MegatronLM from source (no deps)..."
pip install --no-build-isolation --no-deps -e Megatron-LM/

# ── Step 4: CUDA + NCCL paths ────────────────────────────────────────
echo "==> Setting CUDA 12.4 + NCCL paths..."
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "    CUDA_HOME=$CUDA_HOME"
echo "    nvcc: $(nvcc --version | grep release)"

# ── Step 5: Transformer Engine ───────────────────────────────────────
echo "==> Installing Transformer Engine..."
NVTE_FRAMEWORK=pytorch pip install transformer-engine[pytorch]

# ── Step 6: Upgrade PyTorch to match TE wheel ────────────────────────
# TE wheel was built for torch2.10.0+cu128 — must match exactly
echo "==> Upgrading PyTorch to 2.10.0+cu128 to match TE wheel..."
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

echo "==> Verifying PyTorch + TE compatibility..."
python3 -c "import torch; import transformer_engine; print('PyTorch:', torch.__version__); print('TE:', transformer_engine.__version__)"

# ── Step 7: Remaining dependencies ───────────────────────────────────
echo "==> Installing remaining dependencies..."
pip install nvidia-ml-py transformers nltk datasets pydantic pyyaml "numpy<2"

# ── Step 8: Patch fused_softmax.py ───────────────────────────────────
# APEX is not available on this cluster (no cuDNN headers for compilation)
# MegatronLM has a built-in PyTorch fallback — this patch activates it
echo "==> Patching fused_softmax.py to use PyTorch fallback (no APEX)..."
python3 -c "
with open('Megatron-LM/megatron/core/fusions/fused_softmax.py', 'r') as f:
    content = f.read()

# Make is_kernel_available always return False → triggers PyTorch fallback
old = '        \"\"\"\n        attn_batches = b * np'
new = '        \"\"\"\n        return False\n        attn_batches = b * np'

if old in content:
    content = content.replace(old, new, 1)
    with open('Megatron-LM/megatron/core/fusions/fused_softmax.py', 'w') as f:
        f.write(content)
    print('    fused_softmax.py patched successfully')
else:
    print('    fused_softmax.py already patched or structure changed — skipping')
"

# ── Final checks ──────────────────────────────────────────────────────
echo ""
echo "==> All done! Run checks:"
echo "    python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available())\""
echo "    python3 -c \"import transformer_engine; print('TE:', transformer_engine.__version__)\""
echo "    python3 -c \"from pynvml import *; nvmlInit(); print('GPUs:', nvmlDeviceGetCount()); nvmlShutdown()\""
echo "    python3 -c \"import megatron; print('Megatron OK')\""
echo ""
echo "==> Then set env vars before training:"
echo "    export CUDA_HOME=/usr/local/cuda-12.4"
echo "    export PATH=\$CUDA_HOME/bin:\$PATH"
echo "    export CUDA_DEVICE_MAX_CONNECTIONS=1"
echo "    python train.py --tp 1 --pp 1 --dp 4"
#!/bin/bash
# spin_env.sh — install PyTorch with CUDA 12.1, then install MegatronLM and transformer-engine from source
set -e  # stop on first error

echo "==> Installing PyTorch with CUDA 12.1 via pip..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "==> Verifying PyTorch + CUDA..."
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"

echo "==> Installing pybind11 first (needed for MegatronLM build)..."
pip install pybind11

echo "==> Installing MegatronLM from source..."
pip install --no-build-isolation -e Megatron-LM/

echo "==> Installing transformer-engine (requires CUDA 12 + PyTorch already installed)..."
pip install transformer-engine[pytorch]

echo "==> Installing remaining dependencies..."
pip install nvidia-ml-py transformers nltk datasets "numpy<2"

echo ""
echo "==> All done! Verify with:"
echo "    python3 -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\""
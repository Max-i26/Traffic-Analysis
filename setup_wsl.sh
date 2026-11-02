#!/usr/bin/env bash
# ==============================================================================
# Traffic Intelligence AI - WSL2 GPU Environment Setup Script
# For NVIDIA GeForce RTX 2050 (4 GB VRAM) on Ubuntu WSL2
# ==============================================================================

set -e

echo "=== 1. Checking GPU Passthrough in WSL2 ==="
if [ -f /usr/lib/wsl/lib/nvidia-smi ]; then
    /usr/lib/wsl/lib/nvidia-smi
else
    echo "nvidia-smi not found in /usr/lib/wsl/lib! Ensure NVIDIA Windows drivers are installed."
    exit 1
fi

echo "=== 2. Updating Ubuntu Packages and Installing Python Build Tools ==="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libgl1 libglib2.0-0 git

echo "=== 3. Navigating to Project Directory ==="
cd /mnt/e/Neural_Network_project

echo "=== 4. Creating Dedicated WSL Virtual Environment ==="
if [ ! -d "venv_wsl" ]; then
    python3 -m venv venv_wsl
    echo "Virtual environment 'venv_wsl' created."
fi

echo "=== 5. Activating Virtual Environment ==="
source venv_wsl/bin/activate

echo "=== 6. Installing PyTorch with CUDA 12.1 Acceleration ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== 7. Installing Computer Vision & Deep Learning Dependencies ==="
pip install opencv-python-headless pyyaml tqdm matplotlib scikit-learn albumentations

echo "=== 8. Verifying PyTorch CUDA Acceleration on RTX 2050 ==="
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Target GPU:', torch.cuda.get_device_name(0))
    print('VRAM Total (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print('GPU Tensor Test Passed! Allocated VRAM (MB):', round(torch.cuda.memory_allocated(0) / (1024**2), 2))
else:
    print('WARNING: CUDA is not active in PyTorch.')
"

echo "=== WSL2 GPU Environment Setup Complete! ==="

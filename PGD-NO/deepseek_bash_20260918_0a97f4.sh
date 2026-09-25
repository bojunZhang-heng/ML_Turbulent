#!/bin/bash
set -e

ENV_NAME="pgd-no"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Setting up conda environment: $ENV_NAME"
echo "=========================================="

if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 统一从 conda-forge 安装所有包，避免 OpenMP 冲突
echo "Installing PyTorch and dependencies from conda-forge..."
conda install -y -c conda-forge \
    pytorch torchvision torchaudio \
    numpy scipy pandas \
    vtk networkx \
    timm einops tqdm pyvista matplotlib

# 只有 conda-forge 没有的包才用 pip
# （当前项目全部有 conda 版本，所以这里不调用 pip）

echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')"
python -c "import vtk; print(f'VTK version: {vtk.VTK_VERSION}')"
python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import einops; print(f'einops version: {einops.__version__}')"
python -c "import pyvista; print(f'PyVista version: {pyvista.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
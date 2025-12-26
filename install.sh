#!/bin/bash
# TurboDiffusion Installation Script
# Handles git submodules and builds CUDA extensions for RTX 5090 (Blackwell)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "TurboDiffusion Installation Script"
echo "=============================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
echo "CUDA version: $CUDA_VERSION"

# Check for conda environment
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "WARNING: No conda environment active."
    echo "Consider activating: conda activate turbodiffusion"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "Conda environment: $CONDA_DEFAULT_ENV"
fi

# Check Python and PyTorch
echo ""
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not found or CUDA not available"
    exit 1
}

# Initialize git submodules (CUTLASS)
echo ""
echo "Initializing git submodules (CUTLASS)..."
if [ -d ".git" ]; then
    git submodule update --init --recursive
    echo "Submodules initialized."
else
    echo "WARNING: Not a git repository. Checking if CUTLASS exists..."
    if [ ! -f "turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]; then
        echo "ERROR: CUTLASS not found. Please clone with: git clone --recursive <repo>"
        exit 1
    fi
fi

# Verify CUTLASS headers exist
if [ ! -f "turbodiffusion/ops/cutlass/include/cutlass/cutlass.h" ]; then
    echo "ERROR: CUTLASS headers not found after submodule init"
    exit 1
fi
echo "CUTLASS headers verified."

# Clean previous builds (optional)
if [ "$1" == "--clean" ]; then
    echo ""
    echo "Cleaning previous builds..."
    rm -rf build/ dist/ *.egg-info/
    find . -name "*.so" -path "*/turbodiffusion/*" -delete 2>/dev/null || true
    echo "Clean complete."
fi

# Build and install
echo ""
echo "Building TurboDiffusion (this may take several minutes)..."
echo "Compiling CUDA kernels for: sm_80 (Ampere), sm_89 (Ada), sm_90 (Hopper), sm_120a (Blackwell)"
echo ""

pip install -e . --no-build-isolation 2>&1 | tee build.log

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import turbo_diffusion_ops
print('SUCCESS: turbo_diffusion_ops loaded')
print('Available ops:', [x for x in dir(turbo_diffusion_ops) if not x.startswith('_')])
"

echo ""
echo "=============================================="
echo "Installation complete!"
echo "=============================================="
echo ""
echo "Usage:"
echo "  import torch"
echo "  import turbodiffusion"
echo ""

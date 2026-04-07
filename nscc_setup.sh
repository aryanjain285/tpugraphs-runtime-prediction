#!/bin/bash
# nscc_setup.sh — One-time setup on NSCC ASPIRE 2A
# Run this ONCE after cloning the repo.

set -e

echo "=== NSCC Setup ==="

# 1. Load conda
echo "[1/3] Loading conda..."
module load miniforge3
eval "$(conda shell.bash hook)"

# 2. Create environment
echo "[2/3] Creating conda environment..."
conda create -n tpugraphs python=3.10 -y
conda activate tpugraphs

# 3. Install packages
echo "[3/3] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric scipy tqdm numpy pandas

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Extract data:  mkdir -p data && unzip -q predict-ai-model-runtime.zip -d data/"
echo "  2. Submit job:     qsub nscc_job.pbs"

#!/bin/bash
# nscc_setup.sh — One-time setup on NSCC ASPIRE 2A
# Run this ONCE after cloning the repo to your NSCC home directory.
#
# Usage:
#   ssh <your-id>@aspire.nscc.sg
#   git clone https://github.com/aryanjain285/tpugraphs-runtime-prediction.git
#   cd tpugraphs-runtime-prediction
#   bash nscc_setup.sh

set -e

echo "=== NSCC Setup ==="

# 1. Create conda environment
echo "[1/3] Creating conda environment..."
module load miniforge3
conda create -n tpugraphs python=3.10 -y
conda activate tpugraphs

# 2. Install dependencies
echo "[2/3] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric scipy tqdm numpy pandas

# 3. Download data
echo "[3/3] Downloading dataset..."
pip install kaggle
echo "Make sure ~/.kaggle/kaggle.json exists with your API key."
echo "Then run:"
echo "  kaggle competitions download -c predict-ai-model-runtime"
echo "  mkdir -p data && unzip -q predict-ai-model-runtime.zip -d data/"
echo ""
echo "Or transfer the zip from your local machine:"
echo "  scp predict-ai-model-runtime.zip <your-id>@aspire.nscc.sg:~/tpugraphs-runtime-prediction/"
echo ""
echo "=== Setup complete ==="
echo "Submit jobs with: qsub nscc_job.pbs"

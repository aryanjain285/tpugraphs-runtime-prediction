#!/bin/bash
# run_all.sh — End-to-end training and prediction pipeline
# Assumes data is already extracted at data/npz_all/npz/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_ROOT="data/npz_all/npz"

echo "=============================================="
echo "  TpuGraphs — Full Training Pipeline"
echo "=============================================="
echo ""
echo "Running from: $(pwd)"
echo ""

# ── Install dependencies ──
echo "[1/7] Installing Python dependencies..."
pip install -q torch-geometric scipy tqdm
echo ""

# ── Validate dataset ──
echo "[2/7] Validating dataset..."

REQUIRED_DIRS=(
  "$DATA_ROOT/tile/xla/train"
  "$DATA_ROOT/tile/xla/valid"
  "$DATA_ROOT/tile/xla/test"
  "$DATA_ROOT/layout/xla/default/train"
  "$DATA_ROOT/layout/xla/random/train"
  "$DATA_ROOT/layout/nlp/default/train"
  "$DATA_ROOT/layout/nlp/random/train"
)

ALL_OK=true
for dir in "${REQUIRED_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    COUNT=$(find "$dir" -name "*.npz" | wc -l)
    echo "  ✓ $dir ($COUNT files)"
  else
    echo "  ✗ MISSING: $dir"
    ALL_OK=false
  fi
done

if [ "$ALL_OK" = false ]; then
  echo ""
  echo "ERROR: Some data directories are missing."
  echo "Make sure you extracted the zip so the structure is:"
  echo "  data/npz_all/npz/tile/xla/{train,valid,test}/"
  echo "  data/npz_all/npz/layout/{xla,nlp}/{default,random}/{train,valid,test}/"
  exit 1
fi
echo "Dataset OK."
echo ""

# ── Train tile:xla ──
echo "[3/7] Training tile:xla..."
python3 train_tile.py
echo ""

# ── Train layout collections ──
echo "[4/7] Training layout:xla:default..."
python3 train_layout.py --source xla --search default
echo ""

echo "[5/7] Training layout:xla:random..."
python3 train_layout.py --source xla --search random
echo ""

echo "[6/7] Training layout:nlp:default..."
python3 train_layout.py --source nlp --search default
echo ""

echo "[7/7] Training layout:nlp:random + predictions..."
python3 train_layout.py --source nlp --search random
echo ""

# ── Generate submission ──
echo "Generating predictions..."
python3 predict.py
python3 combine_submissions.py
echo ""

echo "=============================================="
echo "  Done! Submission file:"
echo "  outputs/submission/final_submission.csv"
echo "=============================================="

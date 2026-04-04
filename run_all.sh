#!/bin/bash
# run_all.sh — End-to-end training and prediction pipeline.
#
# This script trains models for all 5 sub-collections, generates
# predictions on the test set, and combines them into the final
# submission CSV.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Prerequisites:
#   - Python environment with dependencies installed (see requirements.txt)
#   - Data extracted to data/npz_all/npz/ (see README.md)

set -e  # Exit on error

echo "=============================================="
echo "  TpuGraphs — Full Training Pipeline"
echo "=============================================="
echo ""

# ── Step 1: Train tile:xla ──────────────────────────────────
echo "[1/6] Training tile:xla model..."
python train_tile.py
echo ""

# ── Step 2-5: Train layout collections ──────────────────────
echo "[2/6] Training layout:xla:default model..."
python train_layout.py --source xla --search default
echo ""

echo "[3/6] Training layout:xla:random model..."
python train_layout.py --source xla --search random
echo ""

echo "[4/6] Training layout:nlp:default model..."
python train_layout.py --source nlp --search default
echo ""

echo "[5/6] Training layout:nlp:random model..."
python train_layout.py --source nlp --search random
echo ""

# ── Step 6: Generate predictions and combine ────────────────
echo "[6/6] Generating predictions and combining..."
python predict.py
python combine_submissions.py
echo ""

echo "=============================================="
echo "  Done! Submission file:"
echo "  outputs/submission/final_submission.csv"
echo "=============================================="
echo ""
echo "Upload to Kaggle with:"
echo "  kaggle competitions submit -c predict-ai-model-runtime \\"
echo "    -f outputs/submission/final_submission.csv \\"
echo "    -m 'GNN solution'"

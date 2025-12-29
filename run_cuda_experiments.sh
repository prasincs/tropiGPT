#!/bin/bash
# CUDA-optimized experiment runner for RTX 3080
# Run all 4 configurations with 10k iterations each

set -e

# Setup virtual environment with uv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv pip install torch numpy transformers datasets tiktoken tqdm

# Common settings for RTX 3080
COMMON="--max_iters=10000 --eval_interval=1000 --log_interval=100 --batch_size=64"

echo "=============================================="
echo "TropiGPT CUDA Experiments (RTX 3080)"
echo "=============================================="
echo ""

# 1. Baseline (Standard Attention)
echo "[1/4] Running Baseline (Standard Attention)..."
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py $COMMON \
    --tropical_attention=False \
    --use_abacus=False \
    --out_dir=out-baseline-10k-cuda \
    2>&1 | tee experiment_baseline_10k.log

# 2. Abacus Only
echo "[2/4] Running Abacus Embeddings Only..."
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py $COMMON \
    --tropical_attention=False \
    --use_abacus=True \
    --out_dir=out-abacus-10k-cuda \
    2>&1 | tee experiment_abacus_10k.log

# 3. Tropical Only
echo "[3/4] Running Tropical Attention Only..."
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py $COMMON \
    --tropical_attention=True \
    --use_abacus=False \
    --out_dir=out-tropical-10k-cuda \
    2>&1 | tee experiment_tropical_10k.log

# 4. TropiGPT (Tropical + Abacus)
echo "[4/4] Running TropiGPT (Tropical + Abacus)..."
PYTHONUNBUFFERED=1 .venv/bin/python train_arithmetic.py $COMMON \
    --tropical_attention=True \
    --use_abacus=True \
    --out_dir=out-tropigpt-10k-cuda \
    2>&1 | tee experiment_tropigpt_10k.log

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
echo ""
echo "Run comparison:"
echo "  .venv/bin/python compare_experiments.py"

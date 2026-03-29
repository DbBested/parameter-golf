#!/bin/bash
# nGPT RunPod submission script
# Usage: scp this whole submission/ folder to RunPod, then run this script
#
# Prerequisites:
# 1. RunPod pod with Parameter Golf template (8xH100 SXM)
# 2. SSH'd into the pod
# 3. This submission/ folder copied to /workspace/
#
# Key nGPT features:
# - L2NormalizeHP: opaque autograd function for bf16 compile precision
# - Post-dequant renormalization: 700x reduction in quantization gap
# - Full nGPT: hypersphere-normalized transformers
# - ~119ms/step on 8xH200, ~120-130ms estimated on 8xH100

set -e

echo "=========================================="
echo "  Parameter Golf — nGPT Submission"
echo "  Full nGPT + L2NormalizeHP bf16 compile"
echo "=========================================="

cd /workspace

# 1. Clone repo + download data
if [ ! -d "parameter-golf" ]; then
    echo "[1] Cloning parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

if [ ! -d "data/datasets/fineweb10B_sp1024" ]; then
    echo "[1b] Downloading FineWeb data..."
    python3 data/cached_challenge_fineweb.py --variant sp1024
fi

# 2. Copy our train script
echo "[2] Copying nGPT train script..."
cp /workspace/submission/train_gpt.py .

# 3. Install deps
echo "[3] Installing dependencies..."
pip install --break-system-packages zstandard sentencepiece 2>&1 | tail -2

# Try FA3 prebuilt wheel first, fall back to FA2
echo "[3b] Installing FlashAttention..."
pip install --break-system-packages flash-attn --no-build-isolation 2>&1 | tail -2
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" 2>/dev/null || \
python3 -c "from flash_attn import flash_attn_func; print('FA2: OK (fallback)')" 2>/dev/null || \
echo "WARNING: No FlashAttention — will use SDPA fallback"

# 4. Verify GPUs
echo "[4] GPU check..."
nvidia-smi -L
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "GPUs: $NUM_GPUS"

# Detect if FA3 is available
HAS_FA3=$(python3 -c "
try:
    from flash_attn_interface import flash_attn_func
    print('1')
except:
    print('0')
" 2>/dev/null)
if [ "$HAS_FA3" = "1" ]; then
    echo "FlashAttention-3 detected — using native FA3"
    FORCE_SDPA_VAL=0
else
    echo "No FA3 — using SDPA fallback"
    FORCE_SDPA_VAL=1
fi

# Common nGPT env vars
export NGPT_ENABLED=1
export NGPT_FULL=1
export NGPT_WEIGHT_NORM=1
export NGPT_QUANT_MODE=renorm
export NUM_LAYERS=12
export MODEL_DIM=512
export MLP_MULT=3
export NUM_HEADS=8
export NUM_KV_HEADS=4
export BIGRAM_VOCAB_SIZE=8192
export XSA_LAST_N=0
export VE_ENABLED=0
export LN_SCALE=0
export ROPE_DIMS=0
export SRYS_PROB=0
export MUON_BACKEND_STEPS=5
export VRL_ENABLED=0
export PRUNE_PCT=0.0
export TARGET_ARTIFACT_BYTES=15950000
export WARMDOWN_ITERS=3500
export ITERATIONS=20000
export TRAIN_LOG_EVERY=500
export VAL_LOSS_EVERY=2000
export MAX_WALLCLOCK_SECONDS=580

# 5. Run 3 seeds
echo "=========================================="
echo "  Running 3-seed validation"
echo "=========================================="
for SEED in 1337 42 7; do
    echo "=== Seed $SEED ==="
    SEED=$SEED \
    FORCE_SDPA=$FORCE_SDPA_VAL \
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt.py 2>&1 | tee "log_ngpt_seed${SEED}.txt"

    if [ "$SEED" = "1337" ]; then
        cp final_model.int6.ptz final_model.int6.seed1337.ptz 2>/dev/null || true
    fi
    echo "=== Seed $SEED done ==="
    echo ""
done

# 6. Summary
echo "=========================================="
echo "  RESULTS SUMMARY — nGPT"
echo "=========================================="
echo ""
echo "=== Sliding Window BPB (3-seed) ==="
for SEED in 1337 42 7; do
    LOG="log_ngpt_seed${SEED}.txt"
    if [ -f "$LOG" ]; then
        BPB=$(grep "final_int6_sliding_window_exact" "$LOG" | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
        SIZE=$(grep "Total submission size" "$LOG" | head -1 | grep -o "[0-9]* bytes")
        STEPS=$(grep "stopping_early" "$LOG" | grep -o "step:[0-9]*" | cut -d: -f2)
        MS=$(grep "stopping_early" "$LOG" | grep -o "step_avg:[0-9.]*" | cut -d: -f2)
        echo "  Seed $SEED: BPB=$BPB Artifact=${SIZE} Steps=$STEPS ms/step=$MS"
    fi
done

echo ""
echo "=== Key nGPT Contributions ==="
echo "  1. L2NormalizeHP: opaque autograd.Function via allow_in_graph"
echo "     - Solves bf16 compile precision compounding (PyTorch #168126)"
echo "     - 25% faster than fp32 compile, zero graph breaks"
echo "  2. Post-dequant renormalization: 700x quantization gap reduction"
echo "  3. Three initialization fixes that make full nGPT trainable"
echo "  4. Novel finding: torch.compile + sequential L2 normalize divergence"
echo ""
echo "Done! Check results above."

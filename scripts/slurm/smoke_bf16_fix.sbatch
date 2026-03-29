#!/bin/bash
#SBATCH --job-name=smoke-bf16
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%j-smoke-bf16.out
#SBATCH --error=logs/slurm/%j-smoke-bf16.err

module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf
cd "${SLURM_SUBMIT_DIR}/repo"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}')"
export NCCL_DEBUG=WARN
export LD_PRELOAD=$HOME/.conda/envs/pgolf/lib/glibc_shim.so
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER

SEED="${SEED:-1337}"

echo "=== SMOKE TEST: bf16 compile fix (L2NormalizeHP opaque) ==="
echo "=== Full nGPT, 12L 3x + BigramHash, 8xH200, compiled bf16 ==="

# bf16 compile with opaque normalize — the fix being tested
# NO_COMPILE is NOT set (compile enabled)
# Autocast is always True now (code change)
FORCE_SDPA=1 VRL_ENABLED=0 \
NGPT_ENABLED=1 \
NGPT_FULL=1 \
NGPT_WEIGHT_NORM=1 \
NGPT_QUANT_MODE=renorm \
NUM_LAYERS=12 MODEL_DIM=512 \
MLP_MULT=3 \
NUM_HEADS=8 NUM_KV_HEADS=4 \
BIGRAM_VOCAB_SIZE=8192 \
XSA_LAST_N=0 \
VE_ENABLED=0 \
LN_SCALE=0 \
ROPE_DIMS=0 \
SRYS_PROB=0 \
MUON_BACKEND_STEPS=5 \
MAX_WALLCLOCK_SECONDS=120 \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
TRAIN_LOG_EVERY=100 \
VAL_LOSS_EVERY=500 \
PRUNE_PCT=0.0 TARGET_ARTIFACT_BYTES=15950000 \
SEED="${SEED}" \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "${SLURM_SUBMIT_DIR}/logs/slurm/${SLURM_JOB_ID}-smoke-bf16-s${SEED}.log"

echo "Exit: $?"

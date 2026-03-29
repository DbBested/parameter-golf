#!/bin/bash
#SBATCH --job-name=pgolf-ngpt-bf16
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j-ngpt-bf16.out
#SBATCH --error=logs/slurm/%j-ngpt-bf16.err

module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf
cd "${SLURM_SUBMIT_DIR}/repo"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, {torch.cuda.get_device_name(0)}')"
export NCCL_DEBUG=WARN
export LD_PRELOAD=$HOME/.conda/envs/pgolf/lib/glibc_shim.so
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER

SEED="${SEED:-1337}"
NUM_LAYERS="${NUM_LAYERS:-12}"
MLP_MULT="${MLP_MULT:-3}"
BIGRAM_VOCAB="${BIGRAM_VOCAB:-8192}"
WALLCLOCK="${WALLCLOCK:-560}"
SRYS_PROB="${SRYS_PROB:-0}"
INT5_ALL="${INT5_ALL:-0}"
LABEL="${LABEL:-bf16}"

echo "=== nGPT bf16 compile: ${NUM_LAYERS}L ${MLP_MULT}x BigramHash=${BIGRAM_VOCAB} seed=${SEED} ==="
echo "=== L2NormalizeHP opaque, 8xH200, ${WALLCLOCK}s wallclock ==="

FORCE_SDPA=1 VRL_ENABLED=0 \
NGPT_ENABLED=1 \
NGPT_FULL=1 \
NGPT_WEIGHT_NORM=1 \
NGPT_QUANT_MODE=renorm \
NUM_LAYERS="${NUM_LAYERS}" MODEL_DIM=512 \
MLP_MULT="${MLP_MULT}" \
NUM_HEADS=8 NUM_KV_HEADS=4 \
BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB}" \
XSA_LAST_N=0 \
VE_ENABLED=0 \
LN_SCALE=0 \
ROPE_DIMS=0 \
SRYS_PROB="${SRYS_PROB}" SRYS_START=7 SRYS_END=9 SRYS_MAX_REPS=1 \
MUON_BACKEND_STEPS=5 \
INT5_ALL="${INT5_ALL}" \
MAX_WALLCLOCK_SECONDS="${WALLCLOCK}" \
ITERATIONS=20000 \
WARMDOWN_ITERS=3500 \
TRAIN_LOG_EVERY=500 \
VAL_LOSS_EVERY=2000 \
PRUNE_PCT=0.0 TARGET_ARTIFACT_BYTES=15950000 \
SEED="${SEED}" \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "${SLURM_SUBMIT_DIR}/logs/slurm/${SLURM_JOB_ID}-${LABEL}-s${SEED}.log"

echo "Exit: $?"

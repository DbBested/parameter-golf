#!/bin/bash
#SBATCH --job-name=pgolf-baseline
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%j-%x.out
#SBATCH --error=experiments/logs/%j-%x.err
#SBATCH --nodelist=node4300,node4301

# Environment setup
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
conda activate pgolf

# Navigate to repo
cd "${SLURM_SUBMIT_DIR}/repo"

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

# Set NCCL environment
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Training with optional seed override
SEED="${SEED:-1337}"
echo "Training with seed=${SEED}"

torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "Training complete"

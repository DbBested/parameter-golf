#!/bin/bash
#SBATCH --job-name=pgolf-sota
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j-sota.out
#SBATCH --error=logs/slurm/%j-sota.err

set -euo pipefail
mkdir -p logs/slurm

# Environment
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf

SEED=${1:-1337}
RUN_ID="sota_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=== SOTA Training: seed=$SEED run_id=$RUN_ID ==="
echo "=== Node: $(hostname) GPUs: $SLURM_GPUS_PER_NODE ==="

cd /orcd/home/002/tomli/parameter_golf/repo

SEED=$SEED RUN_ID=$RUN_ID \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee "../logs/${RUN_ID}.log"

echo "=== Training complete ==="
echo "=== Artifact size check ==="
ls -la final_model.int8.ptz 2>/dev/null || echo "No artifact file found"
echo "=== Log file: logs/${RUN_ID}.log ==="

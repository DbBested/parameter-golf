#!/bin/bash
#SBATCH --job-name=pgolf-p6w1
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j-p6w1.out
#SBATCH --error=logs/slurm/%j-p6w1.err
#SBATCH --nodelist=node4300,node4301

module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

cd "${SLURM_SUBMIT_DIR}/repo"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

SEED="${SEED:-1337}"
RUN_ID="p6w1_seed${SEED}"

echo "=== Phase 6 Wave 1: LeakyReLU(0.5)^2, 11L, BigramHash(4096), Partial RoPE 16/64, LN Scale ==="
echo "Seed: ${SEED}, Run ID: ${RUN_ID}"

RUN_ID="${RUN_ID}" SEED="${SEED}" torchrun --standalone --nproc_per_node=8 train_gpt.py

TRAIN_EXIT=$?
echo "Training complete (exit code: ${TRAIN_EXIT})"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit ${TRAIN_EXIT}
fi

cd "${SLURM_SUBMIT_DIR}"
echo "=== Artifact size check ==="
python scripts/eval/check_artifact_size.py --model repo/final_model.int8.ptz --code repo/train_gpt.py
echo "=== Done ==="

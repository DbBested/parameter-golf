#!/bin/bash
#SBATCH --job-name=pgolf-ttt
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%j-moonshot-ttt.out
#SBATCH --error=logs/slurm/%j-moonshot-ttt.err

set -euo pipefail
mkdir -p logs/slurm

module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf

SEED=${1:-1337}
RUN_ID="moonshot_ttt_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=== Moonshot TTT: seed=$SEED run_id=$RUN_ID ==="
echo "=== Node: $(hostname) GPUs: $SLURM_GPUS_PER_NODE ==="

cd /orcd/home/002/tomli/parameter_golf/repo

SEED=$SEED RUN_ID=$RUN_ID \
TTT_ENABLED=1 \
TTT_LORA_RANK=4 \
TTT_LORA_LR=0.01 \
TTT_CHUNK_SIZE=256 \
TTT_EVAL_SEQ_LEN=1024 \
TTT_BATCH_SIZE=64 \
torchrun --standalone --nproc_per_node=8 experiments/moonshot_ttt.py \
    2>&1 | tee "../logs/${RUN_ID}.log"

echo "=== Training complete ==="

ARTIFACT_FILE="final_model.int8.ptz"
if [ -f "$ARTIFACT_FILE" ]; then
    SIZE=$(stat -c%s "$ARTIFACT_FILE")
    CODE_SIZE=$(wc -c < experiments/moonshot_ttt.py)
    TOTAL=$((SIZE + CODE_SIZE))
    echo "=== Artifact: ${SIZE} bytes, Code: ${CODE_SIZE} bytes, Total: ${TOTAL} bytes ==="
else
    echo "WARNING: No artifact file found at $ARTIFACT_FILE"
fi

echo "=== Log file: logs/${RUN_ID}.log ==="

#!/bin/bash
#SBATCH --job-name=pgolf-int5
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j-int5.out
#SBATCH --error=logs/slurm/%j-int5.err

set -euo pipefail
mkdir -p logs/slurm

# Environment
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf

SEED=${1:-1337}
RUN_ID="int5_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "=== Int5/Int6 Training: seed=$SEED run_id=$RUN_ID ==="
echo "=== Node: $(hostname) GPUs: $SLURM_GPUS_PER_NODE ==="

cd /orcd/home/002/tomli/parameter_golf/repo

SEED=$SEED RUN_ID=$RUN_ID \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
    2>&1 | tee "../logs/${RUN_ID}.log"

echo "=== Training complete ==="

# Artifact size cap check
ARTIFACT_FILE="final_model.int8.ptz"
if [ -f "$ARTIFACT_FILE" ]; then
    SIZE=$(stat -c%s "$ARTIFACT_FILE")
    CODE_SIZE=$(wc -c < train_gpt.py)
    TOTAL=$((SIZE + CODE_SIZE))
    echo "=== Artifact: ${SIZE} bytes, Code: ${CODE_SIZE} bytes, Total: ${TOTAL} bytes ==="
    if [ $TOTAL -gt 16000000 ]; then
        echo "FAIL: Total ${TOTAL} exceeds 16,000,000 byte cap by $((TOTAL - 16000000)) bytes"
    else
        echo "PASS: Total ${TOTAL} fits cap with $((16000000 - TOTAL)) bytes headroom"
    fi
else
    echo "WARNING: No artifact file found at $ARTIFACT_FILE"
fi

echo "=== Log file: logs/${RUN_ID}.log ==="

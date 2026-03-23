#!/bin/bash
#SBATCH --job-name=pgolf-multiseed
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/%j-%x.out
#SBATCH --error=experiments/logs/%j-%x.err
#SBATCH --nodelist=node4300,node4301

# Multi-seed baseline training for reproducibility evaluation
# Runs 3 seeds sequentially, logs each, then computes multi-seed stats

# Environment setup
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
eval "$(conda shell.bash hook)"
conda activate pgolf

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Navigate to submission directory
cd "${SLURM_SUBMIT_DIR}"

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}, Device: {torch.cuda.get_device_name(0)}')"

# Set NCCL environment
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Seeds to train
SEEDS="${SEEDS:-1337 42 7}"
FAILED_SEEDS=""
TOTAL_SEEDS=0
SUCCESS_SEEDS=0

for SEED in ${SEEDS}; do
    TOTAL_SEEDS=$((TOTAL_SEEDS + 1))
    export SEED=${SEED}
    RUN_ID="baseline_seed${SEED}"

    echo ""
    echo "============================================"
    echo "=== Training seed ${SEED} (run_id: ${RUN_ID}) ==="
    echo "============================================"
    echo "Seed start: $(date)"

    # Record start time for this seed
    SEED_START=$(date +%s)

    # Run training
    cd "${SLURM_SUBMIT_DIR}/repo"
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    TRAIN_EXIT=$?

    # Record end time
    SEED_END=$(date +%s)
    SEED_WALL_TIME=$((SEED_END - SEED_START))

    echo "Seed ${SEED} training exit code: ${TRAIN_EXIT}"
    echo "Seed ${SEED} wall time: ${SEED_WALL_TIME} seconds"

    cd "${SLURM_SUBMIT_DIR}"

    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "ERROR: Training failed for seed ${SEED} (exit code ${TRAIN_EXIT})"
        FAILED_SEEDS="${FAILED_SEEDS} ${SEED}"
        continue
    fi

    SUCCESS_SEEDS=$((SUCCESS_SEEDS + 1))

    # Check artifact size
    echo ""
    echo "=== Artifact size check (seed ${SEED}) ==="
    python scripts/eval/check_artifact_size.py \
        --model repo/final_model.int8.ptz \
        --code repo/train_gpt.py

    # Process results from this run's log output and log experiment
    echo ""
    echo "=== Processing results for seed ${SEED} ==="

    # Extract results from the SLURM log (which includes this seed's output)
    # We need to extract BPB and timing from the log output
    LOG_FILE="experiments/logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}.out"

    python scripts/eval/process_baseline_results.py \
        --log "${LOG_FILE}" \
        --result-dir experiments/results \
        --run-id "${RUN_ID}" \
        --seed ${SEED} \
        --wall-time ${SEED_WALL_TIME}

    # Copy model to checkpoint directory with seed-specific name
    mkdir -p checkpoints
    cp repo/final_model.int8.ptz "checkpoints/baseline_seed${SEED}.ptz"
    echo "Saved checkpoint: checkpoints/baseline_seed${SEED}.ptz"

    echo ""
    echo "=== Seed ${SEED} complete ==="
    echo ""
done

echo ""
echo "============================================"
echo "=== All seeds complete ==="
echo "============================================"
echo "Total: ${TOTAL_SEEDS}, Success: ${SUCCESS_SEEDS}, Failed: ${FAILED_SEEDS:-none}"
echo ""

# Run multi-seed evaluation
if [ ${SUCCESS_SEEDS} -ge 2 ]; then
    echo "=== Multi-seed BPB evaluation ==="
    python scripts/eval/multiseed_eval.py \
        --result-dir experiments/results \
        --run-id-prefix baseline_seed

    echo ""
    echo "=== Timing calibration ==="
    python scripts/eval/timing_calibration.py \
        --result-dir experiments/results \
        --run-id-prefix baseline_seed
else
    echo "WARNING: Only ${SUCCESS_SEEDS} seeds succeeded; skipping multi-seed analysis"
fi

# Cleanup old checkpoints (keep only the 3 most recent)
echo ""
echo "=== Storage cleanup ==="
python scripts/storage/cleanup.py \
    --checkpoint-dir checkpoints \
    --keep-recent 3 \
    --results-file experiments/results/experiments.jsonl

echo ""
echo "End time: $(date)"
echo "Job complete."

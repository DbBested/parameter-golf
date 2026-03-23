#!/bin/bash
#SBATCH --job-name=pgolf-timing
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

# Timing profiling job for H200-to-H100 calibration
# Runs a single baseline training and measures precise wall-clock time

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

# Use default seed
SEED="${SEED:-1337}"
export SEED

echo ""
echo "=== Timing Profile Run ==="
echo "Seed: ${SEED}"
echo ""

# Record precise start time
TIMING_START=$(date +%s%N)
TIMING_START_S=$(date +%s)

# Run training
cd "${SLURM_SUBMIT_DIR}/repo"
torchrun --standalone --nproc_per_node=8 train_gpt.py
TRAIN_EXIT=$?

# Record precise end time
TIMING_END=$(date +%s%N)
TIMING_END_S=$(date +%s)

# Compute wall time
WALL_TIME_NS=$((TIMING_END - TIMING_START))
WALL_TIME_SECONDS=$((TIMING_END_S - TIMING_START_S))

echo ""
echo "Training exit code: ${TRAIN_EXIT}"
echo "WALL_TIME_SECONDS=${WALL_TIME_SECONDS}"
echo "Wall time: ${WALL_TIME_SECONDS} seconds ($((WALL_TIME_SECONDS / 60)) minutes, $((WALL_TIME_SECONDS % 60)) seconds)"

cd "${SLURM_SUBMIT_DIR}"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    exit ${TRAIN_EXIT}
fi

# Run timing calibration
echo ""
echo "=== H200-to-H100 Timing Calibration ==="
python scripts/eval/timing_calibration.py \
    --h200-seconds ${WALL_TIME_SECONDS} \
    --ratio 1.35

# Also try with the log file
LOG_FILE="experiments/logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}.out"
echo ""
echo "=== Timing from training log ==="
python scripts/eval/timing_calibration.py \
    --log-path "${LOG_FILE}" \
    --ratio 1.35

echo ""
echo "End time: $(date)"
echo "Timing profile complete."

#!/bin/bash
# Run baseline training and process results.
#
# This script:
# 1. Submits the baseline training job to SLURM
# 2. Waits for it to complete
# 3. Processes results (extracts BPB, validates, logs experiment)
#
# Usage:
#   bash scripts/run_baseline.sh
#   SEED=42 bash scripts/run_baseline.sh

set -euo pipefail

PROJECT_DIR="/orcd/home/002/tomli/parameter_golf"
cd "${PROJECT_DIR}"

SEED="${SEED:-1337}"
RUN_ID="baseline_seed${SEED}"

echo "=== Submitting baseline training ==="
echo "Seed: ${SEED}"
echo "Run ID: ${RUN_ID}"

# Ensure log directory exists
mkdir -p experiments/logs

# Submit the job
JOB_ID=$(SEED="${SEED}" RUN_ID="${RUN_ID}" sbatch --parsable scripts/slurm/train_baseline.sbatch)
echo "SLURM Job ID: ${JOB_ID}"
echo "Log file will be: experiments/logs/${JOB_ID}-pgolf-baseline.out"

echo ""
echo "=== Monitor with ==="
echo "  squeue -u \$USER"
echo "  tail -f experiments/logs/${JOB_ID}-pgolf-baseline.out"
echo ""
echo "=== After job completes, process results with ==="
echo "  conda activate pgolf"
echo "  python scripts/eval/process_baseline_results.py \\"
echo "      --log experiments/logs/${JOB_ID}-pgolf-baseline.out \\"
echo "      --result-dir experiments/results \\"
echo "      --run-id ${RUN_ID} \\"
echo "      --seed ${SEED}"
echo ""
echo "=== Or validate manually ==="
echo "  # Extract BPB from log:"
echo "  grep 'final_int8_zlib_roundtrip ' experiments/logs/${JOB_ID}-pgolf-baseline.out"
echo ""
echo "  # Validate BPB:"
echo "  python scripts/eval/validate_bpb.py --measured <BPB>"
echo ""
echo "  # Check artifact size:"
echo "  python scripts/eval/check_artifact_size.py --model repo/final_model.int8.ptz --code repo/train_gpt.py"

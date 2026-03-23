# Plan 01-02 Summary: Baseline Training & Validation

## Status: COMPLETE

## What Was Built

### Artifact Size Checker (`scripts/eval/check_artifact_size.py`)
Validates code + compressed model against the 16,000,000 byte cap. Reports model size, code size, total, and remaining budget. Exits non-zero if limit exceeded. Warns if < 500KB margin.

### BPB Validation (`scripts/eval/validate_bpb.py`)
Compares measured BPB against expected baseline (1.2244) with configurable tolerance (default 0.005). Reports PASS/FAIL with diagnostic notes for large deviations.

### Result Processor (`scripts/eval/process_baseline_results.py`)
Extracts BPB from SLURM log files, validates against expected value, and logs experiment to JSON-lines tracker.

### Run Wrapper (`scripts/run_baseline.sh`)
Submits baseline training SLURM job and prints monitoring/post-processing instructions.

### SLURM Script Fix
Fixed `scripts/slurm/train_baseline.sbatch` to properly initialize conda via `eval "$(conda shell.bash hook)"` before `conda activate pgolf`. Added Python/PyTorch version logging.

## Baseline Results

| Metric | Value |
|--------|-------|
| val_bpb | 1.2259 |
| val_loss | 2.0699 |
| Artifact size | 15,875,299 bytes (15.88 MB) |
| Budget remaining | 124,701 bytes |
| Training wall time | 600s (hit wall clock cap) |
| Steps completed | 13,384 / 20,000 |
| Step avg | 44.83ms |
| Peak memory | 10,360 MiB |
| GPUs | 8x NVIDIA H200 |
| Seed | 1337 |
| Node | node4300 |
| SLURM Job | 10812368 |

## Verification

- BPB validation: **PASS** (1.2259, diff 0.0015 from expected 1.2244, within 0.005 tolerance)
- Artifact size: **WARNING** (under 16MB but only 124KB margin)
- Training completed successfully (exit code 0)

## Key Decisions

- Fixed conda activation in SLURM scripts — must use `eval "$(conda shell.bash hook)"` before `conda activate`
- Baseline uses 600s wall clock cap, reaching step 13384 of 20000 on 8xH200

## Self-Check: PASSED

All acceptance criteria met:
- [x] check_artifact_size.py reports code + model vs 16MB with PASS/FAIL
- [x] validate_bpb.py compares measured BPB to 1.2244 within tolerance
- [x] Baseline training reproduces BPB within 0.005 of 1.2244 (actual: 0.0015 diff)
- [x] Artifact under 16,000,000 bytes (actual: 15,875,299)

## key-files

### created
- scripts/eval/check_artifact_size.py
- scripts/eval/validate_bpb.py
- scripts/eval/process_baseline_results.py
- scripts/run_baseline.sh

### modified
- scripts/slurm/train_baseline.sbatch

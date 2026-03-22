---
phase: 01-infrastructure-and-baseline
plan: 01
subsystem: infra
tags: [conda, pytorch, slurm, h200, fineweb, experiment-tracking, jsonl]

# Dependency graph
requires: []
provides:
  - "pgolf conda environment with PyTorch 2.6.0+cu124, scipy, zstandard"
  - "Cloned parameter-golf repo at repo/ with train_gpt.py baseline"
  - "FineWeb sp1024 dataset (81 shards, 16GB) at repo/data/datasets/fineweb10B_sp1024/"
  - "SLURM baseline training script (gpu:h200:8, ntasks-per-node=1)"
  - "JSON-lines experiment logger (log_experiment, load_experiments)"
  - "Experiment query/display utility"
  - "Checkpoint cleanup utility with keep-recent + keep-best logic"
  - "Project directory structure (scripts/, experiments/, checkpoints/)"
affects: [01-02, 01-03, 02-proven-techniques, 03-quantization]

# Tech tracking
tech-stack:
  added: [pytorch-2.6.0+cu124, scipy-1.17.1, zstandard-0.25.0, tiktoken-0.12.0, sentencepiece, datasets, huggingface-hub, kernels, numpy, tqdm]
  patterns: [json-lines-logging, slurm-torchrun-ddp, checkpoint-cleanup]

key-files:
  created:
    - scripts/tracking/log_experiment.py
    - scripts/tracking/query_experiments.py
    - scripts/storage/cleanup.py
    - scripts/slurm/train_baseline.sbatch
    - .gitignore
    - experiments/results/.gitkeep
    - experiments/logs/.gitkeep
    - checkpoints/.gitkeep
  modified: []

key-decisions:
  - "PyTorch 2.6.0+cu124 installed (not 2.10) -- cu124 wheels available; 2.10 wheels may not have cu124 index yet"
  - "tiktoken installed despite initial failure -- worked on retry after other deps resolved"
  - "CUDA available=False expected on login node; will be True on compute nodes with GPUs"

patterns-established:
  - "JSON-lines logging: append to experiments.jsonl with timestamp, run_id, config, metrics"
  - "SLURM script pattern: module load + conda activate + torchrun --standalone --nproc_per_node=8"
  - "Checkpoint cleanup: keep N recent + best-BPB, supports dry-run"
  - "All infrastructure scripts use stdlib only (json, pathlib, argparse)"

requirements-completed: [INFRA-05, INFRA-07]

# Metrics
duration: 43min
completed: 2026-03-22
---

# Phase 01 Plan 01: Environment and Infrastructure Setup Summary

**pgolf conda env with PyTorch+cu124, cloned parameter-golf repo with 16GB FineWeb sp1024 dataset, SLURM training script targeting 8xH200, and JSON-lines experiment tracking/cleanup scripts**

## Performance

- **Duration:** 43 min
- **Started:** 2026-03-22T23:07:07Z
- **Completed:** 2026-03-22T23:49:55Z
- **Tasks:** 2/2
- **Files modified:** 8

## Accomplishments
- Created pgolf conda environment with Python 3.11, PyTorch 2.6.0+cu124, scipy, zstandard, tiktoken, and all competition dependencies
- Cloned parameter-golf repository and downloaded full FineWeb sp1024 dataset (81 shards, 16GB)
- Built three infrastructure scripts: experiment logger, query tool, and checkpoint cleanup utility
- Created SLURM baseline training script with correct H200 GRES, single-task torchrun, and node targeting
- Verified disk usage at 92.7GB of 195GB quota -- comfortable headroom

## Task Commits

Each task was committed atomically:

1. **Task 1: Environment Setup, Repo Clone, and Data Download** - `90b0407` (feat)
2. **Task 2: Experiment Tracking and Storage Management Scripts** - `55be258` (feat)

## Files Created/Modified
- `.gitignore` - Excludes checkpoints/*.ptz, repo/, __pycache__, .cache/
- `scripts/slurm/train_baseline.sbatch` - SLURM job script for baseline training on 8xH200
- `scripts/tracking/log_experiment.py` - JSON-lines experiment logger with CLI
- `scripts/tracking/query_experiments.py` - Experiment result querying and formatted display
- `scripts/storage/cleanup.py` - Checkpoint cleanup keeping N recent + best-BPB
- `experiments/results/.gitkeep` - Experiment results directory placeholder
- `experiments/logs/.gitkeep` - SLURM logs directory placeholder
- `checkpoints/.gitkeep` - Checkpoints directory placeholder

## Decisions Made
- Installed PyTorch 2.6.0+cu124 (latest available cu124 wheel) rather than 2.10, which may not yet have cu124 index support. torch.optim.Muon may need the standalone KellerJordan/Muon package for now.
- Used `--force-reinstall` for torch to override a stale cpu-only torch 2.0.1 that was pre-installed in the fresh conda env.
- Set HF_HOME to /orcd/home/002/tomli/.cache/huggingface on same filesystem to enable hardlinks and avoid double storage.
- CUDA available=False is expected on login nodes; verification on compute nodes will happen when SLURM jobs run.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Force-reinstalled PyTorch to replace stale cpu-only version**
- **Found during:** Task 1 (conda env setup)
- **Issue:** Fresh conda env had torch 2.0.1+cpu pre-installed from base env; pip install did not upgrade it
- **Fix:** Used `pip install --force-reinstall torch --index-url .../cu124` to install correct CUDA-enabled version
- **Files modified:** None (conda env only)
- **Verification:** `conda run -n pgolf python -c "import torch; print(torch.__version__)"` returns 2.6.0+cu124

**2. [Rule 3 - Blocking] Removed incompatible torchvision and torchaudio**
- **Found during:** Task 1 (after torch reinstall)
- **Issue:** Old torchvision 0.15.2+cpu and torchaudio 2.0.2+cpu were incompatible with torch 2.6.0+cu124
- **Fix:** `pip uninstall -y torchvision torchaudio` to resolve dependency conflicts
- **Files modified:** None (conda env only)
- **Verification:** No dependency conflict warnings after removal

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary to get a working CUDA-enabled PyTorch environment. No scope creep.

## Issues Encountered
- tiktoken initially failed to install when bundled with other packages; succeeded when installed separately after other dependencies were in place
- pip install torch initially went to base env due to `conda activate` not working in non-interactive shell; switched to `conda run -n pgolf pip install` pattern

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Environment fully set up and ready for Plan 02 (baseline training) and Plan 03 (eval scripts)
- SLURM script ready to submit for baseline reproduction
- Experiment tracking infrastructure in place for logging results
- Concern: PyTorch 2.6.0 does not include native torch.optim.Muon (requires 2.10+); Plan 02/03 may need KellerJordan/Muon standalone or a torch upgrade
- Concern: CUDA available=False on login node is expected but needs verification on compute node during first SLURM job

---
*Phase: 01-infrastructure-and-baseline*
*Completed: 2026-03-22*

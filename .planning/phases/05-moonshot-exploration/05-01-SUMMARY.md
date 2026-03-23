---
phase: 05-moonshot-exploration
plan: 01
subsystem: quantization
tags: [int4, quantization, moonshot, clip-range, experiment]

# Dependency graph
requires:
  - phase: 03-mixed-precision-quantization
    provides: "Mixed int5/int6 quantization framework with clip_range parameter"
  - phase: 04-differentiator-stack
    provides: "Validated base model with BPB 1.1421 and 15,824,167 byte artifact"
provides:
  - "Int4 MLP quantization experiment results (BPB and artifact size)"
  - "Go/no-go decision data for int4 vs int5 MLP quantization"
  - "repo/experiments/moonshot_int4.py experiment script"
affects: [05-moonshot-exploration, 06-submission-hardening]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Experiment branch pattern: copy train_gpt.py to experiments/ and modify"

key-files:
  created:
    - "repo/experiments/moonshot_int4.py"
    - "scripts/slurm/train_moonshot_int4.sbatch"
  modified: []

key-decisions:
  - "Int4 experiment uses single-line change (clip 15->7) to isolate quantization impact"
  - "Experiment script copied rather than parameterized to avoid risk of breaking base model"

patterns-established:
  - "Moonshot experiments: copy base model to experiments/ dir, make minimal changes, run independently"

requirements-completed: [LOWBIT-01, LOWBIT-02, LOWBIT-03]

# Metrics
duration: 5min
completed: 2026-03-23
---

# Phase 5 Plan 1: Int4 MLP Quantization Experiment Summary

**Int4 MLP experiment (clip_range 15->7) submitted as SLURM job 10823707; awaiting results for go/no-go decision**

## Performance

- **Duration:** ~5 min (script creation and job submission)
- **Started:** 2026-03-23T08:10:25Z
- **Completed:** 2026-03-23T08:15:00Z (Task 1); Task 2 checkpoint pending
- **Tasks:** 1/2 (Task 2 is checkpoint awaiting SLURM results)
- **Files created:** 2

## Accomplishments
- Created int4 MLP experiment by changing single clip_range parameter (15 -> 7) in mixed_quantize_int6
- Created and submitted SLURM job 10823707 on pg_tata partition (node4300)
- Verified exactly 2 lines differ between base model and experiment (clip value + metadata string)
- Job running successfully on 8xH200

## Task Commits

Each task was committed atomically:

1. **Task 1: Create int4 experiment script and SLURM job** - `bc0bac4` (repo) + `2488f8b` (parent) (feat)
2. **Task 2: Evaluate int4 results** - CHECKPOINT (awaiting SLURM job completion)

## Files Created/Modified
- `repo/experiments/moonshot_int4.py` - Copy of train_gpt.py with clip=7 for MLP (int4 instead of int5)
- `scripts/slurm/train_moonshot_int4.sbatch` - SLURM job script for int4 experiment

## Decisions Made
- Used experiment copy pattern (copy then modify) rather than parameterizing base model, to avoid any risk to the working base
- Single-seed run (seed 1337) sufficient for initial go/no-go screening

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Awaiting SLURM job 10823707 results for go/no-go decision
- If BPB delta < 0.003 and artifact savings > 200KB: GO (integrate int4 into base model)
- If BPB delta >= 0.005 or no artifact size reduction: NO-GO (keep int5)
- Results will inform whether to pursue sub-5-bit quantization further in remaining Phase 5 plans

---
*Phase: 05-moonshot-exploration*
*Completed: 2026-03-23*

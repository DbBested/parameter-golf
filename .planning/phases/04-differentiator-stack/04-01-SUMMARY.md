---
phase: 04-differentiator-stack
plan: 01
subsystem: model
tags: [smeargate, unet-skips, orthogonal-init, pruning, code-size, quantization]

requires:
  - phase: 03-mixed-precision-quantization
    provides: "Mixed int5/int6 quantization, magnitude pruning, artifact pipeline"
provides:
  - "Verified ARCH-05 (SmearGate), ARCH-06 (U-Net skips), ARCH-07 (OrthoInit)"
  - "Stripped train_gpt.py from 62,489 to 53,995 bytes (-13.6%)"
  - "Increased magnitude pruning from 3% to 5% for better zstd compression"
  - "SLURM script for Phase 4 validation run"
affects: [04-02-PLAN, phase-05, phase-06]

tech-stack:
  added: []
  patterns:
    - "Code size management via comment/docstring stripping"
    - "Magnitude pruning at 5% threshold for zstd compression optimization"

key-files:
  created:
    - "scripts/slurm/train_phase4.sbatch"
  modified:
    - "repo/train_gpt.py"

key-decisions:
  - "Pruning threshold increased from 3% to 5% -- expected BPB cost of 0.001-0.002 but significant compression savings"
  - "Removed per-layer quantization sensitivity logging -- Phase 3 diagnostic no longer needed, saved ~500 bytes"
  - "Shortened verbose log messages to reduce code bytes without losing key information"

patterns-established:
  - "Code stripping: remove comments/docstrings/section dividers to reduce artifact code contribution"

requirements-completed: [ARCH-05, ARCH-06, ARCH-07]

duration: 7min
completed: 2026-03-23
---

# Phase 4 Plan 01: Differentiator Verification and Code Size Reduction Summary

**Verified SmearGate, U-Net skips, OrthoInit in both forward paths; stripped code 62.5KB->54KB; increased pruning to 5%**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-23T07:11:05Z
- **Completed:** 2026-03-23T07:18:53Z
- **Tasks:** 2
- **Files modified:** 1 modified, 1 created

## Accomplishments
- Verified all three ARCH requirements (SmearGate, U-Net skip connections, orthogonal init) are present and wired in both `forward()` and `forward_logits()` paths
- Reduced train_gpt.py from 62,489 bytes to 53,995 bytes (-13.6%) by stripping comments, docstrings, section dividers, and shortening verbose log messages
- Changed magnitude pruning threshold from 3% to 5% for better zstd compression of quantized weights
- Created SLURM job script for Phase 4 validation run with artifact size cap check

## Task Commits

Each task was committed atomically:

1. **Task 1: Verify differentiators and strip code** - `c008ccb` (feat) - in repo/
2. **Task 2: Create SLURM job script** - `7814d62` (feat) - in parent

**Plan metadata:** (next commit) (docs: complete plan)

## Files Created/Modified
- `repo/train_gpt.py` - Stripped comments/docstrings, removed sensitivity logging, pruning 3%->5%
- `scripts/slurm/train_phase4.sbatch` - SLURM script for Phase 4 validation with pg_tata, 8 GPUs

## Decisions Made
- Increased pruning from 3% to 5%: the SOTA reference achieves 1.1428 BPB with 3% pruning; at 5% we expect at most 0.001-0.002 BPB cost but significant improvement in zstd compression ratio
- Removed the per-layer quantization sensitivity logging block: it was added for Phase 3 diagnostics and is no longer needed; saved ~500 bytes of code
- Shortened verbose error messages and log strings (e.g., multi-line ValueError messages -> single line) to reduce code size without losing critical information
- Consolidated 3 separate log0() calls for tokenizer/training/validation info into 1 combined call

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- train_gpt.py is ready for a validation training run at 53,995 bytes
- SLURM script ready to submit: `sbatch scripts/slurm/train_phase4.sbatch`
- Plan 04-02 will submit the job and validate artifact size under 16MB and BPB <= 1.145
- Concern: 5% pruning may have a small BPB cost vs 3% -- validation run will confirm

---
*Phase: 04-differentiator-stack*
*Completed: 2026-03-23*

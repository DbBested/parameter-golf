---
phase: 03-mixed-precision-quantization
plan: 01
subsystem: quantization
tags: [int5, int6, mixed-precision, ptq, zstd, mse-logging, slurm]

# Dependency graph
requires:
  - phase: 02-sota-stack-integration
    provides: "int6 uniform PTQ, zstd-22 compression, magnitude pruning, sliding window eval"
provides:
  - "Mixed int5/int6 quantization (MLP=int5, attention=int6)"
  - "Per-layer quantization sensitivity logging (MSE + max_err)"
  - "SLURM script for int5 training runs with artifact cap check"
affects:
  - 03-02 (validates training results from submitted job)
  - 04-differentiator-stack (builds on mixed quantization)
  - 06-submission-hardening (final artifact size validation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Mixed-precision PTQ: _classify_param() drives per-category clip_range selection"
    - "Per-layer sensitivity logging: quantize-dequantize roundtrip MSE before full serialization"
    - "Artifact cap check in SLURM script: stat-based size validation post-training"

key-files:
  created:
    - scripts/slurm/train_int5.sbatch
  modified:
    - repo/train_gpt.py

key-decisions:
  - "Used clip=15 for MLP (int5) and clip=31 for attention/bigram (int6), matching SOTA #1 exactly"
  - "Sensitivity logging placed after magnitude pruning but before quantization serialization"
  - "Meta type reports 'int5' or 'int6' per parameter category for downstream analysis"

patterns-established:
  - "Mixed quantization: _classify_param() category drives bit-width selection in mixed_quantize_int6()"
  - "Artifact cap check pattern: bash stat + wc in SLURM script for post-training validation"

requirements-completed: [QUANT-02, QUANT-04, QUANT-05]

# Metrics
duration: 2min
completed: 2026-03-23
---

# Phase 3 Plan 1: Mixed Int5/Int6 Quantization Summary

**Mixed-precision PTQ with int5 MLP (clip=15) and int6 attention (clip=31), per-layer MSE sensitivity logging, and SLURM training job submitted (job 10821205)**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T06:29:53Z
- **Completed:** 2026-03-23T06:32:03Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Mixed int5/int6 quantization implemented: MLP weights use clip_range=15 (int5, range [-16,15]), attention/bigram weights use clip_range=31 (int6, range [-32,31])
- Per-layer quantization sensitivity logging added to serialization path: logs MSE and max_err for every 2D parameter >8192 elements
- SLURM training job submitted (ID 10821205, seed 1337) with artifact size cap validation built into the script
- Script at 1452 lines (under 1500 hard cap)

## Task Commits

Each task was committed atomically:

1. **Task 1: Mixed int5/int6 quantization + per-layer MSE logging** - `8561612` (feat)
2. **Task 2: SLURM submission script and training job launch** - `5843063` (feat)

## Files Created/Modified
- `repo/train_gpt.py` - Mixed int5/int6 clip_range selection in mixed_quantize_int6(), per-layer sensitivity logging, updated log prefixes
- `scripts/slurm/train_int5.sbatch` - SLURM script for int5 training with artifact cap check

## Decisions Made
- Matched SOTA #1 pattern exactly: `clip = 15 if cat == "mlp" else 31`
- Meta type dynamically reports `int5` or `int6` based on parameter category
- Sensitivity logging placed after magnitude pruning (so MSE reflects pruned+quantized weights)
- Log prefixes changed from `int6` to `int5_6` and `int8_zlib_roundtrip` to `int5_6_roundtrip`
- SLURM script includes post-training artifact size validation against 16MB cap

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- SLURM job 10821205 is pending in queue on pg_tata partition
- Plan 03-02 will validate: artifact size (must be under 16MB), BPB quality, sensitivity log output
- Magnitude pruning (3%) is unchanged and still runs before quantization (verified by code inspection)
- The dequantize_mixed_int6() function was not modified as it already handles any clip_range generically

---
*Phase: 03-mixed-precision-quantization*
*Completed: 2026-03-23*

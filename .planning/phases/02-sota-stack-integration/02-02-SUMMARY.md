---
phase: 02-sota-stack-integration
plan: 02
subsystem: training
tags: [muon, adamw, weight-decay, swa, optimizer, param-routing, smeargate, bigramhash]

# Dependency graph
requires:
  - phase: 02-01
    provides: "SOTA architecture with SmearGate, BigramHash, 10L, MLP 3x, OrthoInit"
provides:
  - "Muon optimizer with decoupled weight decay 0.04"
  - "AdamW for tok and scalar optimizers with weight_decay=0.04"
  - "SWA checkpoint collection and averaging during warmdown"
  - "Correct parameter routing for SmearGate and BigramHash modules"
  - "Updated CONTROL_TENSOR_NAME_PATTERNS with smear and bigram.scale"
affects: [02-03, 03-01, 06-01]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Decoupled weight decay in Muon (p.data.mul_ before gradient update)"
    - "SWA running sum on CPU, divided by count post-training"
    - "Separate param groups: tok_params (AdamW), matrix_params (Muon), scalar_params (AdamW)"

key-files:
  created: []
  modified:
    - "repo/train_gpt.py"

key-decisions:
  - "Muon weight_decay hardcoded to 0.04 (not args.weight_decay) matching SOTA #1"
  - "lm_head optimizer stays as Adam (not AdamW) matching SOTA #1 pattern"
  - "SWA collection placed after step increment, matching SOTA #1 ordering"

patterns-established:
  - "Decoupled weight decay: p.data.mul_(1.0 - lr * wd) before p.add_(g, alpha=-lr)"
  - "SWA accumulates on CPU to avoid GPU memory pressure"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05]

# Metrics
duration: 2min
completed: 2026-03-23
---

# Phase 02 Plan 02: Training Pipeline Summary

**Muon decoupled WD 0.04, Adam->AdamW for tok/scalar optimizers, SWA checkpoint averaging during warmdown, SmearGate/BigramHash parameter routing**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-23T03:17:58Z
- **Completed:** 2026-03-23T03:20:15Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Muon optimizer now applies decoupled weight decay (0.04) via p.data.mul_ before gradient update
- Adam replaced by AdamW for tok and scalar optimizers with weight_decay=args.weight_decay
- SWA checkpoint collection during warmdown (scale < 0.4, every 50 steps) with post-training averaging
- SmearGate gate, BigramHash scale/embed/proj correctly routed to their respective optimizers
- CONTROL_TENSOR_NAME_PATTERNS updated to include smear and bigram.scale

## Task Commits

Each task was committed atomically:

1. **Task 1: Muon weight decay + AdamW + CONTROL_TENSOR update** - `c14dbe2` (feat)
2. **Task 2: SWA checkpoint collection and averaging** - `dd8e25a` (feat)

## Files Created/Modified
- `repo/train_gpt.py` - Added Muon WD, switched Adam to AdamW, added SWA, routed SmearGate/BigramHash params

## Decisions Made
- Muon weight_decay hardcoded to 0.04 (not args.weight_decay) matching SOTA #1 exactly
- lm_head optimizer stays as Adam (not AdamW) -- SOTA #1 does the same
- SWA collection placed after step increment and before should_log_train, matching SOTA #1 ordering
- SWA accumulates on CPU (detach().cpu()) to avoid GPU memory pressure during warmdown

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Training pipeline complete with Muon WD, AdamW, SWA, correct param routing
- Ready for Plan 02-03: quantization (int6 PTQ), zstd-22 compression, sliding window eval, validation run
- Script at 1250 lines (under 1500 cap) with room for Plan 03 additions

---
*Phase: 02-sota-stack-integration*
*Completed: 2026-03-23*

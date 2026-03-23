---
phase: 05-moonshot-exploration
plan: 03
subsystem: eval
tags: [lora, ttt, test-time-training, evaluation, adaptation]

# Dependency graph
requires:
  - phase: 04-differentiator-stack
    provides: Base model with SOTA stack (10L, BigramHash, SmearGate, int5/int6 QAT, SWA)
provides:
  - LoRA TTT eval implementation with score-first-then-train ordering
  - Per-document LoRA adaptation during evaluation
  - Go/no-go measurement of TTT BPB improvement
affects: [06-submission-hardening]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Copy-and-modify pattern for isolated moonshot experiments"
    - "Per-batch-element LoRA with BatchedLinearLoRA class"
    - "Document-aware eval with BOS boundary detection"
    - "Score-first-then-train TTT ordering for rule compliance"

key-files:
  created:
    - repo/experiments/moonshot_ttt.py
    - scripts/slurm/train_moonshot_ttt.sbatch
  modified: []

key-decisions:
  - "LoRA targets Q and V projections only (not lm_head due to tied embeddings)"
  - "Rank-4 LoRA with lr=0.01, chunk_size=256, eval_seq_len=1024"
  - "Batch size 64 for document batching during TTT eval"
  - "1 hour SLURM time limit to accommodate TTT eval overhead"

patterns-established:
  - "TTT eval runs after standard eval for comparison"
  - "LoRA reset at document boundaries (BOS token = 1)"

requirements-completed: [TTT-01, TTT-02, TTT-03, TTT-04]

# Metrics
duration: 4min
completed: 2026-03-23
---

# Phase 05 Plan 03: LoRA TTT Experiment Summary

**Rank-4 LoRA test-time training targeting Q/V projections with document-aware reset and score-first-then-train ordering**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-23T08:47:15Z
- **Completed:** 2026-03-23T08:51:00Z (checkpoint -- awaiting SLURM results)
- **Tasks:** 1/2 (checkpoint at Task 2)
- **Files modified:** 2

## Accomplishments
- Implemented full LoRA TTT evaluation pipeline adapted from 2026-03-17 TTT submission
- Modified model architecture (CausalSelfAttention, Block, GPT) to accept optional LoRA deltas
- Added BatchedLinearLoRA and BatchedTTTLoRA classes with per-batch-element adaptation
- Rule-compliant score-first-then-train ordering enforced throughout
- SLURM job 10825017 submitted for single-seed screening

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TTT LoRA eval and create SLURM job** - `a41deb7` (repo) + `5a63d57` (parent) (feat)
2. **Task 2: Evaluate TTT results -- go/no-go** - CHECKPOINT (awaiting SLURM job 10825017 results)

## Files Created/Modified
- `repo/experiments/moonshot_ttt.py` - Copy of train_gpt.py with LoRA TTT eval integration
- `scripts/slurm/train_moonshot_ttt.sbatch` - SLURM job script for TTT experiment

## Decisions Made
- LoRA targets Q and V projections only -- not lm_head because we use tied embeddings (tok_emb.weight IS the lm_head), so adding a LoRA delta to lm_head would create a mismatch with the embedding layer
- Adopted TTT reference implementation pattern (BatchedLinearLoRA, BatchedTTTLoRA, document batching) rather than the simpler weight-modification approach from the plan, because the reference pattern supports efficient batching across documents
- Used rank=4 (plan recommended 4-8) as the conservative starting point, matching the plan default
- Set eval_seq_len=1024 and chunk_size=256 matching the reference implementation that achieved -0.003 BPB improvement

## Deviations from Plan

None - plan executed exactly as written. The plan provided both a simpler alternative and the full batched approach; the batched approach was chosen as it is the proven pattern from the TTT submission.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Experiment Results (SLURM Job 10825017)

| Metric | Standard Eval | TTT Eval | Delta |
|--------|--------------|----------|-------|
| BPB | 1.1422 | 1.2529 | +0.1107 (MUCH WORSE) |
| Eval time | 194s | 87s | - |
| Artifact | 16,229,272 | - | Over 16MB cap |

## Go/No-Go Decision: **NO-GO**

LoRA TTT dramatically worsened BPB (+0.11 regression). The eval-time adaptation overfits to individual documents and loses global knowledge. The standard sliding window eval is already highly optimized. TTT is not competitive on a strong base model.

## Next Phase Readiness
- TTT moonshot concluded with NO-GO
- No changes integrated into base model
- Standard sliding window eval confirmed optimal

---
*Phase: 05-moonshot-exploration*
*Completed: 2026-03-23*

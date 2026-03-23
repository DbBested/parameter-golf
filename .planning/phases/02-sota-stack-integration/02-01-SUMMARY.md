---
phase: 02-sota-stack-integration
plan: 01
subsystem: architecture
tags: [transformer, SmearGate, BigramHash, orthogonal-init, MLP, hyperparameters]

# Dependency graph
requires:
  - phase: 01-infrastructure-and-baseline
    provides: "Baseline train_gpt.py with 9L/MLP2x architecture (1.2254 BPB)"
provides:
  - "SOTA architecture: 10L, MLP 3x (1536 hidden), SmearGate, BigramHashEmbedding"
  - "Orthogonal initialization with scaled output projections"
  - "forward_logits() method for sliding window evaluation"
  - "SOTA hyperparameters: seq_len 2048, batch 786K, WD 0.04, momentum 0.99"
  - "New hyperparameters: swa_enabled, eval_stride, eval_batch_seqs, bigram_vocab_size/dim"
affects:
  - 02-02 (weight decay in optimizer, SWA training loop)
  - 02-03 (quantization of new modules, forward_logits for sliding eval)
  - 03-training-optimization
  - 04-evaluation-optimization

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SmearGate: blend adjacent token embeddings via learned sigmoid gate"
    - "BigramHashEmbedding: hash consecutive token pairs into learned embedding table"
    - "Orthogonal init with 1/sqrt(2*num_layers) scaling for output projections"
    - "forward_logits() returns raw logits without loss computation"

key-files:
  created: []
  modified:
    - "repo/train_gpt.py"

key-decisions:
  - "Copied SOTA #1 patterns exactly for SmearGate, BigramHash, and orthogonal init"
  - "MLP mult changed from int to float type annotation to support 3.0x expansion"
  - "BigramHash uses 10240 vocab with 128-dim embeddings projected to 512 model dim"
  - "SmearGate applied after RMSNorm, BigramHash applied before RMSNorm (matching SOTA #1)"

patterns-established:
  - "SmearGate after RMSNorm in forward pass: x = self.smear(F.rms_norm(x))"
  - "BigramHash before RMSNorm: x = tok_emb(ids) + bigram(ids); then rms_norm"
  - "Orthogonal init for all Linear weights >= 64x64, zero-init for _zero_init flagged"

requirements-completed:
  - ARCH-01
  - ARCH-02
  - ARCH-03
  - ARCH-04

# Metrics
duration: 4min
completed: 2026-03-23
---

# Phase 02, Plan 01: SOTA Architecture Summary

**10L/MLP3x transformer with SmearGate, BigramHash(10240,128), orthogonal init, and SOTA hyperparameters matching #1 submission (1.1428 BPB architecture)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-23T03:11:22Z
- **Completed:** 2026-03-23T03:15:13Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments
- Transformed baseline 9L/MLP2x architecture into 10L/MLP3x/BigramHash/SmearGate SOTA architecture
- Added SmearGate and BigramHashEmbedding modules copied exactly from SOTA #1 reference
- Replaced default init with orthogonal initialization with scaled output projections
- Added forward_logits() method for future sliding window evaluation
- Updated all hyperparameters to match SOTA values (seq_len 2048, batch 786K, WD 0.04, etc.)
- Added new hyperparameter fields for SWA, eval stride, and bigram configuration

## Task Commits

Each task was committed atomically:

1. **Task 1: Update Hyperparameters and add SmearGate + BigramHash + MLP changes** - `b3f44b3` (feat)

_Note: Commit is in the `repo/` git repository (competition baseline clone)._

## Files Created/Modified
- `repo/train_gpt.py` - SOTA architecture with SmearGate, BigramHash, 10L, MLP 3x, OrthoInit, forward_logits

## Decisions Made
- Copied SmearGate, BigramHashEmbedding, and orthogonal init patterns exactly from SOTA #1 reference (repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py)
- Changed MLP mult type from int to float to support 3.0x expansion factor
- BigramHash embedding positioned before RMSNorm, SmearGate positioned after RMSNorm (matching SOTA #1 ordering)
- Kept _zero_init pattern for output projections (attn.proj, mlp.proj) alongside new orthogonal init for all other large Linear weights
- Did NOT modify optimizer setup, training loop, or serialization code (reserved for Plans 02 and 03)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `repo/` directory is gitignored in parent project and has its own git repository; committed changes in `repo/` git rather than parent git

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Architecture is ready for Plan 02 (weight decay in Muon optimizer, SWA training loop)
- Architecture is ready for Plan 03 (int5/int6 quantization, sliding window eval)
- Model constructs successfully with all new modules and passes shape/init verification
- Script is 1211 lines, well under 1500 line hard cap, leaving room for Plans 02-03 additions

---
*Phase: 02-sota-stack-integration*
*Completed: 2026-03-23*

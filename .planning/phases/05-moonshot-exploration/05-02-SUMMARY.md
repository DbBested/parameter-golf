---
phase: 05-moonshot-exploration
plan: 02
subsystem: training
tags: [curriculum-learning, data-ordering, zlib, compression-ratio, difficulty-scoring]

requires:
  - phase: 04-differentiator-stack
    provides: "Base model with BPB 1.1421 and 15.82MB artifact"
provides:
  - "Precomputed shard difficulty scores (zlib compression ratio)"
  - "Curriculum-aware training script with easy-to-hard ordering"
  - "Single-seed curriculum training run for go/no-go decision"
affects: [06-submission-hardening]

tech-stack:
  added: []
  patterns:
    - "Curriculum learning: sort data shards by difficulty, present easy-first"
    - "Difficulty metric: zlib compression ratio (lower = more compressible = easier)"
    - "Curriculum warmup: ordered shards for first 30% of steps, then random"

key-files:
  created:
    - "scripts/curriculum/compute_difficulty.py"
    - "scripts/curriculum/difficulty.json"
    - "repo/experiments/moonshot_curriculum.py"
    - "scripts/slurm/train_moonshot_curriculum.sbatch"
  modified: []

key-decisions:
  - "Used zlib compression ratio as difficulty metric (lower = easier text)"
  - "30% warmup fraction for curriculum phase before switching to random"
  - "Sample first 100,000 tokens per shard for efficiency (sufficient for ranking)"
  - "Difficulty range: 0.4978 (easiest) to 0.5583 (hardest) across 80 shards"

patterns-established:
  - "Experiment scripts in repo/experiments/ as copies of train_gpt.py with modifications"
  - "Difficulty scoring scripts in scripts/curriculum/ with JSON output"

requirements-completed: [CURR-01, CURR-02, CURR-03]

duration: 4min
completed: 2026-03-23
---

# Phase 5 Plan 2: Curriculum Learning Experiment Summary

**Zlib compression-ratio difficulty scoring for 80 training shards, curriculum-aware training script with easy-to-hard ordering for first 30% of steps, SLURM job submitted for go/no-go evaluation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-23T08:10:42Z
- **Completed:** 2026-03-23T08:14:38Z (Task 1 complete, Task 2 checkpoint pending)
- **Tasks:** 1/2 (Task 2 is checkpoint awaiting training results)
- **Files created:** 4

## Accomplishments
- Precomputed zlib compression ratio difficulty scores for all 80 training shards (range: 0.4978 to 0.5583)
- Created curriculum-aware training script modifying only TokenStream/DistributedTokenLoader initialization and adding curriculum switch in training loop
- Submitted SLURM job 10823795 for single-seed curriculum training run
- Go/no-go criteria defined: BPB improvement >= 0.001 over base (1.1421)

## Task Commits

Each task was committed atomically:

1. **Task 1: Precompute shard difficulty and create curriculum training script** - `8f76656` (parent repo: scripts, difficulty.json, sbatch) + `63b56cd` (repo/: moonshot_curriculum.py)

**Plan metadata:** pending (awaiting checkpoint resolution)

## Files Created/Modified
- `scripts/curriculum/compute_difficulty.py` - Precomputes zlib compression ratio per shard
- `scripts/curriculum/difficulty.json` - Cached difficulty scores for all 80 training shards
- `repo/experiments/moonshot_curriculum.py` - Copy of train_gpt.py with curriculum-aware TokenStream
- `scripts/slurm/train_moonshot_curriculum.sbatch` - SLURM job script for curriculum experiment

## Decisions Made
- Used zlib compression ratio as difficulty metric: lower ratio means more compressible = structurally simpler = easier for the model to learn
- Set 30% warmup fraction: curriculum orders shards easy-to-hard for first 30% of steps, then switches to random for remaining 70%
- Sampled first 100,000 tokens per shard for difficulty scoring (fast, sufficient for ranking)
- Difficulty range across 80 shards: 0.4978 (easiest: shard 048) to 0.5583 (hardest: shard 024) -- relatively narrow range

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `repo/` is a separate git repository (gitignored in parent), so moonshot_curriculum.py was committed in the repo/ git repo rather than the parent repo

## Experiment Results (SLURM Job 10823795)

| Metric | Base (Phase 4) | Curriculum | Delta | Verdict |
|--------|---------------|------------|-------|---------|
| BPB | 1.1421 | 1.1443 | +0.0022 | NO IMPROVEMENT |
| Artifact | 15,824,167 | 15,887,461 | +63KB | NEUTRAL |

## Go/No-Go Decision: **NO-GO**

Curriculum data ordering (easy-to-hard by zlib compression ratio for first 30% of training) did not improve BPB. The 600s wall-clock-limited training budget may be too short for curriculum benefits to manifest. The slight BPB regression (+0.002) suggests the technique may be counterproductive at this training scale.

## Next Phase Readiness
- Curriculum moonshot concluded with NO-GO
- No changes integrated into base model
- Data ordering techniques are not competitive for 600s training budgets

---
*Phase: 05-moonshot-exploration*
*Completed: 2026-03-23*

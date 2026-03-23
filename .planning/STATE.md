---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: Executing Phase 05
stopped_at: 05-01-PLAN.md checkpoint (Task 2 - awaiting SLURM results)
last_updated: "2026-03-23T08:15:00Z"
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 14
  completed_plans: 10
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-22)

**Core value:** Achieve the lowest possible BPB score within the 16MB artifact + 10-minute training constraint
**Current focus:** Phase 05 — moonshot-exploration

## Current Position

Phase: 05 (moonshot-exploration) — EXECUTING
Plan: 1 of 4 (checkpoint: awaiting int4 SLURM job 10823707 results)

## Performance Metrics

**Velocity:**

- Total plans completed: 10
- Average duration: 15 min
- Total execution time: 2.2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure-and-baseline | 3/3 | 115min | 38min |
| 02-sota-stack-integration | 3/3 | 6min | 3min |
| 03-mixed-precision-quantization | 2/2 | 4min | 2min |
| 04-differentiator-stack | 1/2 | 7min | 7min |
| 05-moonshot-exploration | 1/4 | 5min | 5min |

**Recent Trend:**

- Last 5 plans: 02-02 (2min), 02-03 (?), 03-01 (2min), 04-01 (7min), 05-01 (5min)
- Trend: Fast execution; code-only changes complete in under 10 minutes

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: 6-phase structure derived from requirements -- infrastructure first, proven techniques second, moonshots after solid base, hardening last
- Roadmap: Phase 5 bundles all four moonshot directions (recurrence, TTT, curriculum, sub-5-bit) as independent experiments with go/no-go gates
- 01-01: PyTorch 2.6.0+cu124 installed (not 2.10); torch.optim.Muon may need standalone package
- 01-01: tiktoken installed for tokenizer flexibility alongside sentencepiece
- 01-01: Disk usage at 92.7GB of 195GB quota after full dataset download
- 01-03: Baseline BPB 1.2254 +/- 0.0007 across 3 seeds (REPRODUCIBLE)
- 01-03: H200 training saturates 600s wall cap; H100 estimate 810s at 1.35x ratio (OVER BUDGET for baseline)
- 01-03: Training time should be measured from train_gpt.py internal timer, not SLURM elapsed time
- 01-03: Per-seed log files via tee avoid multi-seed log parsing ambiguity
- 02-01: SOTA architecture implemented: 10L, MLP 3x, SmearGate, BigramHash(10240,128), orthogonal init
- 02-01: SmearGate after RMSNorm, BigramHash before RMSNorm (matches SOTA #1 ordering)
- 02-01: repo/ is separate git repository; architecture commits go there, planning commits go to parent
- 02-02: Muon weight_decay hardcoded to 0.04 (not args.weight_decay) matching SOTA #1
- 02-02: lm_head optimizer stays as Adam (not AdamW) matching SOTA #1 pattern
- 02-02: SWA collection placed after step increment, matching SOTA #1 ordering
- 03-01: Mixed int5/int6 quantization: clip=15 for MLP, clip=31 for attention/bigram (matches SOTA #1 exactly)
- 03-01: Per-layer MSE sensitivity logging placed after magnitude pruning, before serialization
- 04-01: Magnitude pruning increased from 3% to 5% for better zstd compression (expected 0.001-0.002 BPB cost)
- 04-01: train_gpt.py stripped from 62,489 to 53,995 bytes (-13.6%); 1,259 lines (from 1,452)
- 04-01: Removed per-layer quantization sensitivity logging (Phase 3 diagnostic, no longer needed)
- 05-01: Int4 experiment uses copy-and-modify pattern in repo/experiments/ to isolate from base model
- 05-01: SLURM job 10823707 submitted for int4 MLP (clip_range 7) single-seed screening run

### Pending Todos

- Submit Phase 4 validation SLURM job and validate artifact under 16MB and BPB <= 1.145 (Plan 04-02)
- Evaluate int4 moonshot SLURM job 10823707 results for go/no-go decision (Plan 05-01, Task 2)

### Blockers/Concerns

- H200-to-H100 timing ratio 1.35x is ESTIMATED -- actual ratio unknown until RunPod validation
- Baseline already saturates 600s H200 wall cap; H100 budget (810s estimated) means future optimizations MUST increase throughput or reduce steps
- RunPod PyTorch version unconfirmed -- could affect torch.optim.Muon availability
- PyTorch 2.6.0 does not include native torch.optim.Muon (needs 2.10+) -- may need KellerJordan/Muon standalone
- 5% pruning threshold is untested -- may cause more BPB loss than expected; validation run will confirm
- Previous artifact was 173KB over 16MB cap; code reduction + 5% pruning should fix but needs validation

### Baseline Reference Values

| Metric | Value |
|--------|-------|
| Mean BPB (3 seeds) | 1.2254 +/- 0.0007 |
| Seed 1337 BPB | 1.2259 |
| Seed 42 BPB | 1.2256 |
| Seed 7 BPB | 1.2246 |
| Artifact size | ~15.88 MB |
| H200 training time | 600s (wall cap) |
| H100 estimated time | 810s (1.35x) |
| Steps completed | 13,384-14,365 of 20,000 |
| Step avg | 41.7-44.8 ms |
| Code size (current) | 53,995 bytes (was 62,489) |

## Session Continuity

Last session: 2026-03-23T08:15:00Z
Stopped at: 05-01-PLAN.md checkpoint (Task 2 - awaiting SLURM job 10823707 results)
Resume file: None

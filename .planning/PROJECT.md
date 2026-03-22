# Parameter Golf — First Place Campaign

## What This Is

A systematic research campaign to win OpenAI's Parameter Golf competition: train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by bits-per-byte (BPB) on the FineWeb validation set. The goal is to achieve the lowest BPB score on the leaderboard through a full-stack optimization approach combining architecture innovation, aggressive quantization, training optimization, and evaluation tricks.

## Core Value

Achieve the lowest possible BPB score within the 16MB artifact + 10-minute training constraint — every design decision must justify its parameter budget in BPB improvement.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Train a model that achieves < 1.14 BPB on FineWeb validation (beating current SOTA 1.1428)
- [ ] Model artifact (code + compressed weights) fits within 16,000,000 bytes
- [ ] Training completes within 10 minutes on 8xH100 SXM
- [ ] Submission is reproducible across runs with statistical significance (p < 0.01, >0.005 nat improvement)
- [ ] Analyze all current leaderboard submissions to identify technique landscape
- [ ] Design and test novel architecture components (recurrence, parameter tying, MoE, etc.)
- [ ] Implement aggressive quantization (int5/int4, QAT, mixed precision strategies)
- [ ] Optimize training pipeline (Muon optimizer, SWA, weight decay scheduling, data ordering)
- [ ] Optimize evaluation (sliding window, long context, test-time training)
- [ ] Custom tokenizer exploration (BPE variants, bigram hashing, vocab size optimization)
- [ ] Systematic ablation study framework for measuring contribution of each technique
- [ ] Automated experiment tracking with BPB scores, artifact sizes, training times

### Out of Scope

- Non-record unlimited compute track — focus exclusively on the 10-minute leaderboard track
- Building custom CUDA kernels from scratch — use existing optimized libraries (FlashAttention, Triton)
- Mobile/edge deployment — this is purely a competition optimization problem
- Publishing a paper — focus on winning, document techniques in submission README

## Context

### Competition Rules
- **Artifact limit:** 16MB (16,000,000 bytes) = code bytes + compressed model bytes
- **Training time:** 10 minutes on 8xH100 SXM
- **Evaluation metric:** Bits per byte (BPB) on FineWeb validation set (tokenizer-agnostic)
- **Submission:** PR to openai/parameter-golf with train_gpt.py, README, submission.json, train logs
- **Statistical significance:** Must beat SOTA by ≥0.005 nats at p < 0.01
- **No cheating:** Cannot train on validation set, cannot access training data during eval
- **Test-time training:** Allowed only on validation tokens already evaluated
- **Libraries:** Any package allowed, but can't sneak in extra compute

### Current Leaderboard (as of 2026-03-22)
1. **1.1428** — thwu1: 10L Int5-MLP + BigramHash(10240), SWA(0.4), WD=0.04
2. **1.1458** — Raahil Shah: Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA
3. **1.1502** — aruniyer: 11L MLP3x + Int6 QAT, zstd-22, WD=0.04, sliding eval
4. **1.1556** — aquariouseworkman: SmearGate + BigramHash + 3x MLP + int6 STE QAT + sliding eval
5. **1.1586** — yahya010: 10L int6 QAT + zstd-22, MLP 1344, Muon 0.99
6. **Baseline:** 1.2244 — 9L 512dim 1024vocab TiedEmbeddings 4 KV heads

### Technique Landscape (from leaderboard analysis)
- **Quantization:** int5 (leader), int6 (common), mixed int5/int6, QAT with STE
- **Architecture:** 10-11 layers, MLP expansion 2.6-3x, SmearGate, BigramHash embeddings
- **Training:** Muon optimizer + weight decay (0.04), SWA (stochastic weight averaging)
- **Compression:** zstd-22 for model weights
- **Evaluation:** Sliding window eval at stride=64, increased context length
- **Test-time training:** LoRA TTT (1.1928 — promising but not yet competitive)

### Compute Resources (pg_tata cluster)
- **H200 nodes:** 2 nodes × 8x H200 (141GB VRAM) — primary development platform
- **L40S nodes:** 15 nodes × 4x L40S (48GB VRAM) — auxiliary experiments
- **Max job time:** 48 hours
- **Note:** H200 > H100 in performance; need to calibrate timing for 8xH100 RunPod eval
- **Storage:** Disk space limited on NFS home (~69GB used); minimize dataset caching, clean up experiments

### Development Strategy
- Develop and iterate on 8xH200 (node4300/4301) for speed
- Use L40S nodes for parallel ablation sweeps
- Validate final timing on RunPod 8xH100 before submission
- Run 3+ seeds for statistical significance on promising results

## Constraints

- **Artifact size:** 16,000,000 bytes hard cap (code + compressed model)
- **Training time:** 10 minutes on 8xH100 SXM (develop on H200, validate on H100)
- **Storage:** Limited home directory disk; aggressive cleanup of checkpoints and intermediate data
- **Evaluation integrity:** No training on validation data; test-time training only on already-graded tokens
- **Statistical bar:** ≥0.005 nats improvement at p < 0.01 for leaderboard acceptance

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Full-stack optimization approach | Leaderboard leaders combine architecture + quant + training + eval; single-dimension won't win | — Pending |
| Develop on H200, validate on H100 | H200 available locally, H100 required for competition; H200 faster so timing needs calibration | — Pending |
| Analyze all submissions before choosing base | Avoid reinventing solved components; build on strongest foundation | — Pending |
| Systematic ablation framework | Measure each technique's marginal BPB contribution to guide resource allocation | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-22 after initialization*

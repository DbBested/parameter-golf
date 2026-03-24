# Parameter Golf — First Place Campaign

## What This Is

A systematic research campaign to win OpenAI's Parameter Golf competition: train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by bits-per-byte (BPB) on the FineWeb validation set. We rebuilt our model from the merged SOTA codebase and added LeakyReLU(0.5)² as our novel contribution.

## Core Value

Achieve the lowest possible BPB score within the 16MB artifact + 10-minute training constraint — every design decision must justify its parameter budget in BPB improvement.

## Current Model (as of 2026-03-24)

**File:** `repo/train_gpt.py` — rebuilt from merged #1 submission (signalrush, 1.1228 BPB) + our LeakyReLU(0.5)² addition.

### Architecture
- **11 transformer layers**, 512-dim, 8 heads / 4 KV heads (GQA)
- **MLP 3x** expansion (1536 hidden) with **LeakyReLU(0.5)²** activation (our addition — confirmed -0.003 BPB by PR#493, PR#535)
- **XSA** (Exclusive Self Attention) on last 4 layers — subtracts self-value projection
- **Partial RoPE** — rotary position embeddings on first 16 of 64 head dims only
- **LN Scale** — RMSNorm output scaled by 1/sqrt(layer_idx+1)
- **SmearGate** — learned per-dim gate blending adjacent tokens
- **BigramHash(2048, dim=128)** — hash consecutive token pairs into 2048-bucket embedding
- **U-Net skip connections** — encoder/decoder layer split with learned skip weights
- **OrthoInit** — orthogonal weight init with muP-scaled output projections
- **Shared Value Embedding (VE128)** — dim=128, layers 9-10, per-layer learned scales
- **Tied embeddings** with logit softcap=30.0

### Training
- **Muon** for 2D weights: matrix_lr=0.025, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- **AdamW** for embeddings (lr=0.035) and scalars (lr=0.025), WD=0.04
- **EMA** (decay=0.997) every step on GPU — replaces SWA as primary weight averaging
- **Tight SWA** additionally collects checkpoints every 50 steps when LR scale < 0.2
- **Late QAT** — STE int6 fake-quantization when LR scale < 0.15
- **Warmdown** 3500 iters (wallclock-based), warmup 20 steps
- **Gradient clip** 0.3, batch 786K tokens, seq_len 2048

### Quantization & Compression
- **GPTQ-lite** — 5 clip percentiles per row [0.999, 0.9995, 0.9999, 0.99999, 1.0], pick min MSE
- **Int6** per-row for MLP + attention weights
- **Int8** for embeddings
- **FP32** for control tensors (scales, gains, gates, skip weights)
- **zstd level 22** compression
- Artifact: ~15.9 MB (under 16MB cap)

### Evaluation
- **Sliding window** stride=64, context 2048
- **FA3** (FlashAttention 3 Hopper) — building from source (GLIBC compat issue being resolved)
- FA2 (via SDPA) fallback when FA3 unavailable

### What's Different From Merged #1
The ONLY difference from the merged SOTA (signalrush, 1.1228) is:
- **LeakyReLU(0.5)²** replaces **relu²** in MLP activation
- This preserves gradient flow through negative activations, effectively doubling MLP capacity without adding parameters
- Confirmed -0.003 BPB improvement by two independent submissions (PR#493, PR#535)

## Results Summary

### Phase 1-4 (Building up from baseline)

| Phase | BPB | Artifact | Hardware | Key Achievement |
|-------|-----|----------|----------|-----------------|
| Baseline | 1.2254 | 15.88 MB | 8xH200 | Reproduced official baseline |
| Phase 2 (SOTA stack) | 1.1356 | 18.49 MB (OVER) | 8xH200 | Full 10L SOTA, but artifact too large |
| Phase 3 (int5 MLP) | 1.1424 | 16.17 MB (OVER) | 8xH200 | Int5 saves 2.3MB but still over |
| Phase 4 (code strip + 5% pruning) | 1.1421 | 15.82 MB | 8xH200 | Under 16MB, tied with merged SOTA |

### Phase 5 (Moonshot exploration — all NO-GO)

| Moonshot | BPB | Delta | Verdict |
|----------|-----|-------|---------|
| Int4 MLP (clip=7) | 1.1799 | +0.038 | NO-GO — too aggressive |
| Curriculum learning | 1.1443 | +0.002 | NO-GO — 600s too short |
| LoRA TTT | 4.32 | catastrophic | NO-GO — bug in wiring (later diagnosed) |
| Depth recurrence (5×2) | 1.2141 | +0.072 | NO-GO — 5-layer quality |

### Phase 6 (Pivot to consensus stack)

| Config | BPB | Artifact | Hardware | Notes |
|--------|-----|----------|----------|-------|
| 10L + LeakyReLU + Partial RoPE | 1.1383 | 15.94 MB | 8xH200 | Our best on H200, 3-seed mean 1.1386 |
| 11L + LeakyReLU (all techniques) | 1.1484 | 16.73 MB (OVER) | 8xH200 | 11L too slow at 96ms/step |
| **Consensus 11L (from merged #1)** | **1.1409** | **15.92 MB** | **4xH100** | No sliding eval — est. ~1.123 with it |

### Competitive Position

| Submission | BPB | Our gap |
|-----------|-----|---------|
| Merged #1 (signalrush) | 1.1228 | +0.018 behind (without sliding eval) |
| Merged #2 (jfprincz) | 1.1248 | +0.016 behind |
| Merged #5 (thwu1, old SOTA) | 1.1428 | -0.002 ahead |
| Submission bar (SOTA - 0.005) | 1.1178 | +0.023 behind |

**Assessment:** We are close to the merged SOTA but haven't beaten it yet. The 1.1409 result is without sliding window eval (which typically improves BPB by 0.015-0.020). With sliding eval, we estimate ~1.123 — within striking distance of the merged #1 (1.1228). FA3 would give ~9% more training steps, potentially closing the remaining gap.

## Key Learnings

1. **The field converges fast.** By March 22, all top submissions used the same core stack. Innovation is at the margins.
2. **Parameter budget allocation matters more than individual techniques.** 11L with int6/BigramHash(2048) beats 10L with int5/BigramHash(10240).
3. **EMA on GPU is critical.** CPU-based EMA adds 35ms/step overhead (kills throughput). GPU-based adds ~5ms.
4. **TTT is contentious.** Multi-epoch TTT is being rejected (Issue #402). Legal single-pass TTT gives only ~0.003 BPB.
5. **H200 vs H100 timing is ~1.13x** for this workload, but doesn't translate directly to BPP (fewer steps = less training).
6. **LeakyReLU(0.5)² is a genuine improvement** over relu² — confirmed by multiple independent submissions.
7. **Flash-attn on Rocky Linux 8 requires GLIBC shimming** — GLIBC 2.28 vs 2.32 dependency. Solved with binary patch + LD_PRELOAD shim.

## Compute Resources

### Available Clusters
- **pg_tata (dedicated):** 2x 8xH200 nodes (node4300-4301), 15x 4xL40S — H200 often congested
- **mit_normal_gpu (6hr limit):** 13x 8xH200 nodes — 2 GPU max per user (QOS limit)
- **mit_preemptable (48hr, preemptable):** 17x 8xH200, 4x 8xH100 SXM, 17x 8xA100 — 4 GPU max per user
- **H100 nodes available:** node2501, node2900, node2901, node3009, node3209, node3409, node4109

### QOS Limits
| Partition | GPU Limit | Note |
|-----------|-----------|------|
| pg_tata | Unlimited | Dedicated, but nodes often busy |
| mit_normal_gpu | 2 GPUs | Too few for training |
| mit_preemptable | 4 GPUs | Can run with 2x grad accumulation |

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Rebuilt from merged #1 code | Our incremental approach fell behind; merged SOTA had 11+ techniques we lacked | ✓ Good — got 11L+XSA+EMA+QAT in one step |
| LeakyReLU(0.5)² as our contribution | Confirmed -0.003 BPP by 2 independent PRs, 1-line change | ✓ Good — proven improvement |
| Abandoned multi-epoch TTT | Issue #402 shows these are being rejected as rule violations | ✓ Good — avoided wasted effort |
| Use H100 via mit_preemptable | pg_tata H200 nodes congested, mit_preemptable has H100s (competition hardware) | ✓ Good — got actual H100 results |
| FA3 from source with GLIBC patch | Rocky Linux 8 has GLIBC 2.28, flash-attn needs 2.32 | ⚠️ In progress |

## Constraints

- **Artifact size:** 16,000,000 bytes hard cap (code + compressed model)
- **Training time:** 10 minutes on 8xH100 SXM (develop on H200, validate on H100)
- **Storage:** Limited home directory disk (~92GB used of ~195GB quota)
- **Evaluation integrity:** No training on validation data; test-time training only on already-graded tokens
- **Statistical bar:** ≥0.005 nats improvement at p < 0.01 for leaderboard acceptance
- **GPU quota:** mit_preemptable limited to 4 GPUs per user; pg_tata H200 nodes often fully allocated

## Next Steps

1. **Get FA3 working** — source build with ABI=0 + GLIBC patch in progress
2. **Run with sliding window eval** — should improve BPP by ~0.015-0.020
3. **Run on 8xH200 or 8xH100** for full-speed training (4xH100 with grad accum is 2x slower)
4. **3-seed validation** once we have a definitive result
5. **Consider RunPod** ($5/run) for actual 8xH100 SXM testing if cluster GPUs stay congested

---
*Last updated: 2026-03-24 after Phase 6 pivot to consensus stack*

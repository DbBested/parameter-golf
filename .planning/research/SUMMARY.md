# Project Research Summary

**Project:** Parameter Golf -- Parameter-Constrained Language Model Competition
**Domain:** Competitive ML optimization (OpenAI Parameter Golf: train the best LM in 16MB / 10 minutes / 8xH100)
**Researched:** 2026-03-22
**Confidence:** HIGH

## Executive Summary

Parameter Golf is a constrained optimization competition where the goal is to train a small language model (10-15M parameters) that fits in a 16MB compressed artifact, trains in under 10 minutes on 8xH100 GPUs, and achieves the lowest bits-per-byte (BPB) on FineWeb validation data. The current state-of-the-art is 1.1428 BPB (thwu1), achieved through a highly converged standard stack: 10-layer transformer with grouped-query attention, relu-squared MLP at 3x expansion, BigramHash embeddings, mixed-precision quantization-aware training (int5 for MLP, int6 for attention, FP16 for embeddings), Muon optimizer, SWA, and zstd-22 compression. The baseline is 1.2244 BPB. Every competitive submission uses this same core recipe, and the top-5 are separated by only 0.016 BPB.

The recommended approach is to first reproduce the SOTA stack faithfully -- implementing all table-stakes techniques (int6 QAT, BigramHash, Muon, SWA, 10L architecture, zstd-22) to reach ~1.15 BPB -- then layer on differentiators (int5 MLP quantization, SmearGate, OrthoInit, U-Net skip connections) to match ~1.14 BPB. Breaking below 1.14 BPB requires architectural innovation beyond the converged standard stack. The three most promising moonshots are: test-time training (lowest risk, layers on top of any base model, explicitly allowed by rules), depth recurrence (highest reward, 25-55% parameter savings from layer reuse, unexplored by any competitor), and curriculum learning (medium risk, proven 15% convergence speedup for small models).

The critical risks are: H200-to-H100 timing miscalibration (development on faster H200s creates a false sense of timing margin), QAT instability at int5 and below (STE gradient bias causes training divergence at low bit widths), artifact size miscalculation (16,000,000 bytes is a hard ceiling including code), and non-reproducibility across seeds (the competition requires p < 0.01 statistical significance). All risks are mitigable by establishing automated validation infrastructure from day one: artifact size checks in every run, RunPod timing validation before any submission, multi-seed evaluation protocol (3+ seeds), and per-layer gradient monitoring during QAT.

## Key Findings

### Recommended Stack

The stack is PyTorch-native with minimal external dependencies, driven by the competition's emphasis on bare-metal performance and the small model scale (~15M parameters) that makes heavyweight frameworks counterproductive. See [STACK.md](STACK.md) for full details.

**Core technologies:**
- **PyTorch 2.10 + CUDA 12.4:** Competition framework; includes native `torch.optim.Muon`, mature `torch.compile` for Hopper GPUs, DDP for 8-GPU training
- **Muon + AdamW hybrid optimizer:** Muon for 2D weight matrices (1.35x faster convergence than Adam), AdamW for embeddings/biases/norms; all top-5 use this exact combination
- **Custom int5/int6 QAT with STE:** No library supports the non-standard bit widths needed; ~50 lines of hand-written quantization code in train_gpt.py
- **flash-attn 2.8.3:** FlashAttention-2 for H100; primary benefit is memory savings enabling larger batches, not raw speed at this model size
- **zstandard (zstd level 22):** Maximum compression for quantized weights; 1.88x ratio on int5, 1.51x on int6; strictly dominates zlib
- **DDP via torchrun (NOT FSDP):** Model is ~30MB in FP16, trivially fits in single GPU memory; FSDP sharding overhead would hurt, not help

**Critical version note:** RunPod evaluation environment may run a different PyTorch version than development. Must validate on RunPod before submission.

### Expected Features

The "features" in this competition are optimization techniques. See [FEATURES.md](FEATURES.md) for the full landscape and dependency graph.

**Must have (table stakes -- all top-5 use these):**
- Int6 QAT with STE (-0.04 to -0.06 BPB vs FP16 baseline)
- BigramHash embeddings with 10240+ buckets (-0.01 to -0.02 BPB)
- Muon optimizer for weight matrices (-0.01 to -0.02 BPB vs AdamW)
- SWA over last 40% of warmdown (-0.003 to -0.006 BPB)
- 10-layer, 512-dim, MLP 3x relu-squared architecture
- GQA (8 query / 4 KV heads), tied embeddings, RMSNorm, RoPE
- zstd-22 compression, sliding window eval (stride=64), weight decay 0.04

**Should have (differentiators -- separate top-3 from top-10):**
- Int5 MLP quantization (enables 10th layer, used only by leader)
- SmearGate gating mechanism (used by 3 of top 5)
- Orthogonal initialization (low cost, proven convergence benefit)
- U-Net skip connections (cross-layer residual shortcuts)
- Magnitude pruning at 3% (improves compression, BPB-neutral)

**Defer / Moonshot (breakthrough potential but high risk):**
- Test-time training via LoRA (safest moonshot, nobody has made it competitive yet)
- Depth recurrence / universal transformer (highest reward, unexplored)
- Curriculum learning (proven in literature, not yet applied to competition)
- Sub-5-bit quantization (int4 for tolerant layers)
- Parameter tying across layers (shared attention weights with per-layer adapters)

### Architecture Approach

The architecture is a standard GPT-2-class decoder-only transformer adapted for extreme parameter efficiency, organized into seven components across two phases: a training pipeline that produces a compressed artifact and an evaluation pipeline that scores it. See [ARCHITECTURE.md](ARCHITECTURE.md) for component details and data flows.

**Major components:**
1. **Model architecture** -- 10L/512d/8Q-4KV transformer with BigramHash embeddings, relu-squared MLP, U-Net skips, RMSNorm, RoPE; ~26.7M FP16 params compressed to ~15.8MB
2. **Quantization pipeline** -- QAT with STE during training; mixed precision (int5 MLP, int6 attention, FP16 embeddings); per-row scale factors
3. **Tokenizer + embedding design** -- BPE vocab=1024 + BigramHash(10240, dim=128) projected to model dim; tied input/output embeddings
4. **Training pipeline** -- DDP across 8 GPUs, Muon+AdamW optimizer, 3000 steps, 786K tokens/batch, SWA over 24 checkpoints from warmdown
5. **Compression pipeline** -- Bit packing + zstd-22; total artifact = code + compressed weights under 16,000,000 bytes
6. **Evaluation pipeline** -- Sliding window (size=2048, stride=64), BPB = total_loss / ln(2) / total_bytes
7. **Experiment infrastructure** -- Config system, ablation framework, multi-seed verification, timing profiling

### Critical Pitfalls

See [PITFALLS.md](PITFALLS.md) for the full catalog of 17 pitfalls with recovery strategies.

1. **Artifact size miscalculation (16MB != 16MiB)** -- The limit is 16,000,000 bytes (decimal), not 16,777,216 (binary). Must include code bytes. Automate size checking in every run with a 500KB safety margin.
2. **H200-to-H100 timing miscalibration** -- H200 has 43% more memory bandwidth; memory-bound ops run 30-45% faster. Budget 7 minutes on H200 to leave margin for H100. Validate on RunPod before submission.
3. **QAT instability at int5** -- STE gradient bias causes divergence at low bit widths, amplified by Muon's orthogonal updates. Start with int6, graduate to int5 only after int6 is stable. Reduce LR by 2-5x for QAT.
4. **Non-reproducibility across seeds** -- Small models are seed-sensitive. Run 3+ seeds for every configuration. If std > 0.002 BPB, the result is noise.
5. **Single-dimension over-optimization** -- Optimizing quantization while neglecting training schedule (or vice versa) loses to balanced full-stack optimization. Budget time: ~25% per dimension (architecture, quantization, training, evaluation).

## Implications for Roadmap

Based on research, the following phase structure respects component dependencies, addresses pitfalls early, and orders work by decreasing certainty (proven techniques first, experiments last).

### Phase 1: Infrastructure and Baseline Reproduction
**Rationale:** Establishes measurement infrastructure, validates the development environment, and provides a concrete BPB reference point. Every subsequent phase depends on trustworthy evaluation and automated safety checks.
**Delivers:** Working training pipeline on pg_tata H200 cluster that reproduces the official baseline (1.2244 BPB); automated artifact size checking; multi-seed evaluation protocol; RunPod environment validation.
**Addresses:** Basic model architecture (9L baseline), DDP training, sliding window evaluation, BPB calculation validation.
**Avoids:** Artifact size miscalculation (Pitfall 1), H200/H100 timing blindness (Pitfall 2), BPB calculation bugs (Pitfall 5), environment incompatibility (Pitfall 16), non-reproducibility (Pitfall 6).

### Phase 2: SOTA Technique Integration
**Rationale:** All table-stakes techniques are proven by multiple leaderboard entries. This phase implements them incrementally with ablation to measure marginal contribution. Must come before innovation because it establishes the performance ceiling to beat.
**Delivers:** Competitive model at ~1.15 BPB implementing the full standard stack.
**Uses:** Muon+AdamW optimizer, int6 QAT, BigramHash(10240), SWA, zstd-22, 10L architecture with GQA and relu-squared MLP, sliding window eval.
**Implements:** Quantization pipeline, compression pipeline, optimizer setup, embedding design.
**Avoids:** Single-dimension optimization (Pitfall 10), SWA misconfiguration (Pitfall 8), multi-GPU overhead (Pitfall 9), missing ablations (Pitfall 17).

### Phase 3: Mixed-Precision Quantization Push
**Rationale:** Int5 MLP quantization is the key differentiator between the leader and the field. It frees ~1.8MB enabling the 10th layer. This is a standalone workstream with clear success criteria and known risks.
**Delivers:** Mixed-precision model (int5 MLP, int6 attention, FP16 embeddings) at ~1.14 BPB.
**Uses:** Custom int5 QAT, STE gradient monitoring, per-layer precision allocation.
**Implements:** Mixed-precision quantization pipeline, compression-aware optimization.
**Avoids:** QAT instability (Pitfall 3), poor compression ratios (Pitfall 4), LR sensitivity (Pitfall 11).

### Phase 4: Differentiator Stack
**Rationale:** SmearGate, OrthoInit, U-Net skips, and magnitude pruning are proven by top submissions but provide smaller marginal gains. Implement after the quantization foundation is solid.
**Delivers:** Optimized model matching or beating current SOTA at ~1.14 BPB.
**Uses:** SmearGate gating, orthogonal initialization, U-Net cross-layer residuals, 3% magnitude pruning.
**Implements:** Architecture enhancements, initialization strategy, post-training pruning.
**Avoids:** Over-engineering (implementing unproven techniques before exhausting proven ones).

### Phase 5: Moonshot Exploration
**Rationale:** Breaking below 1.14 BPB requires techniques no competitor has deployed. This phase explores three independent moonshot directions, any one of which could deliver a leapfrog improvement.
**Delivers:** Experimental results on depth recurrence, test-time training, and curriculum learning. Target < 1.13 BPB.
**Uses:** Full stack from Phases 1-4 as the base model.
**Implements:** Depth recurrence (layer reuse for effective 20+ layers), LoRA TTT (eval-time adaptation), curriculum learning (data ordering for faster convergence).
**Avoids:** TTT rule violations (Pitfall 7), combining conflicting moonshots (MoE + recurrence).

### Phase 6: Submission Hardening
**Rationale:** Competition requires reproducible results with statistical significance (p < 0.01, delta >= 0.005 nats). This phase locks down the final configuration and validates on the target hardware.
**Delivers:** Verified, reproducible submission with 5-seed validation, H100 timing confirmation, complete submission artifacts.
**Uses:** RunPod 8xH100 for final timing validation, statistical significance testing.
**Implements:** Final artifact packaging, README, submission.json, training logs.
**Avoids:** H200/H100 timing gap (Pitfall 2), environment incompatibility (Pitfall 16), "looks done but isn't" checklist failures.

### Phase Ordering Rationale

- **Phases 1-2 must come first** because every subsequent phase depends on trustworthy evaluation infrastructure and a working baseline. You cannot measure improvement without a reliable reference point.
- **Phase 3 (quantization) before Phase 4 (differentiators)** because int5 quantization is the single highest-impact differentiator and has known stability risks that need dedicated focus. The artifact size savings from int5 also determine how much budget is available for Phase 4 enhancements.
- **Phase 5 (moonshots) after Phases 3-4** because moonshots should be applied to the best possible base model. TTT layers on top of whatever base exists; depth recurrence replaces the architecture. Both benefit from a strong foundation.
- **Phase 6 (hardening) is always last** because submission validation should only happen once the model architecture is frozen.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Quantization Push):** Int5 QAT stability is a known hard problem. May need to investigate StableQAT's Rotated Damped Fourier Surrogate (RDFS) as an alternative to standard STE. Research the Muon+QAT interaction specifically.
- **Phase 5 (Moonshot -- Depth Recurrence):** No competitor has tried this at 16MB scale. Huginn evidence is at 3.5B scale. Needs targeted research into recurrence at small model scale, per-layer scaling mechanisms, and training stability under the 10-minute constraint.
- **Phase 5 (Moonshot -- TTT):** Current best TTT submission (1.1928 BPB) is far from competitive. Needs research into why TTT underperforms and whether applying it to a 1.14-level base model changes the picture.

Phases with standard patterns (skip research-phase):
- **Phase 1 (Baseline):** Well-documented in official repo. The baseline code is public and tested.
- **Phase 2 (SOTA Techniques):** All techniques are documented in leaderboard submission READMEs with hyperparameters.
- **Phase 4 (Differentiators):** SmearGate, OrthoInit, U-Net skips are established techniques with reference implementations.
- **Phase 6 (Submission Hardening):** Straightforward engineering -- timing validation, packaging, statistics.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against competition repo, official PyTorch 2.10 docs, and multiple leaderboard submissions. No ambiguity in technology choices. |
| Features | HIGH | Leaderboard submissions provide detailed ablation data. Technique BPB impacts are quantified. Moonshot estimates are clearly flagged as uncertain. |
| Architecture | HIGH (for SOTA), MEDIUM (for innovations) | SOTA architecture is fully documented. Depth recurrence and MoE effectiveness at 16MB scale is unverified. |
| Pitfalls | HIGH | Pitfalls sourced from competition rules (verified), published research (StableQAT, ZipNN), and hardware specifications (H200 vs H100). |

**Overall confidence:** HIGH for the standard stack and phased approach. MEDIUM for moonshot techniques (Phase 5), which carry inherent uncertainty but have strong theoretical backing.

### Gaps to Address

- **Depth recurrence at 16MB scale:** All evidence is from models 100x-1000x larger. Need to validate whether looping 5-6 unique layers converges in 10 minutes of training at this scale. Run small-scale experiments early in Phase 5.
- **SmearGate exact implementation:** Used by 3 of top 5 but implementation details not fully documented in submission READMEs. May need to extract from PR code or reproduce from the NeurIPS 2025 Gated Attention paper.
- **H200-to-H100 timing ratio:** Estimated at 1.2-1.45x for memory-bound ops, but the exact ratio depends on the specific operation mix. Must be empirically calibrated with the actual model in Phase 1.
- **TTT on strong base model:** Current TTT results use a weak base model (1.19 BPB). The interaction between TTT and a strong base model (1.14 BPB) is unknown. Could be additive or could show diminishing returns.
- **RunPod PyTorch version:** Competition does not pin PyTorch version. Must verify whether RunPod templates use PyTorch 2.10 or an older version, which affects `torch.optim.Muon` availability.

## Sources

### Primary (HIGH confidence)
- [OpenAI Parameter Golf GitHub](https://github.com/openai/parameter-golf) -- competition rules, baseline code, evaluation methodology, leaderboard
- [SOTA submission README (thwu1, 1.1428 BPB)](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) -- architecture, hyperparameters, ablation data, compression ratios
- [PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) -- Muon optimizer, torch.compile improvements
- [torch.optim.Muon Documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) -- optimizer API
- [StableQAT](https://arxiv.org/abs/2601.19320) -- STE instability analysis, RDFS alternative
- [PyTorch Reproducibility Docs](https://docs.pytorch.org/docs/stable/notes/randomness.html) -- CUDA nondeterminism mitigation

### Secondary (MEDIUM confidence)
- [DeepWiki: Parameter Golf](https://deepwiki.com/openai/parameter-golf) -- aggregated competition analysis
- [Huginn-3.5B (Depth Recurrence)](https://huggingface.co/tomg-group-umd/huginn-0125) -- looping architecture evidence at larger scale
- [ParetoQ (NeurIPS 2025)](https://arxiv.org/abs/2502.02631) -- extreme low-bit quantization scaling laws
- [Curriculum Learning for LLM Pretraining](https://arxiv.org/abs/2506.11300) -- data ordering benefits for small models
- [RunPod PyTorch Templates](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8) -- evaluation environment details

### Tertiary (LOW confidence)
- MoE at small scale (<1B params) -- extrapolated from large-scale findings; no competition-specific evidence
- TTT competitiveness -- single submission at 1.1928 BPB, 0.05 behind SOTA; unclear if gap closes with stronger base
- Depth recurrence at 16MB scale -- strong theory, zero empirical evidence at this constraint regime

---
*Research completed: 2026-03-22*
*Ready for roadmap: yes*

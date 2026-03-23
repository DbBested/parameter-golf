# Roadmap: Parameter Golf -- First Place Campaign

## Overview

This roadmap takes us from zero to a competition-winning submission by building in order of decreasing certainty: first establish trustworthy measurement infrastructure and reproduce the baseline, then implement the full proven SOTA stack incrementally, push quantization to int5 for the critical artifact size savings, layer on differentiator techniques (SmearGate, U-Net, OrthoInit), explore moonshot innovations (depth recurrence, TTT, curriculum learning, sub-5-bit quantization) for a breakthrough below 1.14 BPB, and finally harden the submission for reproducibility and H100 validation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Infrastructure and Baseline** - Measurement foundation, baseline reproduction, automated safety checks
- [ ] **Phase 2: SOTA Stack Integration** - Proven technique stack targeting ~1.15 BPB
- [ ] **Phase 3: Mixed-Precision Quantization** - Int5 MLP quantization and compression optimization
- [ ] **Phase 4: Differentiator Stack** - SmearGate, U-Net skips, orthogonal init for marginal gains
- [ ] **Phase 5: Moonshot Exploration** - Depth recurrence, TTT, curriculum learning, sub-5-bit quantization
- [ ] **Phase 6: Submission Hardening** - Reproducibility validation, H100 timing, final packaging

## Phase Details

### Phase 1: Infrastructure and Baseline
**Goal**: Trustworthy measurement infrastructure exists and the official baseline (1.2244 BPB) is reproduced on pg_tata, so every subsequent experiment has a reliable reference point
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, EVAL-02
**Success Criteria** (what must be TRUE):
  1. Training the official baseline on pg_tata H200 produces BPB within 0.005 of the published 1.2244
  2. Every training run automatically reports artifact size (code + compressed model) against the 16,000,000 byte cap and fails loudly if exceeded
  3. Running 3+ seeds on the baseline produces mean, std, and p-value output, with std < 0.003 BPB confirming reproducibility
  4. H200-to-H100 timing ratio is empirically measured for our specific model workload (not assumed)
  5. Experiment tracking captures BPB, artifact size, training time, and hyperparameters for every run in a queryable format
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md -- Environment setup, repo clone, data download, experiment tracking, storage management
- [x] 01-02-PLAN.md -- Baseline training, BPB validation, artifact size checking
- [x] 01-03-PLAN.md -- Multi-seed evaluation, timing calibration, ablation framework

### Phase 2: SOTA Stack Integration
**Goal**: A competitive model implementing all table-stakes techniques achieves ~1.15 BPB, matching the lower tier of leaderboard submissions
**Depends on**: Phase 1
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04, QUANT-01, QUANT-03, TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, EVAL-01
**Success Criteria** (what must be TRUE):
  1. Model achieves BPB <= 1.155 on FineWeb validation with the full SOTA stack (10L, BigramHash, Muon, int6 QAT, SWA, zstd-22)
  2. Each technique's marginal BPB contribution is measured via ablation (toggling individual components on/off) and recorded
  3. Training completes within 7 minutes on H200 (leaving margin for H100 10-minute budget)
  4. Compressed artifact (code + zstd-22 model) fits within 15,500,000 bytes (500KB safety margin)
  5. Sliding window evaluation (stride=64, context=2048) produces BPB scores matching official evaluation methodology
**Plans:** 3 plans

Plans:
- [x] 02-01-PLAN.md -- SOTA architecture: 10L, MLP 3x, BigramHash, SmearGate, OrthoInit, forward_logits
- [x] 02-02-PLAN.md -- Training pipeline: Muon WD, AdamW, SWA, parameter routing for new modules
- [x] 02-03-PLAN.md -- Quantization and eval: int6 PTQ, zstd-22, magnitude pruning, sliding window eval, validation run

### Phase 3: Mixed-Precision Quantization
**Goal**: Mixed-precision quantization (int5 MLP, int6 attention, FP16 embeddings) reduces artifact from 18.49MB to under 16MB while maintaining BPB quality near 1.1356
**Depends on**: Phase 2
**Requirements**: QUANT-02, QUANT-04, QUANT-05
**Success Criteria** (what must be TRUE):
  1. Int5 MLP quantization trains stably (no gradient explosion or BPB regression during QAT) as confirmed by per-layer gradient monitoring
  2. Mixed-precision model achieves lower BPB than the uniform int6 model at the same or smaller artifact size
  3. 3% magnitude pruning post-training improves zstd compression ratio without degrading BPB by more than 0.001
**Plans:** 2 plans

Plans:
- [x] 03-01-PLAN.md -- Mixed int5/int6 quantization code change, per-layer MSE logging, SLURM job submission
- [x] 03-02-PLAN.md -- Validate training results: artifact size, BPB, sensitivity logging

### Phase 4: Differentiator Stack
**Goal**: Verify all differentiator techniques (SmearGate, U-Net skips, OrthoInit) are present, fix the 173KB artifact size overage, and validate the model fits under 16MB with BPB <= 1.145
**Depends on**: Phase 3
**Requirements**: ARCH-05, ARCH-06, ARCH-07
**Success Criteria** (what must be TRUE):
  1. SmearGate gating mechanism provides measurable BPB improvement (>0.001) over the Phase 3 model when ablated individually
  2. Each differentiator technique (SmearGate, U-Net skips, OrthoInit) has a measured marginal BPB contribution and artifact size impact
  3. The best combination of differentiators achieves BPB <= 1.145 while remaining within artifact size budget
**Plans:** 2 plans

Plans:
- [ ] 04-01-PLAN.md -- Verify differentiators (SmearGate, U-Net, OrthoInit), strip code comments, increase pruning to 5%
- [ ] 04-02-PLAN.md -- Submit SLURM training job, validate artifact under 16MB and BPB <= 1.145

### Phase 5: Moonshot Exploration
**Goal**: At least one moonshot technique (depth recurrence, TTT, curriculum learning, or sub-5-bit quantization) delivers a measurable BPB improvement beyond the optimized SOTA stack, targeting < 1.13 BPB
**Depends on**: Phase 4
**Requirements**: RECUR-01, RECUR-02, RECUR-03, RECUR-04, TTT-01, TTT-02, TTT-03, TTT-04, CURR-01, CURR-02, CURR-03, LOWBIT-01, LOWBIT-02, LOWBIT-03
**Success Criteria** (what must be TRUE):
  1. Depth-recurrent model (5-6 unique layers looped 2-3x) trains stably and achieves lower BPB than the non-recurrent model at the same artifact size
  2. LoRA TTT provides measurable BPB improvement (>0.002) on top of the best base model while completing within the eval time budget and complying with competition rules
  3. Each moonshot direction has a clear go/no-go result: either integrated into the best model or documented as not competitive with quantified evidence
  4. The best model configuration (base + any successful moonshots) achieves BPB < 1.135
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD
- [ ] 05-04: TBD

### Phase 6: Submission Hardening
**Goal**: The final model is validated as reproducible, rule-compliant, and competition-ready with a complete submission package tested on RunPod 8xH100
**Depends on**: Phase 5
**Requirements**: SUB-01, SUB-02, SUB-03, SUB-04, SUB-05, EVAL-03
**Success Criteria** (what must be TRUE):
  1. Final model achieves < 1.14 BPB on FineWeb validation, beating current SOTA (1.1428)
  2. Artifact (code + compressed weights) fits within 16,000,000 bytes, verified on the actual submission package
  3. 5-seed evaluation demonstrates statistical significance (p < 0.01, >0.005 nat improvement over SOTA)
  4. Training and evaluation both complete within 10 minutes on RunPod 8xH100 SXM
  5. Complete submission package (train_gpt.py, README.md, submission.json, training logs, requirements.txt) passes all competition validation checks
**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure and Baseline | 3/3 | Complete | 2026-03-23 |
| 2. SOTA Stack Integration | 3/3 | Complete | 2026-03-23 |
| 3. Mixed-Precision Quantization | 2/2 | Complete | 2026-03-23 |
| 4. Differentiator Stack | 0/2 | In Progress | - |
| 5. Moonshot Exploration | 0/4 | Not started | - |
| 6. Submission Hardening | 0/2 | Not started | - |

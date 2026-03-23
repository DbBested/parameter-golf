# Requirements: Parameter Golf — First Place Campaign

**Defined:** 2026-03-22
**Core Value:** Achieve the lowest possible BPB score within the 16MB artifact + 10-minute training constraint

## v1 Requirements

Requirements for the competition campaign. Each maps to roadmap phases.

### Infrastructure

- [ ] **INFRA-01**: Training pipeline reproduces official baseline (1.2244 BPB) on pg_tata H200 cluster
- [ ] **INFRA-02**: Automated artifact size validation reports code bytes + compressed model bytes against 16,000,000 byte cap after every run
- [ ] **INFRA-03**: Multi-seed evaluation protocol runs 3+ seeds and reports mean, std, and p-value for BPB differences
- [ ] **INFRA-04**: H200-to-H100 timing calibration empirically measures slowdown factor for our model workload
- [ ] **INFRA-05**: Experiment tracking records BPB, artifact size, training time, and hyperparameters for every run
- [ ] **INFRA-06**: Ablation framework enables toggling individual techniques on/off to measure marginal BPB contribution
- [ ] **INFRA-07**: Storage management automatically cleans up checkpoints and intermediate artifacts to stay within disk quota

### Architecture

- [ ] **ARCH-01**: Model uses 10-layer transformer with 512 hidden dim, MLP 3x expansion with relu-squared activation
- [ ] **ARCH-02**: Grouped-query attention with 8 query heads and 4 KV heads
- [ ] **ARCH-03**: BigramHash embeddings with 10240 buckets (dim=128, projected to model dim)
- [ ] **ARCH-04**: Tied input/output token embeddings with RMSNorm and RoPE
- [ ] **ARCH-05**: SmearGate gating mechanism integrated after attention for local context enhancement
- [ ] **ARCH-06**: U-Net skip connections linking early layers to late layers
- [ ] **ARCH-07**: Orthogonal initialization for all weight matrices

### Quantization

- [ ] **QUANT-01**: Int6 post-training quantization (PTQ) with per-row scaling for all weight matrices (research found SOTA uses PTQ, not QAT — QAT explored in Phase 3 if needed)
- [ ] **QUANT-02**: Mixed-precision quantization: int5 for MLP weights, int6 for attention weights, FP16 for embeddings
- [ ] **QUANT-03**: zstd level 22 compression of quantized weights with bit-packing
- [ ] **QUANT-04**: 3% magnitude pruning post-training to improve compression ratio
- [ ] **QUANT-05**: Per-layer gradient monitoring during QAT to detect instability early

### Training

- [ ] **TRAIN-01**: Muon optimizer for 2D weight matrices (matrix_lr=0.02, momentum=0.99) with AdamW for embeddings/biases
- [ ] **TRAIN-02**: SWA averaging over 20-24 checkpoints from last 40% of warmdown schedule
- [ ] **TRAIN-03**: Weight decay 0.04 applied through both Muon and AdamW
- [ ] **TRAIN-04**: DDP training across 8 GPUs via torchrun with optimized batch size (~786K tokens/batch)
- [ ] **TRAIN-05**: Training completes within 7 minutes on H200 (targeting 10 minutes on H100 with safety margin)

### Evaluation

- [ ] **EVAL-01**: Sliding window evaluation with stride=64 and context length 2048
- [ ] **EVAL-02**: BPB calculation validated against official evaluation script with exact byte-level accounting
- [ ] **EVAL-03**: Evaluation completes within 10 minutes on 8xH100 (separate from training time budget)

### Moonshot — Depth Recurrence

- [ ] **RECUR-01**: Implement depth recurrence: 5-6 unique transformer layers looped 2-3x for 15-18 effective layers
- [ ] **RECUR-02**: Per-recurrence-step adaptation via small learned scaling factors or LoRA-like adapters
- [ ] **RECUR-03**: Training stability maintained under recurrence (no gradient explosion/vanishing through loops)
- [ ] **RECUR-04**: Depth-recurrent model achieves lower BPB than non-recurrent model at same artifact size

### Moonshot — Test-Time Training

- [ ] **TTT-01**: LoRA TTT adapts model during evaluation using next-token prediction on already-evaluated tokens only
- [ ] **TTT-02**: TTT provides measurable BPB improvement (>0.002) on top of the strongest base model
- [ ] **TTT-03**: TTT evaluation completes within the 10-minute eval time budget
- [ ] **TTT-04**: TTT implementation is auditably rule-compliant (no training on future tokens)

### Moonshot — Curriculum Learning

- [ ] **CURR-01**: Training data ordered by difficulty metric (compression ratio or readability score)
- [ ] **CURR-02**: Curriculum schedule progresses from easy to hard samples during training
- [ ] **CURR-03**: Curriculum learning provides measurable BPB improvement vs random ordering

### Moonshot — Sub-5-bit Quantization

- [ ] **LOWBIT-01**: Int4 quantization implemented for MLP layers with int5/int6 for sensitive layers
- [ ] **LOWBIT-02**: Mixed int4/int5/int6 allocation optimized per-layer based on sensitivity analysis
- [ ] **LOWBIT-03**: Sub-5-bit model achieves lower BPB than int5 model at same or smaller artifact size

### Submission

- [ ] **SUB-01**: Final model achieves < 1.14 BPB on FineWeb validation (beating current SOTA 1.1428)
- [ ] **SUB-02**: Model artifact (code + compressed weights) fits within 16,000,000 bytes
- [ ] **SUB-03**: Training reproducible across 5 seeds with statistical significance (p < 0.01, >0.005 nat improvement over SOTA)
- [ ] **SUB-04**: Complete submission package: train_gpt.py, README.md, submission.json, training logs, requirements.txt
- [ ] **SUB-05**: Submission validated on RunPod 8xH100 SXM within 10-minute training budget

## v2 Requirements

Deferred techniques — explore only if v1 targets are met early.

### Advanced Moonshots

- **ADV-01**: Mixture of Experts with shared parameters (high risk at small scale)
- **ADV-02**: Neural Architecture Search for optimal layer/width/precision allocation
- **ADV-03**: Novel tokenizer design (BPE variants, Fusion Token, SuperBPE)
- **ADV-04**: Knowledge distillation from larger teacher model (rules compliance uncertain)
- **ADV-05**: Aggressive parameter tying across layers (shared attention dictionaries)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Non-record unlimited compute track | Focus exclusively on 10-minute leaderboard for maximum impact |
| Custom CUDA kernels from scratch | Development time exceeds competition duration; use FlashAttention/Triton |
| Mobile/edge deployment | Competition-only optimization problem |
| Binary/ternary quantization (1-2 bit) | ParetoQ shows catastrophic quality degradation below 3 bits at this scale |
| Ensemble of multiple models | Multiple models must share 16MB; each sub-model too small |
| Very deep narrow models (20+ layers, 256 dim) | Insufficient representational capacity per layer |
| Complex data augmentation | 10-minute budget too tight; curriculum ordering is better |
| Paper publication | Focus on winning; document techniques in submission README |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| INFRA-05 | Phase 1 | Pending |
| INFRA-06 | Phase 1 | Pending |
| INFRA-07 | Phase 1 | Pending |
| ARCH-01 | Phase 2 | Pending |
| ARCH-02 | Phase 2 | Pending |
| ARCH-03 | Phase 2 | Pending |
| ARCH-04 | Phase 2 | Pending |
| ARCH-05 | Phase 4 | Pending |
| ARCH-06 | Phase 4 | Pending |
| ARCH-07 | Phase 4 | Pending |
| QUANT-01 | Phase 2 | Pending |
| QUANT-02 | Phase 3 | Pending |
| QUANT-03 | Phase 2 | Pending |
| QUANT-04 | Phase 3 | Pending |
| QUANT-05 | Phase 3 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| TRAIN-04 | Phase 2 | Pending |
| TRAIN-05 | Phase 2 | Pending |
| EVAL-01 | Phase 2 | Pending |
| EVAL-02 | Phase 1 | Pending |
| EVAL-03 | Phase 6 | Pending |
| RECUR-01 | Phase 5 | Pending |
| RECUR-02 | Phase 5 | Pending |
| RECUR-03 | Phase 5 | Pending |
| RECUR-04 | Phase 5 | Pending |
| TTT-01 | Phase 5 | Pending |
| TTT-02 | Phase 5 | Pending |
| TTT-03 | Phase 5 | Pending |
| TTT-04 | Phase 5 | Pending |
| CURR-01 | Phase 5 | Pending |
| CURR-02 | Phase 5 | Pending |
| CURR-03 | Phase 5 | Pending |
| LOWBIT-01 | Phase 5 | Pending |
| LOWBIT-02 | Phase 5 | Pending |
| LOWBIT-03 | Phase 5 | Pending |
| SUB-01 | Phase 6 | Pending |
| SUB-02 | Phase 6 | Pending |
| SUB-03 | Phase 6 | Pending |
| SUB-04 | Phase 6 | Pending |
| SUB-05 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-22 after roadmap creation*

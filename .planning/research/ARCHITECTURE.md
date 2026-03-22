# Architecture Patterns

**Domain:** Parameter-constrained language model (OpenAI Parameter Golf)
**Researched:** 2026-03-22
**Overall Confidence:** HIGH (verified against leaderboard submissions, official repo, and published research)

---

## Recommended Architecture

The system consists of seven major components organized into two phases: a **training pipeline** that produces a compressed model artifact, and an **evaluation pipeline** that loads and scores it. Every architectural decision is governed by a single constraint: the final compressed artifact (code + model weights) must fit in 16,000,000 bytes while maximizing BPB on FineWeb validation.

### System Overview

```
Training Phase (10 min, 8xH100)
================================

  FineWeb Data ──> DistributedTokenLoader ──> Model (FP32/BF16 training)
       |                                          |
       |                                     QAT Pipeline
       |                                     (STE gradients)
       |                                          |
       v                                          v
  Data Sharding ──────────────────────> Muon Optimizer + SWA
  (across 8 GPUs)                              |
                                               v
                                    Quantization (Int5/Int6)
                                               |
                                               v
                                    Compression (zstd-22)
                                               |
                                               v
                                    Artifact (<=16MB)

Evaluation Phase (10 min)
==========================

  Artifact ──> Decompress ──> Dequantize ──> Model (FP16 inference)
                                                  |
                                                  v
                                          Sliding Window Eval
                                          (stride=64, context=2048)
                                                  |
                                                  v
                                          [Optional: TTT LoRA]
                                                  |
                                                  v
                                              val_bpb score
```

---

## Component 1: Model Architecture

### Current SOTA Configuration (1.1428 BPB)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Layers | 10 | Sweet spot: 9 too shallow, 11+ busts budget with int5 |
| Hidden dim | 512 | Matches baseline; wider requires fewer layers |
| Attention heads | 8 | Standard for d=512 (head_dim=64) |
| KV heads | 4 | GQA saves ~25% attention params vs MHA |
| MLP expansion | 3x (1536) | 3x empirically better than 4x under budget |
| MLP activation | relu-squared | Higher quality than GELU at same param count |
| Embeddings | Tied (input=output) + BigramHash(10240, dim=128) | Tied saves V*d params; BigramHash adds local bigram context |
| Skip connections | U-Net style cross-layer | Connects early layers to late layers |
| Normalization | RMSNorm (pre-norm) | Fewer params than LayerNorm, better training stability |
| Positional encoding | RoPE | No learned parameters, extrapolates to longer sequences |

**Confidence:** HIGH -- verified from SOTA submission README at github.com/openai/parameter-golf

### Parameter Budget Breakdown (SOTA 10L Model)

This is the critical constraint analysis. All numbers are **pre-quantization FP16 parameters**, then shown with quantized sizes.

```
Component                  | FP16 Params  | FP16 Bytes  | Quantized | Compressed
---------------------------|-------------|-------------|-----------|----------
Token Embeddings (1024*512)| 524,288     | 1,048,576   | FP16 kept | ~800KB
BigramHash (10240*128+proj)| 1,376,256   | 2,752,512   | FP16 kept | ~2.1MB
Attention (per layer)      |             |             |           |
  Q,K,V projections        | 512*512*3=  |             |           |
  (4 KV heads, 8 Q heads)  | 655,360/L   | 1,310,720/L | Int6      | ~110KB/L
  Output projection        | 262,144/L   | 524,288/L   | Int6      | ~44KB/L
MLP (per layer)            |             |             |           |
  Up projection (512*1536) | 786,432/L   | 1,572,864/L | Int5      | ~530KB/L
  Down projection (1536*512)| 786,432/L  | 1,572,864/L | Int5      | ~530KB/L
RMSNorm (per layer)        | 512/L       | 1,024/L     | FP16 kept | ~1KB/L
Final LN + logits          | ~1,024      | ~2,048      | FP16      | ~2KB
---------------------------|-------------|-------------|-----------|----------
TOTAL (10 layers)          | ~26.7M      | ~53.4MB     |           | ~15.8MB
```

**Key insight:** The budget is extremely tight. Int5 MLP quantization (5 bits/param) is what enables 10 layers instead of 9. The leading submission uses **mixed precision** -- Int5 for MLP (tolerates noise), Int6 for attention (precision-sensitive), FP16 for embeddings (highest sensitivity).

### Recommended Architecture Modifications to Explore

#### Direction 1: Depth Recurrence (HIGH priority)
**What:** Run 6-8 unique layers but loop the middle 4 layers 2-3 times, Huginn-style.
**Why:** With weight sharing, a 6-layer model looped 2x has the compute of 10 layers but the parameters of 6. This frees ~40% of parameter budget for wider layers or more embedding capacity.
**Evidence:** Huginn-3.5B achieves Pythia-12B-level performance with 3.5B params by looping 4 recurrent blocks. RingFormer matches standard transformers at 20% of parameter count. Published at ICLR 2026.
**Risk:** The 10-minute training constraint means fewer unique parameters to train. Looping may not converge as well in limited training steps.
**Confidence:** MEDIUM -- strong evidence at larger scale, unverified at 16MB scale.

#### Direction 2: Multi-Scale / U-Net Skip Connections (MEDIUM priority)
**What:** Dense connections from layer i to layer (L-i), already used in SOTA submission.
**Why:** Allows later layers to directly access early representations, improving gradient flow and feature reuse. The leading submission already uses this.
**Evidence:** Confirmed in SOTA submission README. U-Net-style skips are well-established in vision; adaptation to language transformers is newer.
**Confidence:** HIGH -- already proven in competition.

#### Direction 3: Mixture of Experts with Shared Experts (LOW priority)
**What:** Replace dense MLP with 2-4 experts, top-1 routing, shared base expert.
**Why:** MoE increases effective capacity without proportional parameter increase. Could enable wider experts within same budget.
**Risk:** At 16MB scale, the router overhead and load-balancing complexity may not pay off. All experts must be stored (even if only 1 is active per token). Expert collapse risk is high at small scale.
**Evidence:** Research shows MoE is less effective for models under 1B params. Dense models perform better at small scale.
**Confidence:** LOW -- theory suggests benefits but no evidence at this scale.

#### Direction 4: Linear Attention Variants (LOW priority)
**What:** Replace softmax attention with linear attention (RetNet, RWKV-style).
**Why:** O(n) instead of O(n^2) enables longer context at same compute cost.
**Risk:** Quality typically drops vs softmax attention. FlashAttention-3 already makes softmax attention fast enough on H100. The sequence length (2048) is short enough that attention is not the bottleneck.
**Evidence:** Mamba matches transformers at 2x scale for some benchmarks, but at constrained scale, softmax attention with FlashAttention is likely better.
**Confidence:** LOW -- wrong tradeoff for this constraint regime.

---

## Component 2: Quantization Pipeline

### Architecture

```
Training Loop (FP32/BF16 master weights)
    |
    v
Forward Pass: Quantize-on-the-fly via STE
    |
    ├── MLP weights: FP32 -> round to Int5 [-16,15] -> STE gradient
    ├── Attn weights: FP32 -> round to Int6 [-32,31] -> STE gradient
    └── Embeddings: Keep FP16 (too sensitive to quantize)
    |
    v
Backward Pass: STE passes gradients through rounding
    |
    v
Optimizer Step: Update FP32 master weights
    |
    v
[After training] Final quantization + per-row scale factors
    |
    v
Pack bits + zstd-22 compression
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| QAT vs PTQ | QAT (quantization-aware training) | Model learns to compensate for quantization noise during training |
| STE variant | Straight-through estimator | Standard, well-proven for gradient flow through rounding |
| MLP precision | Int5 (5 bits) | MLPs are ~66% of params; Int5 vs Int6 saves ~1.8MB, enabling extra layer |
| Attention precision | Int6 (6 bits) | Attention is precision-sensitive; Int6 preserves quality |
| Embedding precision | FP16 (16 bits) | Embeddings are the most sensitive; quantization degrades BPB significantly |
| Scale granularity | Per-row | Good balance of compression and accuracy |
| Compression | zstd level 22 | Best ratio for quantized weights (1.5-1.9x on int5/int6) |

### Compression Ratios (from SOTA submission)

| Component | Bit width | zstd-22 ratio | Notes |
|-----------|-----------|---------------|-------|
| MLP weights | Int5 | 1.88x | Highest compression; values cluster around 0 |
| Attention weights | Int6 | 1.51x | Moderate compression |
| Embeddings | FP16 | ~1.3x | Least compressible |
| Scale factors | FP16 | ~1.5x | Small but necessary overhead |

**Confidence:** HIGH -- all numbers verified from SOTA submission.

---

## Component 3: Tokenizer and Embedding Design

### Architecture

```
Input bytes ──> BPE Tokenizer (vocab=1024) ──> Token IDs
                                                   |
                                                   v
                                          Standard Embedding (1024 x 512)
                                                   |
                                                   v
                     BigramHash: hash(token[i], token[i-1]) % 10240
                                                   |
                                                   v
                                     BigramHash Embedding (10240 x 128)
                                                   |
                                                   v
                                       Linear projection (128 -> 512)
                                                   |
                                                   v
                                     Add to token embedding
                                                   |
                                                   v
                                    Combined representation (dim=512)
```

### Design Rationale

**Vocabulary size = 1024 (tiny):**
- Standard LLMs use 32K-256K vocab. Parameter Golf uses 1024.
- At 1024 vocab, each token covers ~1.2 bytes on average (very small subwords).
- Embedding table: 1024 * 512 = 524K params = ~1MB in FP16. Manageable.
- Tradeoff: Sequences become longer (more tokens per byte), but attention cost at 2048 seq len is not the bottleneck.

**BigramHash(10240):**
- Hashes consecutive token pairs into 10240 buckets.
- Adds local bigram context that the small vocab cannot capture alone.
- 10240 * 128 = 1.31M params. This is ~5% of total parameter budget but provides meaningful bigram statistics.
- Projected from 128 -> 512 via learned linear (65K more params).
- The SOTA submission showed 10240 buckets improve ~0.001 BPB over 4096 buckets.

**Tied embeddings:**
- Input embedding = output embedding (transposed). Saves 524K params.
- Universal practice at this scale. No reason to untie.

### Vocab Size Tradeoffs

| Vocab Size | Embed Params | Avg Bytes/Token | Seq Length for 2048B | Notes |
|------------|-------------|-----------------|---------------------|-------|
| 512 | 262K | ~0.9 | ~2275 tokens | Too granular |
| 1024 | 524K | ~1.2 | ~1707 tokens | Current SOTA |
| 2048 | 1.05M | ~1.5 | ~1365 tokens | Shorter sequences |
| 4096 | 2.1M | ~2.0 | ~1024 tokens | Embedding budget too large |

**Recommendation:** Stay with 1024 vocab. The parameter budget for larger vocabs is not justified by the BPB improvement from shorter sequences.

**Confidence:** HIGH for current approach, MEDIUM for alternatives.

---

## Component 4: Training Pipeline

### Architecture

```
8x H100 GPUs (DDP)
    |
    v
DistributedTokenLoader
    ├── Shards FineWeb across GPUs
    ├── Sequence length: 2048 tokens
    └── Batch: ~786K tokens total (~96 seqs/GPU)
    |
    v
Model Forward (with QAT)
    |
    v
Cross-Entropy Loss (token-level)
    |
    v
Backward Pass (STE for quantized layers)
    |
    v
Optimizer Step
    ├── Muon (matrix_lr=0.02, momentum=0.99) for 2D params
    ├── AdamW for embeddings and 1D params
    └── Weight decay: 0.04 everywhere
    |
    v
LR Schedule
    ├── 20-step warmup
    └── 3000-step warmdown (cosine decay)
    |
    v
SWA Collection (last 40% of warmdown)
    ├── Save checkpoint every 50 steps
    ├── Collect 24 checkpoints
    └── Average weights
    |
    v
Final Model
```

### Key Training Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Optimizer | Muon + AdamW hybrid | Muon achieves 2x compute efficiency over AdamW for 2D params; AdamW better for embeddings |
| Weight decay | 0.04 | Critical for Muon stability at scale; prevents weight growth |
| SWA | 40% warmdown, 24 checkpoints | Improves generalization ~0.002-0.005 BPB |
| Gradient clipping | 0.3 | Prevents instability from STE gradients |
| Magnitude pruning | 3% | Removes smallest weights, slight compression benefit |
| Distributed strategy | DDP (not FSDP) | Model fits in single GPU memory; DDP simpler and lower overhead |
| Initialization | Orthogonal + muP-scaled outputs | Better convergence than default init |

### Timing Budget (10 min = 600s on 8xH100)

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Data loading + init | ~10s | Negligible |
| Training steps | ~540s | ~3000 steps at ~180ms/step |
| SWA averaging | ~5s | CPU operation on saved checkpoints |
| Quantization | ~5s | Apply int5/int6 quantization |
| Compression | ~30s | zstd-22 is slow but not prohibitive |
| Buffer | ~10s | Safety margin |

**Confidence:** HIGH -- timing from SOTA submission with 3-seed verification.

---

## Component 5: Evaluation Pipeline

### Architecture

```
Compressed Artifact ──> Decompress (zstd) ──> Dequantize ──> FP16 Model
                                                                  |
                                                                  v
                                                    Load FineWeb validation
                                                    (first 50K documents)
                                                                  |
                                                                  v
                                                    Sliding Window Eval
                                                    ┌─────────────────────────┐
                                                    │ Window size: 2048 tokens │
                                                    │ Stride: 64 tokens       │
                                                    │                         │
                                                    │ For each window:        │
                                                    │   1. Forward pass       │
                                                    │   2. Score last 64 toks │
                                                    │   3. Slide forward 64   │
                                                    └─────────────────────────┘
                                                                  |
                                                                  v
                                                    val_bpb = total_loss / ln(2) / total_bytes
```

### Sliding Window Details

- **Window size:** 2048 tokens (matches training sequence length).
- **Stride:** 64 tokens. Only the last 64 tokens in each window are scored; the preceding 1984 tokens provide context.
- **Impact:** Stride=64 gives near-optimal BPB. Stride=1 would be perfect but 32x slower. Stride=256 loses ~0.001 BPB.
- **BPB formula:** `val_bpb = (sum of cross-entropy losses) / ln(2) / (total bytes in validation set)`

### Test-Time Training (TTT) -- Optional Extension

```
During evaluation:
    For each sliding window:
        1. Score tokens normally
        2. After scoring, use those tokens (now "already evaluated") for LoRA adaptation
        3. Update LoRA weights via gradient step
        4. Use adapted model for next window

    Constraint: Can only train on tokens ALREADY scored
```

**Current status:** TTT via LoRA achieved 1.1928 BPB in one submission -- not yet competitive with pure model quality approaches, but promising as an orthogonal improvement vector.

**Confidence:** HIGH for sliding window eval, LOW for TTT competitiveness.

---

## Component 6: Compression Pipeline

### Architecture

```
Trained FP32 Model
    |
    v
Quantize to mixed precision
    ├── MLP weights ──> Int5 (5 bits/param, per-row scale)
    ├── Attention weights ──> Int6 (6 bits/param, per-row scale)
    ├── Embedding weights ──> FP16 (16 bits/param)
    └── LayerNorm/RMSNorm ──> FP16 (tiny, not worth quantizing)
    |
    v
Pack quantized values
    ├── Int5: pack 8 values into 5 bytes (bit packing)
    ├── Int6: pack 4 values into 3 bytes
    └── FP16: direct bytes
    |
    v
zstd compression (level 22)
    |
    v
Artifact = compressed_weights + code_bytes
    |
    v
Verify: artifact_size <= 16,000,000 bytes
```

### Artifact Size Budget

| Component | Uncompressed | Compressed (zstd-22) | % of Budget |
|-----------|-------------|---------------------|-------------|
| MLP weights (10L, Int5) | ~9.8MB | ~5.2MB | 33% |
| Attention weights (10L, Int6) | ~3.7MB | ~2.5MB | 16% |
| Embeddings (FP16) | ~3.8MB | ~2.9MB | 18% |
| BigramHash (FP16) | ~2.8MB | ~2.1MB | 13% |
| Scale factors (FP16) | ~0.3MB | ~0.2MB | 1% |
| LayerNorm params (FP16) | ~0.01MB | ~0.01MB | <1% |
| **Total model** | **~20.4MB** | **~12.9MB** | **81%** |
| Code (train_gpt.py etc.) | ~0.1MB | ~0.1MB | 1% |
| **Remaining budget** | | **~2.9MB** | **18%** |

The ~3MB remaining budget is the "innovation headroom" -- it can be spent on:
- More BigramHash buckets
- Wider hidden dimension
- An additional layer
- Separate embedding parameters (untied output head)
- Extra expert weights (MoE)

**Confidence:** MEDIUM -- estimates based on SOTA ratios, exact numbers vary with weight distributions.

---

## Component 7: Experiment Infrastructure

### Architecture

```
Experiment Runner
    |
    ├── Config system (env vars / YAML)
    │   ├── Architecture params (layers, dim, heads, etc.)
    │   ├── Training params (lr, wd, swa, etc.)
    │   ├── Quantization params (bit widths, which layers)
    │   └── Eval params (stride, window size)
    |
    ├── Ablation framework
    │   ├── Baseline config (current SOTA)
    │   ├── Single-variable sweeps
    │   ├── Multi-seed runs (3+ seeds: 42, 1337, 2024)
    │   └── Statistical significance test (p < 0.01, delta > 0.005 nats)
    |
    ├── Experiment tracking
    │   ├── val_bpb per step
    │   ├── Artifact size per config
    │   ├── Training time per config
    │   └── Quantization noise metrics
    |
    └── Compute allocation
        ├── H200 nodes (primary): full training runs
        ├── L40S nodes (auxiliary): ablation sweeps
        └── RunPod H100 (validation): timing verification
```

### Suggested Ablation Protocol

```
For each proposed change:
    1. Train with change ON vs OFF (3 seeds each)
    2. Measure: val_bpb, artifact_size, training_time
    3. Compute: delta_bpb, p-value (paired t-test)
    4. Accept if: delta_bpb > 0.005, p < 0.01
    5. Log result to tracking system
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Uniform Quantization
**What:** Quantizing all layers to the same bit width.
**Why bad:** Attention layers are precision-sensitive. Uniform Int5 degrades attention quality more than it saves in size. The SOTA uses Int5 for MLP only.
**Instead:** Mixed-precision QAT with per-component bit widths.

### Anti-Pattern 2: Large Vocabulary
**What:** Using vocab > 2048 to reduce sequence length.
**Why bad:** Embedding table grows linearly with vocab. At 8192 vocab: 8192*512 = 4.2M params in FP16 = ~8.4MB, consuming over half the budget on embeddings alone.
**Instead:** Keep vocab=1024, use BigramHash for local context.

### Anti-Pattern 3: Ignoring SWA
**What:** Using final checkpoint instead of weight averaging.
**Why bad:** SWA consistently provides 0.002-0.005 BPB improvement at zero parameter cost.
**Instead:** Always use SWA over last 40% of warmdown.

### Anti-Pattern 4: Over-Engineering MoE at This Scale
**What:** Adding mixture-of-experts routing to a 16MB model.
**Why bad:** Router parameters, load-balancing losses, and the requirement to store ALL experts (not just active ones) make MoE inefficient below ~1B params. Expert collapse risk is high.
**Instead:** Dense MLP with int5 quantization. If exploring capacity expansion, try depth recurrence first.

### Anti-Pattern 5: Neglecting Compression-Awareness
**What:** Designing the model without considering zstd compressibility.
**Why bad:** Some weight distributions compress much better than others. A model that is 15.5MB uncompressed but compresses to 13MB is better than 14MB uncompressed that compresses to 14MB.
**Instead:** Monitor compressed artifact size during development, not just raw parameter count.

### Anti-Pattern 6: Post-Training Quantization (PTQ)
**What:** Training in FP32, then quantizing after training is done.
**Why bad:** Model never learns to compensate for quantization noise. At Int5, PTQ loses 0.01-0.02 BPB vs QAT.
**Instead:** Always use QAT with STE from the start.

---

## Scalability Considerations

This domain does not have traditional scalability concerns (users, requests). Instead, "scalability" means: how does the architecture respond to changes in the constraint budget?

| Budget Change | Impact | Approach |
|--------------|--------|----------|
| 16MB -> 8MB | Need Int4 or fewer layers | Depth recurrence becomes critical; 6 unique layers looped 2x |
| 16MB -> 32MB | Can add layers or widen | 14-16 layers at 512 dim, or 10 layers at 640 dim |
| 10 min -> 5 min | Fewer training steps | Wider model (fewer layers) converges faster |
| 10 min -> 20 min | More training steps | Deeper model (more layers) benefits from longer training |
| BPE 1024 -> 2048 | Shorter sequences, larger embedding | Only worthwhile if embedding budget allows |

---

## Suggested Build Order (Dependencies)

The following build order reflects component dependencies. Each component depends on the ones above it.

### Phase 1: Baseline Reproduction
**Components:** Model architecture (baseline 9L), Training pipeline (basic DDP), Evaluation pipeline (sliding window)
**Goal:** Reproduce baseline 1.2244 BPB
**Dependencies:** None
**Why first:** Validates infrastructure, establishes measurement baseline

### Phase 2: SOTA Reproduction
**Components:** Quantization pipeline (QAT + Int5/Int6), Compression pipeline (zstd-22), BigramHash embeddings, Muon optimizer
**Goal:** Reproduce SOTA 1.1428 BPB
**Dependencies:** Phase 1 (baseline model and eval pipeline)
**Why second:** Establishes the competitive baseline to beat; validates all proven techniques

### Phase 3: Ablation Framework
**Components:** Experiment infrastructure, Ablation protocol, Multi-seed verification
**Goal:** Measure marginal contribution of each SOTA technique
**Dependencies:** Phase 2 (need working SOTA to ablate)
**Why third:** Informs which components to optimize vs which are saturated

### Phase 4: Architectural Innovation
**Components:** Depth recurrence, U-Net skip connections (if not already in SOTA), novel embedding schemes
**Goal:** Beat SOTA by >= 0.005 nats
**Dependencies:** Phase 3 (ablation results tell us where to invest)
**Why fourth:** Innovation is most productive when you understand the current technique landscape

### Phase 5: Submission Optimization
**Components:** Timing calibration (H200 -> H100), Statistical significance testing, Code cleanup
**Goal:** Verified, reproducible submission
**Dependencies:** Phase 4 (need the best model to submit)
**Why last:** Optimization and polish after architecture is locked

---

## Data Flow Summary

### Training Data Flow

```
FineWeb (raw text)
    │
    ▼
BPE Tokenizer (vocab=1024)
    │
    ▼
Token sequences (length=2048)
    │
    ▼
DistributedTokenLoader (shard across 8 GPUs)
    │
    ▼
Model Forward Pass
    ├── Token embeddings (FP16, from tied embedding table)
    ├── BigramHash embeddings (FP16, from hash table)
    ├── Combined embedding (dim=512)
    ├── 10x Transformer blocks:
    │       ├── Pre-norm (RMSNorm)
    │       ├── GQA Attention (8Q/4KV heads, Int6 QAT)
    │       │       └── FlashAttention-3 kernel
    │       ├── Pre-norm (RMSNorm)
    │       ├── MLP (3x expansion, relu^2, Int5 QAT)
    │       └── U-Net skip connection (if applicable)
    ├── Final RMSNorm
    └── Output logits (tied embedding transposed)
    │
    ▼
Cross-entropy loss (per token)
    │
    ▼
Backward pass (STE for quantized operations)
    │
    ▼
Muon/AdamW optimizer step
    │
    ▼
[Repeat for ~3000 steps]
    │
    ▼
SWA averaging (24 checkpoints from warmdown)
    │
    ▼
Quantize + compress -> artifact
```

### Evaluation Data Flow

```
FineWeb validation (50K documents)
    │
    ▼
BPE Tokenizer (same as training)
    │
    ▼
Sliding window iterator (size=2048, stride=64)
    │
    ▼
Model forward pass (FP16, no gradient)
    │
    ▼
Cross-entropy loss on last 64 tokens per window
    │
    ▼
Accumulate: total_loss, total_bytes
    │
    ▼
val_bpb = total_loss / ln(2) / total_bytes
```

---

## Sources

### HIGH Confidence (official / verified)
- [OpenAI Parameter Golf repository](https://github.com/openai/parameter-golf) -- competition rules, baseline, evaluation methodology
- [SOTA submission README (thwu1, 1.1428 BPB)](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) -- architecture details, ablation data, quantization ratios
- [GQA paper (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245) -- grouped query attention
- [FlashAttention-3 paper](https://tridao.me/publications/flash3/flash3.pdf) -- H100-optimized attention
- [Muon optimizer](https://github.com/KellerJordan/Muon) -- optimizer details, weight decay findings
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) -- Muon at scale with weight decay

### MEDIUM Confidence (cross-referenced)
- [Depth-Recurrent Transformer (Huginn-3.5B)](https://huggingface.co/tomg-group-umd/huginn-0125) -- looping architecture at larger scale
- [From Growing to Looping (ICLR 2026)](https://openreview.net/pdf/183334103d5fda67d365e08fec721accd09b8ec8.pdf) -- unified view of progressive and looped architectures
- [Allocation of Parameters in Transformers](https://arxiv.org/abs/2510.03784) -- parameter distribution research
- [Parameter-Efficient Transformer Embedding](https://arxiv.org/html/2505.02266v1) -- Fourier-based embedding alternatives
- [ZipNN: Lossless Compression for AI Models](https://arxiv.org/html/2411.05239v2) -- neural network weight compression
- [Transformer parameter counting](https://michaelwornow.net/2024/01/18/counting-params-in-transformer) -- parameter budget formulas
- [DeepWiki: Parameter Golf](https://deepwiki.com/openai/parameter-golf) -- community analysis

### LOW Confidence (single source / unverified at competition scale)
- MoE at small scale effectiveness -- extrapolated from large-scale findings, no competition-specific evidence
- Linear attention competitiveness at 2048 seq len -- theoretical argument only
- TTT LoRA competitiveness -- single submission at 1.1928 BPB, far from SOTA
- Progressive growing during training -- no evidence in parameter golf context

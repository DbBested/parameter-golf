# Feature Research: Parameter Golf Competition Techniques

**Domain:** Parameter-constrained language model optimization (OpenAI Parameter Golf)
**Researched:** 2026-03-22
**Confidence:** HIGH (leaderboard submissions analyzed, techniques verified against papers and official docs)

## Context

This research maps the technique landscape for OpenAI's Parameter Golf competition: train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by bits-per-byte (BPB) on FineWeb validation. The "features" here are optimization techniques, and the "users" are the BPB metric -- every technique must justify its parameter budget in BPB improvement.

Current SOTA: **1.1428 BPB** (thwu1). Baseline: **1.2244 BPB**. Gap exploited so far: ~0.082 BPB. Target: < 1.14 BPB.

---

## Feature Landscape

### Table Stakes (Must Have or Cannot Compete)

Techniques every competitive submission uses. Missing any of these means starting 0.01-0.05 BPB behind the leaders.

| Technique | Why Required | Complexity | Est. BPB Impact | Notes |
|-----------|-------------|------------|-----------------|-------|
| **Int6 QAT with STE** | All top-5 use int6 minimum; enables ~2.5x more parameters in 16MB budget vs FP16 | MEDIUM | -0.04 to -0.06 vs FP16 baseline | Straight-Through Estimator (STE) allows gradients to flow through quantization during training. Every submission on leaderboard uses at least int6. |
| **BigramHash Embeddings (>=4096)** | Replaced learned vocab embeddings; hashes consecutive token pairs into buckets for local context. All top-4 use this | MEDIUM | -0.01 to -0.02 vs tied embeddings alone | Hash consecutive token pairs into N-bucket embedding table (dim=128), project via learned linear to model dim. Reduces collision vs small vocab. |
| **Muon Optimizer** | ~2x compute efficiency vs AdamW; all top submissions use Muon for weight matrices | LOW | -0.01 to -0.02 vs AdamW | Use Muon for matrix params (matrix_lr=0.02, momentum=0.99), AdamW for embeddings/scalars. Weight decay 0.04 is standard. |
| **Stochastic Weight Averaging (SWA)** | Free BPB improvement by averaging checkpoints from late training. All top-4 use SWA | LOW | -0.003 to -0.006 | Average 20-24 checkpoints from last 40-50% of warmdown. Nearly zero cost -- just checkpoint collection and averaging. |
| **MLP 3x Expansion with relu^2** | Standard MLP design across all top submissions; relu^2 empirically outperforms GELU at this scale | LOW | Foundational (not optional) | relu^2 provides sparsity (many activations zero) which compresses better. 3x expansion (1536 hidden for 512 dim) is the sweet spot for parameter budget. |
| **10-11 Transformer Layers** | Sweet spot for depth vs width at 16MB; baseline was 9 layers, leaders use 10-11 | LOW | -0.005 to -0.01 vs 9 layers | Adding layer 10 requires freeing ~1.8MB via int5 MLP quantization. 11 layers used by some but requires aggressive compression elsewhere. |
| **Grouped-Query Attention (GQA)** | 8 query heads, 4 KV heads. Reduces attention param count while preserving quality | LOW | Enables more layers; indirect BPB benefit | Standard across all submissions. Fewer KV heads = fewer attention params = room for more layers or wider MLP. |
| **zstd Compression (level 22)** | Compresses quantized weights for artifact storage; all submissions use high-level zstd | LOW | Enables fitting more params in 16MB | Int5 weights achieve ~1.88x compression ratio; int6 achieves ~1.51x. Critical for meeting the 16MB hard cap. |
| **Sliding Window Evaluation** | Stride=64 gives model more context per prediction; standard eval optimization | LOW | -0.002 to -0.005 vs naive eval | Smaller stride = more context = better predictions. Stride 64 is common; diminishing returns below 32. Only affects eval, no training cost. |
| **Weight Decay 0.04** | Consistent across top submissions; prevents overfitting in short training runs | LOW | Embedded in training quality | Applied via both Muon and AdamW. Higher than typical LLM training due to short schedule. |
| **Tied Token Embeddings** | Input/output embedding sharing; baseline technique that saves substantial params | LOW | Foundational | Every submission ties embeddings. Frees params for other components. |

### Differentiators (Competitive Advantage -- What Separates Top 3 from Top 10)

Techniques that distinguish leaders from followers. Each provides incremental BPB improvement but requires careful implementation.

| Technique | Value Proposition | Complexity | Est. BPB Impact | Notes |
|-----------|-------------------|------------|-----------------|-------|
| **Int5 MLP Quantization** | Leader uses int5 [-16,15] for MLP weights, saving ~1.86MB vs uniform int6. This freed space enables 10th layer | HIGH | -0.003 (from extra layer it enables) | Only the #1 submission (1.1428) uses int5. Requires careful mixed-precision: int5 for MLP, int6 for attention, FP16 for embeddings. Higher compression ratio (1.88x vs 1.51x for int6). Risk: quality degradation if applied to attention weights. |
| **BigramHash 10240 Buckets** | 2.5x larger hash table than 4096 baseline; reduces collisions, improves local context capture | LOW | -0.0008 vs 8192 buckets | Leader expanded from 4096 to 10240. Diminishing returns expected beyond ~12K. Sweet spot analysis needed. |
| **SmearGate** | Gating mechanism for local context integration, likely related to NeurIPS 2025 Gated Attention work | MEDIUM | -0.002 to -0.004 (estimated) | Used by submissions #2 and #4. Applies learnable sigmoid gate after attention. Enhances training stability and allows larger learning rates. Exact implementation needs extraction from PRs. |
| **Orthogonal Initialization (OrthoInit)** | Provably faster convergence, gradient stability, diverse feature learning | LOW | -0.001 to -0.002 | Used by submission #2. Ensures W^T W = I at init. Particularly helpful for deep networks (10+ layers). Near-zero implementation cost. |
| **U-Net Skip Connections** | Cross-layer residual shortcuts beyond standard per-layer residuals | MEDIUM | -0.001 to -0.002 (estimated) | Used by leader. Connects early layers to late layers, improving gradient flow and feature reuse. Small param overhead for connection weights. |
| **Magnitude Pruning (3%)** | Zeros out 3% smallest-magnitude weights post-training; improves compression ratio | LOW | Net neutral BPB, but saves ~0.5MB artifact space | Leader uses 3% pruning. Aggressive pruning (>5%) degrades BPB. The saved space can be reinvested in more parameters. |
| **Warmdown Schedule Tuning** | 3000-iteration warmdown with specific checkpoint collection windows | LOW | -0.0001 to -0.001 | Fine-tuning the learning rate schedule tail. Combined with SWA checkpoint collection. |
| **Mixed-Precision Strategy** | Different quantization levels for different component types based on sensitivity | HIGH | -0.002 to -0.005 cumulative | Int5 for MLP (tolerant), int6 for attention (sensitive), FP16 for embeddings and final-layer key projections. Requires ablation to find optimal allocation. |

### Moonshots (High Risk, High Reward -- Could Leapfrog the Field)

Techniques NOT yet on the leaderboard that could provide breakthrough BPB improvements. These are the unexplored frontier.

| Technique | Potential Value | Complexity | Est. BPB Impact | Risk Assessment |
|-----------|----------------|------------|-----------------|-----------------|
| **Depth Recurrence / Universal Transformers** | Reuse a small set of transformer blocks N times, achieving effective depth of N*blocks with only 1 block's parameters. Research shows 25-55% parameter savings at comparable quality | HIGH | -0.01 to -0.03 (if it works) | **HIGH RISK, HIGH REWARD.** Huginn-3.5B shows recurrence works at scale. Key question: does it work at 16MB scale with 10-min training? Could allow 20+ effective layers with 5-6 unique layers. Requires per-layer scaling/adaptation (small LoRA-like adapters per recurrence step). Training stability is the main concern. |
| **Mixture of Experts (MoE) with Shared Parameters** | Route tokens to specialized sub-networks while sharing base parameters. Could increase effective model capacity within fixed param budget | HIGH | -0.01 to -0.02 (if it works) | **HIGH RISK.** MoE typically increases params, but with shared expert bases + small routing, could increase capacity within 16MB. PuzzleMoE's sparse expert merging shows experts share substantial knowledge. Challenge: routing overhead, training instability at small scale, load balancing in 10 minutes. |
| **Test-Time Training (LoRA TTT)** | Adapt model to validation data distribution during eval using next-token prediction on already-evaluated tokens. Current best: 1.1928 BPB (not competitive yet) | HIGH | -0.005 to -0.015 (if optimized) | **MEDIUM-HIGH RISK.** Rules explicitly allow TTT on already-evaluated tokens. Current gap to SOTA is ~0.05 BPB, but TTT is applied on top of a weaker base model. If applied to a 1.14-level base model, could push to 1.13+. Key: LoRA rank, learning rate, which layers to adapt, eval-time budget (10 min limit for eval too). End-to-End TTT (meta-learning approach) is most promising variant. |
| **Sub-5-bit Quantization (int4, int3)** | int4 saves another ~20% vs int5, enabling 1-2 more layers or wider model. ParetoQ shows ternary 600M outperforms prior SOTA ternary 3B | VERY HIGH | -0.005 to -0.01 (from extra capacity) | **HIGH RISK.** ParetoQ (NeurIPS 2025) shows ternary and 2-bit can maintain quality with QAT, but there is a "learning transition" between 2 and 3 bits where representations change drastically. Int4 is more feasible than ternary for this competition. Would need careful mixed-precision: int4 for most-tolerant layers, int5/int6 for sensitive ones. |
| **Aggressive Parameter Tying Beyond Embeddings** | Share attention weights across layers (e.g., Sandwich-style, Subformer). Can reduce params by 66.7% for attention | HIGH | -0.005 to -0.015 (from freed capacity) | **MEDIUM RISK.** Recent 2025 work shows decomposing Q/K/V/O into shared dictionary atoms reduces attention params by 66.7% with on-par performance. Combining with per-layer scaling factors recovers most quality. Sandwich-style sharing (unique top/bottom, shared middle) works well for generative models. Synergizes with depth recurrence. |
| **Curriculum Learning / Data Ordering** | Reorder training data easy-to-hard; 18-45% fewer steps to reach baseline performance. MIT-IBM 2025: up to 15% improvement from data ordering alone | MEDIUM | -0.005 to -0.01 | **MEDIUM RISK.** Sorting by compression ratio or readability is simple to implement. Benefits are more pronounced for smaller models. 10-minute training budget means every step counts -- faster convergence = more effective training. Challenge: computing difficulty metrics for FineWeb adds preprocessing cost. |
| **Novel Tokenizer Design** | Optimize tokenizer for compression ratio rather than linguistic quality. Fusion Token achieves compression of 1M-vocab BPE with only 1024 extra tokens | MEDIUM | -0.003 to -0.008 | **MEDIUM RISK.** Competition is tokenizer-agnostic (BPB metric), so better tokenization directly improves BPB. SuperBPE enables multi-word tokens. Unigram achieves lowest fertility scores. Challenge: tokenizer must be included in 16MB artifact. Larger vocab = more embedding params but fewer tokens to predict. |
| **Knowledge Distillation During Training** | Use a pre-trained teacher model to guide small model training. MiniPLM (ICLR 2025) enables offline teacher inference, no extra training cost | MEDIUM | -0.005 to -0.01 | **MEDIUM-HIGH RISK.** Rules don't explicitly prohibit distillation during training (only prohibit training on validation data). Teacher logits could be precomputed offline. But: 10-minute training budget is tight, and distillation loss adds compute per step. May conflict with competition spirit. Need to verify rules carefully. |
| **Neural Architecture Search (NAS)** | Automatically find optimal layer config, width, expansion ratios for 16MB budget | HIGH | -0.003 to -0.008 | **LOW-MEDIUM RISK for the search itself, HIGH COST.** Use zero-cost proxies (RZ-NAS) to search without training. LLaMA-NAS showed 1.5x model reduction with negligible accuracy loss. But: search space is already narrow (10-11 layers, 512 dim), so gains may be small. Better used to optimize mixed-precision allocation and MLP ratios. |

### Anti-Features (Things to Deliberately NOT Do)

Approaches that seem promising but waste time, violate rules, or have negative expected value.

| Anti-Feature | Why Tempting | Why Problematic | What to Do Instead |
|-------------|-------------|-----------------|-------------------|
| **Training on validation data** | Would directly optimize the eval metric | Explicitly prohibited by rules. Disqualification risk. | Use TTT on already-evaluated tokens only (allowed). |
| **Full FP16 model (no quantization)** | Simpler implementation, no quantization noise | Cannot fit competitive model in 16MB. FP16 limits to ~8M params; int5 allows ~25M+ | Always quantize. Int5/int6 QAT is table stakes. |
| **Extremely large vocabulary (>2K)** | More tokens = fewer predictions needed | Embedding table eats parameter budget. BigramHash achieves similar benefit without vocab size cost | Use BigramHash with 10K+ buckets instead of large vocab. |
| **Binary/ternary quantization (1-2 bit)** | Maximum compression; could fit enormous model | ParetoQ shows "learning transition" below 3 bits where representations change drastically. Quality degradation likely catastrophic at this scale with 10-min training | Stay at int4 minimum for moonshots; int5 for reliable gains. |
| **Custom CUDA kernels from scratch** | Could optimize training speed | Development time >> competition duration. Existing optimized libraries (FlashAttention, Triton) cover needed ops | Use existing kernels. Focus on algorithmic innovation. |
| **Ensemble of multiple models** | Ensembles typically improve predictions | Multiple models must fit in 16MB total. Each sub-model too small to be effective | Single model with SWA (implicit ensemble). |
| **Very deep narrow models (20+ layers, 256 dim)** | More layers = more transformations | Narrow models lack representational capacity per layer. Communication overhead on 8xH100 increases with depth | Prefer 10-12 layers at 512 dim. Depth via recurrence if needed. |
| **Pruning > 10% of weights** | Saves more artifact space | Aggressive pruning degrades BPB significantly. 3% is the tested ceiling | Cap pruning at 3-5%. Use quantization for compression instead. |
| **Complex data augmentation** | More diverse training signal | 10-minute budget is too tight. Augmentation compute competes with training steps | Use curriculum ordering instead -- same benefit, lower cost. |
| **Downloading external data during eval** | Could leverage additional knowledge | Explicitly prohibited by rules. No external downloads during evaluation | All knowledge must be in the 16MB artifact. |

---

## Feature Dependencies

```
[Int5/Int6 QAT]
    |
    +--enables--> [More Layers (10-11)]
    |                 |
    |                 +--enhances--> [U-Net Skip Connections]
    |                 |
    |                 +--enables--> [Depth Recurrence] (moonshot)
    |
    +--requires--> [zstd-22 Compression]
    +--requires--> [STE Gradient Estimation]

[BigramHash Embeddings]
    |
    +--replaces--> [Large Vocabulary]
    +--enhances--> [Tied Embeddings] (orthogonal, can coexist)
    +--independent of--> [Quantization] (BigramHash in FP16)

[Muon Optimizer]
    |
    +--enhances--> [SWA] (Muon's trajectory + SWA averaging)
    +--requires--> [AdamW for embeddings] (Muon only for matrix params)
    +--enhances--> [Weight Decay] (Muon WD=0.04 standard)

[SWA]
    |
    +--requires--> [Warmdown Schedule] (checkpoints collected during warmdown)
    +--enhances--> [Any base model] (always additive)

[Depth Recurrence] --conflicts with-- [More Unique Layers]
    (recurrence saves params by reusing layers; more unique layers
     spends params on distinct layers. Choose one strategy.)

[MoE Routing] --conflicts with-- [Depth Recurrence]
    (both increase effective capacity differently; combining adds
     excessive complexity for 10-min training)

[Test-Time Training] --independent of-- [All training techniques]
    (TTT is eval-time only; can layer on top of any base model)

[Curriculum Learning] --enhances--> [Muon Optimizer]
    (faster convergence from data ordering + efficient optimizer)

[Knowledge Distillation] --conflicts with-- [10-min Training Budget]
    (distillation adds compute per step, reducing total steps)

[Sub-4-bit Quantization] --requires--> [ParetoQ-style QAT]
    (standard STE not sufficient below int4; need specialized training)
```

### Dependency Notes

- **QAT enables depth:** Going from FP16 to int5 frees ~60% of parameter budget, enabling 10-11 layers vs 5-6. This is the single most impactful dependency.
- **Depth recurrence conflicts with unique layers:** If you reuse 5 layer blocks 4 times (20 effective layers), you cannot also have 11 unique layers. Must choose one strategy.
- **TTT is orthogonal to everything:** Test-time training happens during evaluation and does not interfere with any training technique. It is the safest moonshot because it layers on top of whatever base model you build.
- **MoE and recurrence are competing moonshots:** Both aim to increase effective capacity per parameter. Combining them creates excessive complexity for a 10-minute training window.
- **Curriculum learning is low-risk enhancement:** Can be added to any training pipeline. The main cost is preprocessing FineWeb with difficulty metrics.

---

## Implementation Phases (MVP to Full Campaign)

### Phase 1: Reproduce Baseline + Table Stakes (Target: ~1.15 BPB)

Implement all table stakes techniques to reach competitive baseline.

- [ ] **Int6 QAT with STE** -- Foundation quantization; all submissions use this minimum
- [ ] **BigramHash(10240)** -- Replace learned embeddings with hash-based approach
- [ ] **Muon optimizer + AdamW** -- Muon for weights, AdamW for embeddings
- [ ] **SWA (last 40% warmdown)** -- Free improvement via checkpoint averaging
- [ ] **10L, 512d, MLP 3x relu^2** -- Standard architecture config
- [ ] **GQA (8Q/4KV heads)** -- Grouped-query attention
- [ ] **Tied embeddings** -- Input/output sharing
- [ ] **zstd-22 compression** -- Artifact compression
- [ ] **Sliding window eval (stride=64)** -- Eval optimization
- [ ] **Weight decay 0.04** -- Regularization

### Phase 2: Differentiator Stack (Target: ~1.14 BPB)

Layer on differentiating techniques to match/beat current SOTA.

- [ ] **Int5 MLP quantization** -- Mixed precision: int5 MLP, int6 attention, FP16 embeddings
- [ ] **SmearGate** -- Gated attention mechanism for local context
- [ ] **OrthoInit** -- Orthogonal initialization for all weight matrices
- [ ] **U-Net skip connections** -- Cross-layer residual shortcuts
- [ ] **Magnitude pruning (3%)** -- Post-training weight pruning for compression
- [ ] **Warmdown schedule optimization** -- Fine-tune LR schedule tail

### Phase 3: Moonshot Exploration (Target: < 1.13 BPB)

Explore high-risk, high-reward techniques for leapfrog gains.

- [ ] **Test-time training (LoRA TTT)** -- Safest moonshot; layers on top of best base model
- [ ] **Depth recurrence** -- Reuse transformer blocks for effective depth 20+ with 5-6 unique blocks
- [ ] **Curriculum learning** -- Order FineWeb data easy-to-hard for faster convergence
- [ ] **Aggressive parameter tying** -- Share attention weights across layers with per-layer adapters
- [ ] **Int4 mixed precision** -- Push MLP quantization to int4 for most-tolerant layers

### Future Consideration (If time permits)

- [ ] **Novel tokenizer** -- BPE variant optimized for compression ratio on FineWeb
- [ ] **MoE with shared experts** -- Small-scale mixture of experts with shared base parameters
- [ ] **NAS for mixed-precision allocation** -- Automated search for optimal bit allocation per layer
- [ ] **Knowledge distillation** -- Precompute teacher logits offline, distill during training

---

## Technique Prioritization Matrix

| Technique | BPB Impact | Implementation Cost | Risk | Priority |
|-----------|-----------|---------------------|------|----------|
| Int6 QAT + STE | HIGH (-0.05) | MEDIUM | LOW | **P0** |
| BigramHash(10240) | MEDIUM (-0.015) | MEDIUM | LOW | **P0** |
| Muon + AdamW | MEDIUM (-0.015) | LOW | LOW | **P0** |
| SWA (0.4) | MEDIUM (-0.005) | LOW | LOW | **P0** |
| 10L 512d MLP3x relu^2 | HIGH (foundational) | LOW | LOW | **P0** |
| Sliding window eval | LOW (-0.003) | LOW | LOW | **P0** |
| Int5 MLP quantization | MEDIUM (-0.003+) | HIGH | MEDIUM | **P1** |
| SmearGate | MEDIUM (-0.003) | MEDIUM | MEDIUM | **P1** |
| OrthoInit | LOW (-0.001) | LOW | LOW | **P1** |
| U-Net skip connections | LOW (-0.001) | MEDIUM | LOW | **P1** |
| Magnitude pruning 3% | LOW (artifact space) | LOW | LOW | **P1** |
| Test-time training | HIGH (-0.01+) | HIGH | MEDIUM | **P2** |
| Depth recurrence | VERY HIGH (-0.02+) | VERY HIGH | HIGH | **P2** |
| Curriculum learning | MEDIUM (-0.007) | MEDIUM | MEDIUM | **P2** |
| Parameter tying | HIGH (-0.01+) | HIGH | MEDIUM | **P2** |
| Int4 quantization | MEDIUM (-0.005) | VERY HIGH | HIGH | **P2** |
| Novel tokenizer | MEDIUM (-0.005) | HIGH | MEDIUM | **P3** |
| MoE shared experts | HIGH (-0.015) | VERY HIGH | HIGH | **P3** |
| Knowledge distillation | MEDIUM (-0.007) | HIGH | HIGH | **P3** |
| NAS for bit allocation | LOW (-0.005) | HIGH | MEDIUM | **P3** |

**Priority key:**
- **P0:** Must implement first -- these are table stakes that all competitive submissions use
- **P1:** Implement next -- these differentiate top-3 from top-10 on the leaderboard
- **P2:** Moonshot exploration -- high potential for leapfrog, requires experimentation
- **P3:** If time permits -- lower expected value or higher implementation cost

---

## Competitor Technique Analysis

| Technique | #1 thwu1 (1.1428) | #2 R. Shah (1.1458) | #3 aruniyer (1.1502) | #4 aquarious (1.1556) | #5 yahya010 (1.1586) | Our Approach |
|-----------|-------------------|---------------------|----------------------|-----------------------|----------------------|--------------|
| Quantization | Int5 MLP + Int6 Attn | Int6 | Int6 QAT | Int6 STE QAT | Int6 QAT | Start Int6, graduate to Int5 MLP mixed |
| Layers | 10 | (likely 10-11) | 11 | (likely 10) | 10 | 10L baseline, explore recurrence for effective 20+ |
| MLP Expansion | 3x | 3x | 3x | 3x | (custom) | 3x with relu^2, explore MoE variant |
| SmearGate | Yes | Yes | No | Yes | No | Implement -- used by 3 of top 5 |
| BigramHash | 10240 | Yes | (unknown) | Yes | (unknown) | 10240 minimum, explore 12K-16K |
| OrthoInit | (unknown) | Yes | (unknown) | (unknown) | (unknown) | Implement -- low cost, proven benefit |
| SWA | 0.4 start | Yes | (unknown) | (unknown) | (unknown) | 0.4 start, 24 checkpoints |
| Muon WD | 0.04 | Yes | (unknown) | (unknown) | 0.99 momentum | 0.04 WD, 0.99 momentum |
| Sliding eval | Yes | (unknown) | stride=64 | stride=64 | (unknown) | stride=64 minimum, explore stride=32 |
| TTT | No | No | No | No | No | **Key differentiator** -- nobody has made it work yet |
| Recurrence | No | No | No | No | No | **Key differentiator** -- unexplored territory |
| Curriculum | No | No | No | No | No | **Key differentiator** -- proven in literature |

---

## Key Strategic Insights

### 1. The Field is Converging on a Standard Stack

All top-5 submissions use the same core: int6+ QAT, BigramHash, Muon, SWA, 10-11 layers, MLP 3x relu^2. The marginal gains from optimizing within this stack are shrinking (0.003 BPB between #1 and #2). **Winning requires breaking out of the standard stack.**

### 2. Test-Time Training is the Lowest-Risk Moonshot

TTT layers on top of any base model, does not affect training, and is explicitly allowed by rules. Current TTT submissions use weak base models (1.19 BPB). Applying optimized TTT to a 1.14-level base model could yield 1.13 or better. The key challenges are LoRA rank selection, learning rate tuning, and staying within the 10-minute eval budget.

### 3. Depth Recurrence is the Highest-Reward Moonshot

Literature shows 25-55% parameter savings at comparable quality. In the 16MB constraint, this could mean effective depth of 20 layers with the params of 10. Nobody on the leaderboard has tried this. The risk is training stability with 10-minute budget and small model scale.

### 4. Curriculum Learning is Underexplored Free Performance

MIT-IBM 2025 research shows up to 15% improvement from data ordering, and benefits are *more pronounced for smaller models*. Sorting FineWeb by compression ratio or readability is implementable in hours and could yield 0.005-0.01 BPB improvement.

### 5. The 16MB Budget Creates a Precise Optimization Problem

Every technique competes for the same 16MB. Int5 saves 1.86MB which enables layer 10 which yields -0.003 BPB. This creates a rich optimization landscape where the winning strategy is not "use every technique" but "allocate the 16MB budget optimally across techniques."

---

## Sources

### Competition
- [OpenAI Parameter Golf GitHub](https://github.com/openai/parameter-golf) -- competition rules, baseline, leaderboard
- [OpenAI Parameter Golf Official Announcement](https://openai.com/index/parameter-golf/) -- competition overview and expected techniques
- [Leading Submission README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) -- detailed ablation results from #1 entry
- [Community Leaderboard Monitor](https://github.com/openai/parameter-golf/issues/158) -- tracking tool for submissions

### Quantization
- [EfficientQAT (ACL 2025)](https://arxiv.org/abs/2407.11062) -- efficient QAT for LLMs
- [ParetoQ (NeurIPS 2025)](https://arxiv.org/abs/2502.02631) -- scaling laws for extreme low-bit quantization
- [Mixed-Precision Quantization Survey](https://arxiv.org/html/2510.16805v1) -- techniques and prospects

### Architecture
- [Depth-Recurrent Transformers](https://www.emergentmind.com/topics/depth-recurrent-transformer) -- parameter sharing via layer reuse
- [Mixture-of-Recursions](https://arxiv.org/html/2507.10524v1) -- adaptive recursive depth
- [Relaxed Recursive Transformers](https://ajithp.com/2024/10/29/recursive-transformers/) -- enhanced parameter sharing
- [Dynamic Layer Tying](https://arxiv.org/abs/2401.12819) -- RL-based layer selection and tying
- [Subformer (EMNLP 2021)](https://arxiv.org/abs/2101.00234) -- sandwich-style parameter sharing

### Training
- [Muon Scalability (2025)](https://arxiv.org/abs/2502.16982) -- Muon optimizer for LLM training
- [SWA Original Paper](https://arxiv.org/abs/1803.05407) -- stochastic weight averaging theory
- [Curriculum Learning for LLM Pretraining](https://arxiv.org/abs/2506.11300) -- data ordering benefits
- [MiniPLM (ICLR 2025)](https://arxiv.org/abs/2410.17215) -- knowledge distillation for pre-training

### Evaluation
- [Test-Time Learning for LLMs (ICML 2025)](https://arxiv.org/pdf/2505.20633) -- TTL paradigm
- [End-to-End TTT](https://arxiv.org/abs/2512.23675) -- meta-learning approach to test-time training
- [LoRA-TTT](https://openreview.net/forum?id=P2XhjOJL7Z) -- low-rank test-time training

### Gating and Initialization
- [Gated Attention (NeurIPS 2025 Best Paper)](https://openreview.net/forum?id=1b7whO4SfY) -- sigmoid gating after attention
- [Orthogonal Initialization Benefits](https://arxiv.org/abs/2001.05992) -- provable convergence speedup

---
*Feature research for: Parameter Golf Competition Techniques*
*Researched: 2026-03-22*

# Phase 5: Moonshot Exploration - Research

**Researched:** 2026-03-23
**Domain:** Four independent experiment directions -- depth recurrence, test-time training, curriculum learning, sub-5-bit quantization
**Confidence:** MEDIUM (individual techniques well-documented in literature, but behavior at 16MB/10-minute competition scale is unverified)

## Summary

This research investigates four moonshot techniques that could push BPB below 1.135 when applied on top of the current competitive base model (BPB 1.1421, artifact 15.82MB). Each moonshot is an independent experiment with distinct implementation patterns, risk profiles, and go/no-go criteria.

**Depth recurrence** (Huginn-style layer looping) replaces 10 unique transformer blocks with ~5 unique blocks looped 2-3x, saving significant parameter budget from weight sharing. The shared weights compress nearly perfectly under zstd, freeing artifact space to widen the model or add more unique capacity. The Huginn architecture uses a Prelude/Recurrent/Coda structure with an adapter matrix that re-injects embeddings at each loop iteration, plus 4x RMSNorm "sandwich" per layer for training stability.

**Test-time training (TTT)** adapts the model during evaluation using LoRA on already-evaluated tokens. An existing competition submission demonstrates the pattern: per-document LoRA adaptation with rank-8, lr=0.01, targeting Q/V projections and lm_head. The existing submission achieves only 1.1928 BPB on a weak base model, but crucially, most of its improvement (-0.011 BPB) comes from document-isolated evaluation rather than TTT itself (-0.003 BPB). Applying TTT to our 1.14-level model could be additive, but the marginal TTT-specific gain may be small.

**Curriculum learning** orders training data by difficulty (compression ratio being the recommended metric). Research shows 18-45% fewer steps to reach baseline performance, and up to 3.5% sustained improvement when used as a warmup strategy. Implementation requires precomputing difficulty scores and sorting/bucketing the training shards.

**Sub-5-bit quantization** (int4 for MLP layers) saves ~20% over int5, freeing 1-2MB for additional parameters. The current quantization code uses `quantize_intN_per_row` with clip_range parameter -- int4 means clip_range=7. The risk is quality degradation, which ParetoQ research suggests is manageable above 3 bits with QAT.

**Primary recommendation:** Prioritize TTT and depth recurrence as the highest-impact experiments. TTT is lowest risk (eval-time only, orthogonal to training). Depth recurrence has highest potential reward (parameter savings + compression benefit). Curriculum and sub-5-bit are secondary -- implement if time permits or as quick experiments.

## Standard Stack

No new libraries are needed -- all moonshots are implemented as modifications to the existing `train_gpt.py`. The standard stack is purely implementation patterns within PyTorch.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.6.0+cu124 | All model code, LoRA, quantization | Already in use; no alternatives needed |
| zstandard | (installed) | Compression of quantized weights | Already in use |
| zlib | stdlib | Compression ratio as curriculum difficulty metric | Built into Python |
| numpy | (installed) | Data manipulation for curriculum ordering | Already in use |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sentencepiece | (installed) | Tokenizer for document boundary detection in TTT | Already in use |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| zlib for difficulty metric | gzip module | Equivalent; zlib is slightly faster for per-sequence compression ratio |
| Custom LoRA | PEFT library | PEFT adds dependency; custom LoRA is ~30 lines and already proven in TTT submission |

**Installation:** No new packages needed. All moonshots use existing dependencies.

## Architecture Patterns

### Moonshot 1: Depth Recurrence Architecture

**Structure: Prelude + Recurrent Core + Coda**

Based on Huginn-3.5B (2 prelude, 4 recurrent, 2 coda layers), adapted to our 16MB scale:

```
# At 16MB scale (our model):
# Option A: 2 prelude + 3 recurrent x 2 loops + 2 coda = 10 effective layers, 7 unique
# Option B: 1 prelude + 3 recurrent x 3 loops + 1 coda = 11 effective layers, 5 unique
# Option C: 0 prelude + 5 recurrent x 2 loops + 0 coda = 10 effective layers, 5 unique (simplest)

src/
  train_gpt.py
    class Block          # unchanged
    class GPT            # modified forward() to loop recurrent blocks
      - self.prelude_blocks    # unique first N layers
      - self.recurrent_blocks  # shared layers, looped K times
      - self.coda_blocks       # unique last N layers
      - self.adapter           # Linear(2*dim, dim) to re-inject embeddings per loop
      - self.loop_scales       # Per-loop learned scaling factors (small)
```

**Key implementation pattern from Huginn:**

```python
# Source: Huginn paper (arXiv:2502.05171v2), adapted to our scale
class GPTRecurrent(nn.Module):
    def __init__(self, ...):
        # Prelude: unique entry layers
        self.prelude = nn.ModuleList([Block(...) for _ in range(n_prelude)])
        # Recurrent core: shared layers, looped
        self.core = nn.ModuleList([Block(...) for _ in range(n_core)])
        # Coda: unique exit layers
        self.coda = nn.ModuleList([Block(...) for _ in range(n_coda)])
        # Adapter: re-inject embeddings at each loop iteration
        # Huginn uses Linear(2*dim, dim) to map concat(state, embed) -> dim
        self.adapter = CastedLinear(2 * dim, dim, bias=False)
        # Per-loop scaling factors (lightweight adaptation)
        self.loop_scales = nn.Parameter(torch.ones(n_loops, dim))

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        embed = x  # save for re-injection

        # Prelude
        x0 = x
        for block in self.prelude:
            x = block(x, x0)

        # Recurrent core: loop K times
        for loop_idx in range(self.n_loops):
            # Re-inject embeddings (Huginn pattern)
            x = self.adapter(torch.cat([x, embed], dim=-1))
            # Per-loop scaling
            x = x * self.loop_scales[loop_idx].to(x.dtype)[None, None, :]
            for block in self.core:
                x = block(x, x0)

        # Coda
        for block in self.coda:
            x = block(x, x0)

        # Head
        x = self.final_norm(x).reshape(-1, x.size(-1))
        ...
```

**Compression benefit analysis:**
- Current model: 10 unique blocks, each ~2.5M params = ~25M block params
- With 5 unique + 2 loops: 5 blocks * 2.5M = 12.5M unique block params
- The shared weights appear identically in the state dict (stored once)
- zstd sees repeated patterns from loop re-use in activations, but the key saving is **halving the number of stored parameters**
- Estimated artifact savings: 5-6MB freed, which can be reinvested in wider layers (e.g., dim=640 instead of 512)

**Training stability (from Huginn paper):**
1. **Sandwich RMSNorm:** 4 RMSNorm layers per block (before/after attention and MLP) -- more than our current 2. Consider adding post-attention and post-MLP norms.
2. **Truncated backprop:** Huginn only backprops through last k=8 iterations. At our scale with 2-3 loops, backprop through all iterations (gradient path is short enough).
3. **Embedding re-injection:** Critical -- without re-injecting the input embedding at each loop, the model loses access to the original signal and quality degrades.
4. **Lower learning rate:** Huginn uses 5e-5 (lower than typical). Our Muon lr may need reduction (try 0.01 instead of 0.02).
5. **Initial state:** Huginn initializes s_0 from truncated normal. At our scale, the prelude output serves as s_0; no random init needed.

**U-Net skip connection interaction:**
The current U-Net skip connection code expects encoder/decoder halves. With recurrence, skip connections should either:
- Only apply within the prelude/coda (simpler)
- Be removed entirely (recurrence provides its own cross-depth information flow)
- Apply between loop iterations (experimental)

Recommendation: Start with Option C (no prelude/coda, just 5 recurrent blocks x 2 loops) and add prelude/coda only if needed. Drop U-Net skips for the recurrent model -- the embedding re-injection provides analogous information flow.

### Moonshot 2: Test-Time Training (LoRA TTT)

**Existing implementation pattern from TTT submission:**

```python
# Source: repo/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py
# Key hyperparameters (proven):
#   ttt_lora_rank = 8
#   ttt_lora_lr = 0.01
#   ttt_chunk_size = 256
#   ttt_eval_seq_len = 1024
#   ttt_batch_size = 64

# Architecture:
# 1. BatchedLinearLoRA: per-batch-element LoRA with A (bsz, rank, in) and B (bsz, out, rank)
# 2. BatchedTTTLoRA: bundles LoRA for lm_head + Q/V projections in all blocks
# 3. Per-document: reset LoRA, process chunks, score then train

# Critical rule compliance:
# - Score chunk FIRST (accumulate BPB), THEN train on it
# - Reset LoRA between documents (no cross-document leakage)
# - Only train on tokens already evaluated
```

**Adapting to our model:**

Key differences from the existing TTT submission to our model:
1. Our model has SmearGate, BigramHash, U-Net skips -- TTT LoRA targets only linear layers, so these are unaffected
2. Our model uses int5/int6 quantization -- TTT should operate on the dequantized FP16 model at eval time
3. Our eval uses sliding window (stride=64) -- TTT submission uses document-based chunking. These are different eval strategies.

**Critical insight from TTT ablations:**

```
Baseline (cross-doc, flat stream): 1.2278 BPB
+ Doc-isolated:                    1.2168 BPB  (-0.0110)
+ Stride (chunk=256):             1.1941 BPB  (-0.0337 total)
+ LoRA TTT:                       1.1910 BPB  (-0.0368 total, TTT adds only -0.0031)
```

The doc-isolated and strided evaluation provide most of the gain. Our model already uses sliding window eval (stride=64), which captures much of the stride benefit. The pure TTT marginal gain was only 0.003 BPB on the weak base model.

**Recommended TTT approach for our model:**
1. Keep sliding window eval (stride=64) as the scoring mechanism
2. Add LoRA adaptation after each scored window
3. Reset LoRA at document boundaries (detect via BOS tokens)
4. Target Q, V, and lm_head (same as proven submission)
5. Start with rank=4 (our model is larger/stronger, may need less adaptation)
6. Tune LR: start with 0.01, try 0.001 and 0.1

**Time budget concern:**
- TTT submission used ~1/10th of eval budget (so ~1 minute)
- With batch_size=64 documents, the overhead is manageable
- Our model is larger (more params to LoRA), but rank-4 keeps it small
- Estimate: 1-3 minutes added to eval time on 8xH100
- Must verify total eval + TTT stays under 10 minutes

### Moonshot 3: Curriculum Learning

**Implementation in existing training loop:**

```python
# Approach: Sort training data shards by difficulty
# Difficulty metric: compression ratio (zlib) per shard or per sequence

# OPTION A: Pre-sort shards (coarse-grained, simple)
# Before training, compute avg compression ratio per shard
# Sort shards easy-to-hard
# Modify TokenStream to iterate in sorted order instead of sequential

# OPTION B: Within-shard sorting (fine-grained, better)
# Preprocess: for each training sequence, compute zlib compression ratio
# Group into N=10 difficulty buckets
# During training, sample from progressively harder buckets

# OPTION C: Curriculum warmup (recommended by research)
# Use curriculum ordering for first 10-20% of training steps
# Then switch to random sampling for remainder
# This provides "sustained 3.5% improvement"
```

**Difficulty metric implementation:**

```python
# Source: arxiv.org/abs/2506.11300 -- compression ratio as difficulty metric
import zlib

def compute_difficulty(token_sequence: bytes) -> float:
    """Lower ratio = more compressible = easier text."""
    compressed = zlib.compress(token_sequence, 1)  # level 1 for speed
    return len(compressed) / len(token_sequence)
```

**Integration with TokenStream:**

The current `TokenStream` class reads shards sequentially. To implement curriculum:
1. Precompute difficulty scores for each shard (offline, one-time)
2. Sort `self.files` by average difficulty (ascending = easy first)
3. Optionally: after warmup fraction (e.g., 30% of steps), shuffle back to random

This requires minimal code change -- just reordering `self.files` in `TokenStream.__init__`.

**Preprocessing cost:**
- FineWeb training data: ~80 shards
- Computing zlib ratio per shard: read shard, compress representative sample, compute ratio
- Total preprocessing time: ~30 seconds (negligible)
- Can be done once and cached in a difficulty.json file

**Expected impact:**
- Research suggests 18-45% fewer steps to baseline performance
- At our 600s wall cap, this means each step is "more effective"
- Conservative estimate: 0.001-0.005 BPB improvement
- Best case (if combined with warmup strategy): 0.005-0.01 BPB

### Moonshot 4: Sub-5-bit Quantization (Int4 MLP)

**Current quantization code path:**

```python
# Source: repo/train_gpt.py lines 465-507
# Current: mixed_quantize_int6() calls quantize_intN_per_row()
# MLP layers use clip_range=15 (int5: values in [-16, 15])
# Attention layers use clip_range=31 (int6: values in [-32, 31])

# For int4: use clip_range=7 (values in [-8, 7])
# This is a ONE-LINE change in mixed_quantize_int6():

if cat in int6_cats and t.ndim >= 1:
    if cat == "mlp":
        clip = 7   # int4 instead of 15 (int5)
    else:
        clip = 31  # keep int6 for attention
    q, s = quantize_intN_per_row(t, clip_range=clip)
```

**Size savings analysis:**

```
Current int5 MLP storage per layer:
  fc: 512 * 1536 = 786,432 params * 5 bits = 491,520 bytes raw
  proj: 1536 * 512 = 786,432 params * 5 bits = 491,520 bytes raw
  Total per layer: 983,040 bytes raw
  After zstd-22 (~1.88x): ~522,872 bytes compressed

Int4 MLP storage per layer:
  fc: 786,432 params * 4 bits = 393,216 bytes raw
  proj: 786,432 params * 4 bits = 393,216 bytes raw
  Total per layer: 786,432 bytes raw
  After zstd-22 (~1.9x est): ~413,912 bytes compressed

Savings per layer: ~109KB compressed
Savings for 10 layers: ~1.09MB freed
```

Note: The actual bit-packing in the current code packs int5 values into int8 tensors (torch.int8 with clip_range=15). Going to clip_range=7 still stores as int8 but with more zeros/low values, which may compress even better under zstd. The size saving comes from better compression ratios, not from actual 4-bit packing (the code uses int8 storage with per-row scales).

**IMPORTANT REALIZATION:** The current code does NOT do actual bit-packing. It stores quantized values as int8 (1 byte per value) with per-row float16 scales. The "int5" and "int6" naming refers to the clip range, not the storage format. The compression savings come from zstd compressing the int8 values more effectively when the range is narrower.

This means:
- int5 (clip=15): values in [-16,15], stored as int8, compressed by zstd
- int4 (clip=7): values in [-8,7], stored as int8, compressed by zstd
- The savings are purely from better zstd compression of narrower-range int8 values
- Expected: int4 compresses ~10-20% better than int5 under zstd (narrower range = more repetitive patterns)

**Estimated savings:** ~0.5-1MB freed from artifact, which could be reinvested in:
- Wider model (dim=544 instead of 512)
- More BigramHash buckets (12K-16K)
- Additional prelude/coda layers in recurrent model

**Quality risk:**
- int4 has only 16 discrete levels vs int5's 32 levels -- 50% reduction in representational precision
- MLP layers are more tolerant than attention, but this is still aggressive
- QAT with STE helps the model learn to compensate
- Go/no-go: measure BPB impact of int4-MLP vs int5-MLP on same model

### Recommended Project Structure for Experiments

```
repo/
  train_gpt.py                    # Base model (keep untouched as reference)
  experiments/
    moonshot_recurrence.py         # Depth recurrence variant
    moonshot_ttt.py               # TTT eval variant
    moonshot_curriculum.py         # Curriculum data ordering
    moonshot_int4.py              # Int4 quantization variant
  scripts/
    eval/
      ablation_runner.py           # Existing ablation framework
      multiseed_eval.py           # Existing multi-seed eval
    curriculum/
      compute_difficulty.py        # Precompute shard difficulty scores
      difficulty.json             # Cached difficulty scores
```

Alternative approach (simpler): Create separate branches per moonshot, each modifying train_gpt.py directly.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LoRA implementation | Custom from scratch | Copy BatchedLinearLoRA from TTT submission | Proven, batched, handles per-document reset |
| Document boundary detection | Custom BOS scanning | Copy `_find_docs()` from TTT submission | Handles edge cases, BOS_ID=1 confirmed |
| Difficulty metric | Complex NLP features | zlib compression ratio | Research confirms it works as well as sophisticated metrics |
| Curriculum scheduling | Complex pacing functions | Simple shard sorting + warmup fraction | Linear pacing with 10 buckets matches complex schedules |
| Per-loop adaptation | Complex adapter networks | Learned scaling factors (1 per dim per loop) | Huginn uses adapter matrix but scaling factors suffice at small scale |

**Key insight:** The TTT submission provides battle-tested code for LoRA, document handling, and batched evaluation. Copy and adapt rather than reimplementing.

## Common Pitfalls

### Pitfall 1: TTT Rule Violation (Training on Future Tokens)
**What goes wrong:** Accidentally training LoRA on tokens that haven't been scored yet, which violates competition rules.
**Why it happens:** The score-then-train ordering is easy to reverse, especially when optimizing for speed.
**How to avoid:** Follow the existing TTT submission pattern exactly: `ptl = model(x, y, lora=cur_lora)` -> accumulate BPB -> `cur_opt.step()`. Score FIRST, train SECOND.
**Warning signs:** BPB improvement from TTT is suspiciously large (>0.01) -- likely indicates data leakage.

### Pitfall 2: Recurrence Gradient Explosion
**What goes wrong:** Gradients grow exponentially through multiple loops, causing NaN losses or training instability.
**Why it happens:** Each loop iteration multiplies gradients by the Jacobian of the shared block. With K loops, gradients scale as O(Jacobian^K).
**How to avoid:**
- Use sandwich RMSNorm (4 norms per block instead of 2)
- Start with 2 loops (short gradient path), increase only if stable
- Reduce learning rate (try matrix_lr=0.01 instead of 0.02)
- Monitor gradient norms per loop iteration
- Consider gradient clipping per loop (not just globally)
**Warning signs:** Loss spikes, NaN values, gradient norm increasing with more loops.

### Pitfall 3: Recurrence Model Not Compressing Better
**What goes wrong:** The recurrent model's artifact is not significantly smaller than the non-recurrent model because the shared weights are stored only once anyway (they're the same Python object).
**Why it happens:** In PyTorch, `nn.ModuleList` with shared modules stores weights once in the state_dict. The compression benefit is automatic -- shared blocks have identical parameters that compress perfectly.
**How to avoid:** Verify by checking `base_model.state_dict()` -- shared blocks should appear once. The artifact size savings are from having fewer unique parameters, not from compression of duplicates.
**Warning signs:** Artifact size not decreasing when switching from 10 unique to 5 shared blocks.

### Pitfall 4: Curriculum Overfitting to Easy Data
**What goes wrong:** Model over-trains on easy data during curriculum warmup, then struggles to adapt to hard data.
**Why it happens:** Short training budget (600s) means the curriculum warmup may consume too large a fraction of total training.
**How to avoid:**
- Limit curriculum warmup to first 10-20% of training steps
- Use linear pacing across 10 difficulty groups rather than strict easy-first ordering
- Monitor validation loss during curriculum vs. random phases
**Warning signs:** Validation loss improves during easy phase but stalls or regresses during hard phase.

### Pitfall 5: Int4 Destroying Attention Quality
**What goes wrong:** Accidentally applying int4 to attention layers (which are precision-sensitive), causing severe BPB degradation.
**Why it happens:** The `_classify_param()` function in train_gpt.py classifies layers by name patterns. A misclassification routes attention weights through int4.
**How to avoid:** Only apply int4 (clip_range=7) to MLP layers. Keep int6 (clip_range=31) for attention. Verify classification with print statements before running.
**Warning signs:** BPB degradation > 0.005 when switching to int4 -- likely hitting attention layers.

### Pitfall 6: TTT Exceeding Eval Time Budget
**What goes wrong:** Adding LoRA TTT adaptation causes evaluation to exceed the 10-minute budget.
**Why it happens:** Each LoRA gradient step adds ~2x overhead per chunk (forward + backward instead of just forward). With 50K documents and multiple chunks per document, this compounds.
**How to avoid:**
- Profile TTT eval time early and separately from training
- Use batch_size=64 for document batching (proven efficient in TTT submission)
- Consider applying TTT only to a subset of layers (Q, V only -- not all attention)
- Reduce rank from 8 to 4 to halve LoRA parameter count
- Use only 1 gradient step per chunk (proven sufficient)
**Warning signs:** Single-GPU eval taking >2 minutes with TTT enabled.

### Pitfall 7: Combining Moonshots Before Individual Validation
**What goes wrong:** Implementing recurrence + TTT + curriculum + int4 simultaneously, then not knowing which technique is helping or hurting.
**Why it happens:** Desire to see maximum improvement quickly.
**How to avoid:** Test each moonshot independently against the base model with a 1-seed quick check. Only combine techniques after individual validation.
**Warning signs:** Combined model BPB is worse than base model -- cannot debug because changes are entangled.

## Code Examples

### Depth Recurrence: Modified GPT Forward Pass

```python
# Source: Huginn paper + our codebase, adapted
# Key change: loop core blocks K times with embedding re-injection

class GPTRecurrent(nn.Module):
    def __init__(self, ..., n_core: int = 5, n_loops: int = 2):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(...) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        # All blocks are "core" in simplest version (no prelude/coda)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(n_core)
        ])
        # Adapter to re-inject embeddings per loop
        self.adapter = CastedLinear(2 * model_dim, model_dim, bias=False)
        # Per-loop scaling (lightweight per-iteration adaptation)
        self.loop_scales = nn.Parameter(torch.ones(n_loops, model_dim, dtype=torch.float32))
        self.final_norm = RMSNorm()
        # No skip_weights needed -- recurrence provides cross-depth flow

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        embed = x  # save for re-injection
        x0 = x

        for loop_idx in range(self.n_loops):
            if loop_idx > 0:
                # Re-inject embeddings (Huginn pattern)
                x = self.adapter(torch.cat([x, embed], dim=-1))
            # Per-loop scaling
            scale = self.loop_scales[loop_idx].to(x.dtype)[None, None, :]
            x = x * scale
            for block in self.blocks:
                x = block(x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
```

### TTT LoRA: Minimal Integration with Sliding Window

```python
# Source: repo/records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py, adapted
# Key: integrate TTT with our existing sliding window eval

class BatchedLinearLoRA(nn.Module):
    """Per-batch-element LoRA. x @ A^T @ B^T = LoRA delta."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()

# Integration: modify eval_val_sliding to accept optional LoRA
# Score window -> backprop through LoRA -> step optimizer -> next window
# Reset LoRA at document boundaries (BOS token detection)
```

### Curriculum: Shard Difficulty Scoring

```python
# Precompute difficulty per training shard
import json, zlib
from pathlib import Path

def score_shard_difficulty(shard_path: Path, sample_size: int = 100_000) -> float:
    """Compute average compression ratio for a shard. Lower = easier."""
    tokens = load_data_shard(shard_path)
    sample = tokens[:sample_size].numpy().tobytes()
    compressed = zlib.compress(sample, 1)
    return len(compressed) / len(sample)

def sort_shards_by_difficulty(pattern: str) -> list[Path]:
    """Sort training shards easy-to-hard by compression ratio."""
    files = sorted(glob.glob(pattern))
    difficulties = {f: score_shard_difficulty(Path(f)) for f in files}
    return [Path(f) for f in sorted(files, key=lambda f: difficulties[f])]

# Save difficulty scores for reuse
# difficulties = {str(f): score_shard_difficulty(f) for f in files}
# json.dump(difficulties, open("scripts/curriculum/difficulty.json", "w"))
```

### Int4 Quantization: One-Line Change

```python
# Source: repo/train_gpt.py lines 496-501
# Current code (int5 for MLP):
if cat in int6_cats and t.ndim >= 1:
    clip = 15 if cat == "mlp" else 31   # int5 for MLP, int6 for attention
    q, s = quantize_intN_per_row(t, clip_range=clip)

# Int4 change:
if cat in int6_cats and t.ndim >= 1:
    clip = 7 if cat == "mlp" else 31    # int4 for MLP, int6 for attention
    q, s = quantize_intN_per_row(t, clip_range=clip)

# That's it. One number changed: 15 -> 7
# clip_range=7 means values in [-8, 7], which is 4-bit range
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Uniform int6 quantization | Mixed int5/int6 (SOTA #1) | March 2026 | Enabled 10th layer, -0.003 BPB |
| Random data ordering | Curriculum learning (research) | 2025-2026 papers | 18-45% faster convergence |
| 10 unique layers | Depth recurrence (Huginn) | Jan 2025 | 25-55% param savings at same quality |
| No eval-time adaptation | LoRA TTT | March 2026 (competition) | -0.003 BPB on weak base |
| Document-agnostic eval | Document-isolated eval | March 2026 (TTT submission) | -0.011 BPB (larger than TTT itself) |

**Deprecated/outdated:**
- Full-parameter TTT: Too expensive for eval time budget. LoRA TTT is the standard now.
- Uniform quantization: Mixed precision strictly dominates.
- Post-training quantization (PTQ): QAT strictly dominates at int5 and below.

## Open Questions

### 1. Depth recurrence quality at 16MB scale
- **What we know:** Huginn shows recurrence works at 3.5B params. RingFormer matches transformers at 20% params.
- **What's unclear:** No evidence at 10-15M param scale. The adapter matrix adds ~0.5M params. Training may not converge in 600s with shared blocks.
- **Recommendation:** Quick 1-seed experiment with 5 blocks x 2 loops. Compare BPB and artifact size vs 10 unique blocks. If BPB is within 0.005 and artifact is >1MB smaller, proceed.

### 2. TTT improvement on strong base model
- **What we know:** TTT on weak base (1.19 BPB) gives -0.003 BPB. Document isolation gives -0.011 BPB.
- **What's unclear:** Does TTT provide diminishing returns on a 1.14 BPB model? The strong model may already capture patterns that TTT would learn.
- **Recommendation:** Quick 1-seed test of TTT on our current model. If TTT adds <0.001 BPB, it's not worth the eval time cost. The document-isolation trick may already be captured by our sliding window eval.

### 3. Curriculum + wall clock interaction
- **What we know:** Curriculum provides 18-45% fewer steps to baseline. Our training is wall-clock limited (600s).
- **What's unclear:** If curriculum makes each step "count more," does it help when we're already optimizing for steps-per-second? The benefit may be that we reach a better BPB in the same number of steps.
- **Recommendation:** Test with curriculum warmup (first 30% of steps easy-to-hard, then random). Compare final BPB vs random ordering.

### 4. Int4 quality impact in our specific model
- **What we know:** Int4 has 16 levels vs int5's 32. ParetoQ says int4 is manageable with QAT. Our current code supports clip_range=7.
- **What's unclear:** How much BPB degradation int4 causes on our specific relu-squared MLP weights, which have a specific value distribution.
- **Recommendation:** Fastest experiment -- change one number (15 -> 7) and run 1 seed. Measure BPB delta. If <0.002, proceed; if >0.005, abandon.

### 5. Interaction between moonshots
- **What we know:** TTT is orthogonal to everything (eval-time only). Recurrence changes the model architecture. Curriculum changes data ordering. Int4 changes quantization.
- **What's unclear:** Does recurrence + int4 compound (fewer params but lower precision)? Does curriculum help recurrence more (shared blocks need better training signal)?
- **Recommendation:** Test each independently first. Only combine the top 2 performers.

## Sources

### Primary (HIGH confidence)
- [Huginn architecture paper (arXiv:2502.05171v2)](https://arxiv.org/abs/2502.05171) -- Prelude/Core/Coda structure, adapter matrix, sandwich RMSNorm, truncated backprop
- [TTT submission](repo/records/track_10min_16mb/2026-03-17_LoRA_TTT/) -- BatchedLinearLoRA, document isolation, ablations showing TTT adds only -0.003 BPB
- [Current train_gpt.py](repo/train_gpt.py) -- GPT class, Block, quantize_intN_per_row, mixed_quantize_int6, eval_val_sliding

### Secondary (MEDIUM confidence)
- [Curriculum learning for LLM pretraining (arXiv:2506.11300)](https://arxiv.org/abs/2506.11300) -- compression ratio metric, 18-45% convergence speedup, warmup strategy
- [ParetoQ (arXiv:2502.02631)](https://arxiv.org/abs/2502.02631) -- learning transition at 2-3 bits, int4 manageable with QAT
- [Test-Time Learning for LLMs (arXiv:2505.20633)](https://arxiv.org/abs/2505.20633) -- LoRA for eval-time adaptation, perplexity minimization
- [RingFormer / depth-recurrent transformers](https://www.emergentmind.com/topics/depth-recurrent-transformer) -- 20% parameter count matching standard transformers

### Tertiary (LOW confidence)
- Int4 artifact savings estimate (calculated, not empirically verified with our specific weight distributions)
- Curriculum impact on our training pipeline (extrapolated from research on different model sizes)
- TTT improvement on strong base model (no data -- only one competition submission on weak base)
- Compression ratio of shared vs unique weights under zstd-22 (theoretical argument, not measured)

## Metadata

**Confidence breakdown:**
- Depth recurrence patterns: MEDIUM -- well-documented at large scale (Huginn), unverified at 16MB. Implementation pattern is clear from paper.
- TTT implementation: HIGH -- exact code available from competition submission. Only question is marginal value on strong base.
- Curriculum learning: MEDIUM -- strong research backing, but unknown interaction with 600s wall clock and Muon optimizer.
- Sub-5-bit quantization: MEDIUM -- trivial code change (clip_range 15->7), but quality impact unknown for our specific model.
- Overall phase: MEDIUM -- each technique individually understood, but combined impact at competition scale is uncertain.

**Research date:** 2026-03-23
**Valid until:** 2026-04-15 (competition deadline is April 30; techniques are stable but leaderboard may shift)

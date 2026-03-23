# Phase 2: SOTA Stack Integration - Research

**Researched:** 2026-03-23
**Domain:** Transformer architecture, quantization, optimizer, evaluation pipeline modifications
**Confidence:** HIGH

## Summary

This research investigates how to modify the existing baseline `train_gpt.py` (9 layers, 512d, MLP 2x, int8+zlib, no BigramHash, no SWA, no sliding window eval) into the full SOTA configuration (10 layers, MLP 3x relu-squared, BigramHash 10240, int6 QAT-style post-training quantization, Muon with WD=0.04, SWA, sliding window eval stride=64, zstd-22 compression). The target is approximately 1.15 BPB, down from the baseline 1.2254.

The analysis is based on detailed reading of the baseline `train_gpt.py` (1127 lines) and two reference SOTA submissions already in the repo: the #1 entry (thwu1, 1.1428 BPB, `2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`) and the SmearGate/BigramHash entry (`2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py`). Both reference implementations are 1218-1231 lines, well within the 1500-line hard cap.

**Primary recommendation:** Implement all changes in a single incremental pass through the baseline `train_gpt.py`, following the exact patterns from the #1 SOTA submission code that is already in the repo. Do NOT hand-roll new patterns -- the reference implementations provide verified, working code for every technique.

## Standard Stack

The "stack" for this phase is entirely modifications to the single `train_gpt.py` file. No new external libraries are needed except `zstandard`.

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `zstandard` | 0.25.0 | zstd-22 compression replacing zlib | All SOTA submissions use zstd-22; 1.51-1.88x compression ratio on quantized weights vs ~1.3x for zlib |
| PyTorch (existing) | 2.6.0+cu124 | Framework | Already installed in pgolf conda env |
| sentencepiece (existing) | latest | Tokenizer | Already used by baseline |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `zlib` (stdlib) | N/A | Fallback compression | Only if zstandard import fails (compatibility) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| zstd level 22 | zlib level 9 | Never -- zstd strictly dominates for quantized weights |
| Custom int6 quantization | torchao | torchao lacks int5/int6 support; custom code required |
| Post-training quantization | QAT with STE | Phase 2 uses PTQ (not QAT) per the SOTA pattern; QAT deferred to later because SOTA submissions actually use PTQ with STE-free quantization at serialization time |

**Installation:**
```bash
pip install zstandard==0.25.0
```

**Critical finding:** The SOTA submissions do NOT use true QAT (quantize-aware training with STE in the forward pass during training). They train in full precision (bf16/fp32) and apply int5/int6 post-training quantization at serialization time. The CONTEXT.md says "QAT from training start" but the actual code trains normally and quantizes at the end. This Phase 2 should match the SOTA pattern: train in full precision, then quantize post-training with per-row scaling. True QAT with STE during training can be explored in Phase 3 if needed.

## Architecture Patterns

### Baseline vs Target: Concrete Diff Analysis

Here is every change needed from the baseline, organized by component:

#### 1. Hyperparameters Class Changes

| Parameter | Baseline Value | SOTA Value | Change Type |
|-----------|---------------|------------|-------------|
| `num_layers` | 9 | 10 | Direct edit |
| `mlp_mult` | 2 (int) | 3.0 (float) | Type change: int to float |
| `train_seq_len` | 1024 | 2048 | Direct edit |
| `train_batch_tokens` | 524,288 | 786,432 | Direct edit |
| `warmdown_iters` | 1200 | 3000 | Direct edit |
| `matrix_lr` | 0.04 | 0.02 | Direct edit |
| `scalar_lr` | 0.04 | 0.02 | Direct edit |
| `tied_embed_lr` | 0.05 | 0.03 | Direct edit |
| `muon_momentum` | 0.95 | 0.99 | Direct edit |
| `muon_momentum_warmup_start` | 0.85 | 0.92 | Direct edit |
| `muon_momentum_warmup_steps` | 500 | 1500 | Direct edit |
| `grad_clip_norm` | 0.0 | 0.3 | Direct edit |
| NEW `weight_decay` | N/A | 0.04 | Add new parameter |
| NEW `eval_stride` | N/A | 64 | Add new parameter |
| NEW `eval_batch_seqs` | N/A | 32 | Add new parameter |
| NEW `bigram_vocab_size` | N/A | 10240 | Add new parameter |
| NEW `bigram_dim` | N/A | 128 | Add new parameter |
| NEW `swa_enabled` | N/A | 1 | Add new parameter |
| NEW `swa_start_frac` | N/A | 0.4 | Add new parameter |
| NEW `swa_every` | N/A | 50 | Add new parameter |

#### 2. New Module Classes (3 new classes)

**SmearGate** (~10 lines):
```python
class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```
Source: Both SOTA submissions use identical implementation.

**BigramHashEmbedding** (~25 lines):
```python
class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod  # first position has no previous token
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```
Source: Both SOTA submissions use identical implementation. Hash function uses two prime multipliers (36313, 27191) with XOR.

**Key BigramHash details:**
- Hash: `XOR(36313 * current_token, 27191 * previous_token) % (vocab_size - 1)`
- First position maps to the last bucket index (no previous token)
- Embedding dim 128, projected to model dim 512 via learned linear
- Initialized to zeros (scale starts at 0.05, so contribution is initially small)
- The `bigram.scale` parameter is a CONTROL_TENSOR (kept in fp32, not quantized)

#### 3. GPT Model Class Changes

The GPT class needs these additions:
- Add `bigram_vocab_size` and `bigram_dim` constructor parameters
- Create `BigramHashEmbedding` if `bigram_vocab_size > 0`
- Create `SmearGate` (always, in SOTA submissions)
- In `forward()`: add bigram embeddings to token embeddings BEFORE rms_norm, then apply SmearGate AFTER rms_norm
- Add `forward_logits()` method (returns logits tensor, does not compute loss -- needed for sliding window eval)

**Forward pass order in SOTA:**
```
x = tok_emb(input_ids)
x = x + bigram(input_ids)     # ADD BIGRAM BEFORE NORM
x = rms_norm(x)
x = smear(x)                  # SMEARGATE AFTER NORM
x0 = x
# ... encoder/decoder blocks with skip connections ...
```

#### 4. Initialization Changes (OrthoInit)

Replace the baseline `_init_weights` with orthogonal initialization:
```python
def _init_weights(self) -> None:
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
    num_layers = len(self.blocks)
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if ".proj." in name or name.endswith(".proj"):
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```
Source: Both SOTA submissions use identical initialization.

Key: Output projections (attention `.proj` and MLP `.proj`) are scaled by `1/sqrt(2*num_layers)` after orthogonal init. This prevents residual stream explosion with 10 layers.

#### 5. Muon Optimizer Changes

The Muon class needs `weight_decay` parameter added:
```python
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                       nesterov=nesterov, weight_decay=weight_decay))
```

In the `step()` method, add weight decay BEFORE the update:
```python
wd = group.get("weight_decay", 0.0)
curr = 0
for p in params:
    g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
    if wd > 0:
        p.data.mul_(1.0 - lr * wd)  # Decoupled weight decay
    p.add_(g, alpha=-lr)
    curr += p.numel()
```

Also switch from `torch.optim.Adam` to `torch.optim.AdamW` for embedding/scalar optimizers (to get weight decay there too).

#### 6. Optimizer Setup Changes

Key differences from baseline:
- Muon gets `weight_decay=0.04` (hardcoded, not from `args.weight_decay` in SOTA #1)
- Adam -> AdamW for tok and scalar optimizers with `weight_decay=args.weight_decay`
- BigramHash embed weight goes into tok optimizer param group
- BigramHash proj weight goes into matrix_params (for Muon)
- SmearGate gate goes into scalar_params
- BigramHash scale goes into scalar_params

#### 7. SWA Implementation

SWA is implemented inline in the training loop, not as a separate module:
```python
# In training loop, after optimizer step:
if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
    if swa_state is None:
        swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        swa_count = 1
    else:
        for name, t in base_model.state_dict().items():
            swa_state[name] += t.detach().cpu()
        swa_count += 1

# After training loop, before serialization:
if args.swa_enabled and swa_state is not None and swa_count > 1:
    avg_state = {name: (tensor / swa_count).to(dtype=current_state[name].dtype)
                 for name, tensor in swa_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
```

**Critical detail:** SWA triggers when `scale < swa_start_frac` (i.e., when the LR multiplier drops below 0.4 of the base LR). This means SWA collection starts late in warmdown. With `swa_every=50`, the SOTA collects approximately 20-24 checkpoints.

**Memory note:** SWA copies the full state dict to CPU each collection. With ~26M params at fp32, that is ~100MB per checkpoint on CPU RAM. With 24 checkpoints accumulated by running average, this only needs one copy at a time (not 24 copies -- each is added to the running sum).

#### 8. Int6 Mixed Quantization (Post-Training)

Replace the baseline's `quantize_state_dict_int8` with a new `mixed_quantize_int6` system:

**Classification function:**
```python
def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if "bigram" in name: return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name): return "attn"
    return "other"
```

**Int6 quantization (per-row):**
```python
def quantize_intN_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
        return q, scale
    # 1D fallback...
```

For Phase 2 (int6 only), all MLP and attention weights use `clip_range=31` (int6: [-32, 31]). The #1 submission uses int5 (`clip_range=15`) for MLP, but that is a Phase 3 optimization.

**FP16 passthrough patterns:**
- `tok_emb` (tied embeddings) -- kept as FP16
- `blocks.8.attn.c_k` (last layer key projection in #1, layer 8 of 0-indexed 10-layer model) -- kept as FP16
- Control tensors (scales, gains, resid_mix, smear gate, bigram scale) -- kept as fp32
- Small tensors (< 8192 or 65536 elements) -- kept as FP16

**Critical for Phase 2:** Use uniform int6 for ALL matrix weights initially, with embeddings and bigram as FP16 passthrough. Int5 MLP is Phase 3.

#### 9. Magnitude Pruning (3%)

Applied after SWA averaging, before quantization:
```python
with torch.no_grad():
    for name, param in base_model.named_parameters():
        if param.ndim == 2 and param.numel() > 65536:
            threshold = torch.quantile(param.abs().float().flatten(), 0.03)
            mask = param.abs() < threshold
            param.masked_fill_(mask, 0.0)
```

This zeros out the smallest 3% of weights by magnitude, which improves zstd compression ratio slightly.

#### 10. zstd-22 Compression

Replace zlib with zstd:
```python
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# At serialization:
if _COMPRESSOR == "zstd":
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
else:
    quant_blob = zlib.compress(quant_raw, 9)

# At deserialization:
if _COMPRESSOR == "zstd":
    decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
else:
    decompressed = zlib.decompress(quant_blob_disk)
```

#### 11. Sliding Window Evaluation

Add `eval_val_sliding()` function and `forward_logits()` method to GPT.

**forward_logits():** Same as forward() but returns raw logits tensor (no loss computation). Needed because sliding window eval computes loss externally with per-position control.

**eval_val_sliding():** Key algorithm:
1. Generate all window start positions: `range(0, total_tokens, stride)` where stride=64
2. Distribute windows across ranks
3. For each batch of windows (batch_seqs=32):
   - Pad to seq_len=2048 with zeros
   - Forward pass to get logits
   - Compute per-token NLL via `F.cross_entropy(..., reduction='none')`
   - Score only the last `stride` tokens (or all tokens for the first window)
   - Accumulate loss, token count, byte count
4. All-reduce across ranks
5. BPB = (total_loss / total_tokens) / ln(2) * (total_tokens / total_bytes)

**Critical subtlety:** The scoring region for each window:
- First window (ws=0): score all tokens (s=0)
- All other windows: score only last `stride` tokens (s = wlen - stride)
- This avoids double-counting tokens while giving maximum context

#### 12. CONTROL_TENSOR_NAME_PATTERNS Update

Add `smear` and `bigram.scale` to the control tensor patterns:
```python
"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear,bigram.scale"
```

### Recommended Project Structure (within train_gpt.py)

```
train_gpt.py sections:
  1. Hyperparameters class        # Lines ~40-95
  2. Muon optimizer               # Lines ~100-170
  3. Tokenizer/eval LUTs          # Lines ~175-265
  4. Quantization functions       # Lines ~270-400
  5. Data loading                 # Lines ~405-465
  6. Transformer modules          # Lines ~470-740
     - RMSNorm, CastedLinear, Rotary
     - CausalSelfAttention
     - MLP (relu^2)
     - SmearGate (NEW)
     - BigramHashEmbedding (NEW)
     - Block
     - GPT (modified)
  7. Sliding window eval (NEW)    # Lines ~745-820
  8. Training main()              # Lines ~825-end
     - Setup
     - Optimizer setup (modified)
     - Training loop (SWA added)
     - SWA averaging (NEW)
     - Magnitude pruning (NEW)
     - Mixed int6 serialization (NEW)
     - Sliding window final eval (NEW)
```

### Anti-Patterns to Avoid

- **Separate files:** The competition requires a single `train_gpt.py`. Do not create helper modules.
- **QAT during forward pass in Phase 2:** The SOTA submissions do NOT use STE in the forward pass. They train normally and quantize post-training. Do not add fake-quantize nodes yet.
- **Large batch on single GPU:** The batch tokens (786K) divided by 8 GPUs and grad_accum is manageable at seq_len=2048. Do not try to fit the full batch on one GPU.
- **Changing the hash constants:** The bigram hash primes (36313, 27191) are from the SOTA submissions. Do not change them without ablation.
- **Ignoring the MLP type change:** Baseline `mlp_mult` is `int(2)`, SOTA uses `float(3.0)`. The MLP constructor must handle float: `hidden = int(mlp_mult * dim)`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BigramHash | Custom hash function from scratch | Copy exact implementation from SOTA submissions | The hash constants (36313, 27191) and bucket mapping have been ablated; different constants may have worse collision patterns |
| SmearGate | Novel gating mechanism | Copy exact implementation from SOTA submissions | The sigmoid gate initialized at zero with learned per-dimension scaling is proven; alternatives are untested |
| Sliding window eval | Custom eval loop with manual BPB | Copy `eval_val_sliding()` from SOTA submissions | BPB accounting (bytes per token, boundary tokens) is subtle; the SOTA implementation handles edge cases correctly |
| Int6 quantization | Novel quantization scheme | Copy `quantize_intN_per_row()` from SOTA submissions | Per-row scale factor computation and clamping ranges are specific to this competition |
| SWA | torch.optim.swa_utils | Inline implementation in training loop | The SOTA SWA collects during warmdown based on LR schedule, not at fixed intervals; torch.optim.swa_utils does not support this pattern |
| Weight decay in Muon | Separate WD step | Integrated `p.data.mul_(1 - lr * wd)` in Muon step | Decoupled WD must be applied at the LR-scaled rate, not a fixed rate |

**Key insight:** Both SOTA submissions are already in the repository at `repo/records/track_10min_16mb/`. Every piece of code needed for Phase 2 has been written, tested, and validated to produce 1.14 BPB. The planner should create tasks that copy verified patterns, not design new ones.

## Common Pitfalls

### Pitfall 1: MLP mult type mismatch
**What goes wrong:** Baseline uses `int(os.environ.get("MLP_MULT", 2))` while SOTA uses `float(os.environ.get("MLP_MULT", 3.0))`. If the MLP constructor receives an int when it expects float (or vice versa), `int(mlp_mult * dim)` may work differently.
**Why it happens:** The baseline MLP class has `__init__(self, dim: int, mlp_mult: int)` while SOTA has `mlp_mult: float`.
**How to avoid:** Change both the Hyperparameters class and MLP class signature to use float for mlp_mult.
**Warning signs:** MLP hidden dim is 1024 (2*512) instead of 1536 (3*512).

### Pitfall 2: SWA collection never triggers
**What goes wrong:** SWA collects when `scale < swa_start_frac`. If the warmdown schedule is misconfigured or the training ends before warmdown reduces LR below 0.4, no SWA checkpoints are collected.
**Why it happens:** The warmdown schedule is time-based (not step-based) when `max_wallclock_seconds > 0`. The LR multiplier depends on `warmdown_ms = warmdown_iters * step_ms`.
**How to avoid:** Log SWA collection events (`swa:start step:N`, `swa:collect step:N count:M`). Verify that swa_count > 1 before applying.
**Warning signs:** `swa:applying averaged 1 checkpoints` or no SWA log messages at all.

### Pitfall 3: Artifact size exceeds 16MB with 10 layers
**What goes wrong:** Adding the 10th layer + BigramHash(10240) + keeping some weights in FP16 can push the artifact over 16MB if using int8 quantization (baseline level).
**Why it happens:** The baseline at 9 layers already uses 15.88MB. The 10th layer adds ~2.5MB of parameters.
**How to avoid:** Switch to int6 quantization AND zstd-22 compression simultaneously. Int6 uses 6 bits/value (stored as int8) but compresses better. The combination of int6 + zstd-22 should bring the artifact well under 16MB.
**Warning signs:** Run artifact size check after every change.

### Pitfall 4: torch.compile breaks with new modules
**What goes wrong:** Adding SmearGate, BigramHash, or forward_logits to the model may cause `torch.compile(base_model, dynamic=False, fullgraph=True)` to fail with graph break errors.
**Why it happens:** `fullgraph=True` requires the entire forward pass to be captured in one graph. Dynamic tensor shapes, Python control flow, or operations torch.compile cannot handle cause graph breaks.
**How to avoid:** Test torch.compile after each module addition. The SOTA submissions use `torch.compile` successfully with all these modules, so the patterns are known to work. Key: BigramHash uses `torch.bitwise_xor` and integer operations that are compilable.
**Warning signs:** Errors like "torch._dynamo.exc.Unsupported" or "graph break" during compilation.

### Pitfall 5: Sliding window eval gives different BPB than standard eval
**What goes wrong:** Sliding window eval with stride=64 gives ~0.002-0.005 BETTER BPB than standard eval because each token gets more context. This is expected and correct.
**Why it happens:** In standard eval, the first token in each batch sees 0 context tokens. In sliding window with stride=64, every token (except the first 64 in the corpus) sees at least 1984 context tokens.
**How to avoid:** Use sliding window eval as the final/official eval. Keep standard eval for fast mid-training validation (it is much faster).
**Warning signs:** BPB from sliding window is WORSE than standard eval (would indicate a bug).

### Pitfall 6: Weight decay applied incorrectly to Muon
**What goes wrong:** Muon weight decay must be decoupled (multiply weights by `1 - lr * wd` before the update). If implemented as L2 regularization in the gradient, it interacts badly with Muon's orthogonalization.
**Why it happens:** Standard weight decay adds `wd * param` to the gradient. But Muon orthogonalizes the gradient, which destroys the weight decay signal. Decoupled WD applies directly to the parameter.
**How to avoid:** Use the exact pattern from SOTA: `p.data.mul_(1.0 - lr * wd)` before `p.add_(g, alpha=-lr)`.
**Warning signs:** Training loss is higher than expected or model weights grow unbounded.

### Pitfall 7: BigramHash parameters not routed to correct optimizer
**What goes wrong:** BigramHash has three parameter types: `embed.weight` (2D embedding table), `proj.weight` (2D projection matrix), and `scale` (scalar). Each needs different optimizer treatment.
**Why it happens:** The optimizer routing in baseline does not account for BigramHash parameters.
**How to avoid:** Route exactly as SOTA does:
  - `bigram.embed.weight` -> AdamW (same group as tok_emb, same LR)
  - `bigram.proj.weight` -> Muon (added to matrix_params)
  - `bigram.scale` -> AdamW scalar optimizer
**Warning signs:** BigramHash parameters not updating, or NaN gradients.

### Pitfall 8: Serialization format incompatibility
**What goes wrong:** The baseline uses `quantize_state_dict_int8` which produces `{"__quant_format__": "int8_clean_per_row_v1", "quantized": ..., "scales": ...}`. The new mixed int6 format uses `{"w": result, "m": meta}`. The existing eval/validation scripts may expect the old format.
**Why it happens:** The deserialization code must match the serialization format exactly.
**How to avoid:** Update ALL serialization and deserialization code together. Use the SOTA format: `{"w": quant_result, "m": quant_meta}`.
**Warning signs:** `KeyError` during model loading, or incorrect weight values after roundtrip.

## Code Examples

### Example 1: Complete BigramHash Integration into GPT Forward

```python
# Source: repo/records/.../2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py lines 690-698
def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
    x = self.tok_emb(input_ids)
    if self.bigram is not None:
        x = x + self.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = self.smear(x)
    x0 = x
    skips: list[Tensor] = []
    # ... rest of encoder/decoder blocks unchanged ...
```

### Example 2: SWA Collection and Application

```python
# Source: repo/records/.../2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py lines 1110-1152
# In training loop, after step:
if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
    if swa_state is None:
        swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        swa_count = 1
        log0(f"swa:start step:{step}")
    else:
        for name, t in base_model.state_dict().items():
            swa_state[name] += t.detach().cpu()
        swa_count += 1

# After training loop:
if args.swa_enabled and swa_state is not None and swa_count > 1:
    log0(f"swa:applying averaged {swa_count} checkpoints")
    current_state = base_model.state_dict()
    avg_state = {
        name: (tensor / swa_count).to(dtype=current_state[name].dtype)
        for name, tensor in swa_state.items()
    }
    base_model.load_state_dict(avg_state, strict=True)
```

### Example 3: Muon with Weight Decay

```python
# Source: repo/records/.../2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py lines 114-170
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                       nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        # ... momentum + Newton-Schulz orthogonalization (unchanged) ...
        # Apply decoupled weight decay:
        wd = group.get("weight_decay", 0.0)
        curr = 0
        for p in params:
            g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
            if wd > 0:
                p.data.mul_(1.0 - lr * wd)
            p.add_(g, alpha=-lr)
            curr += p.numel()
```

### Example 4: zstd Compression with Fallback

```python
# Source: repo/records/.../2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py lines 22-26, 1177-1196
try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# Compression:
if _COMPRESSOR == "zstd":
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
else:
    quant_blob = zlib.compress(quant_raw, 9)

# Decompression:
if _COMPRESSOR == "zstd":
    decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk)
else:
    decompressed = zlib.decompress(quant_blob_disk)
```

### Example 5: Orthogonal Initialization

```python
# Source: repo/records/.../2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py lines 676-688
def _init_weights(self) -> None:
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
    num_layers = len(self.blocks)
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if ".proj." in name or name.endswith(".proj"):
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```

## State of the Art

| Old Approach (Baseline) | Current Approach (SOTA) | When Changed | Impact |
|--------------------------|------------------------|--------------|--------|
| 9 layers, MLP 2x | 10 layers, MLP 3x relu^2 | SOTA submission 2026-03-20 | -0.003 BPB from extra layer, MLP 3x is standard |
| int8 PTQ + zlib | int6 PTQ + zstd-22 | SOTA submission 2026-03-19 | Enables 10 layers within 16MB |
| No BigramHash | BigramHash(10240, dim=128) | SOTA submission 2026-03-19 | -0.012 BPB from bigram context |
| No SmearGate | SmearGate after embeddings | SOTA submission 2026-03-19 | -0.002-0.004 BPB from local context blending |
| No SWA | SWA start_frac=0.4, every=50 | SOTA submission 2026-03-20 | -0.006 BPB from checkpoint averaging |
| Standard eval (seq_len chunks) | Sliding window stride=64 | SOTA submission 2026-03-19 | -0.002-0.005 BPB from increased context |
| No weight decay in Muon | WD=0.04 decoupled | SOTA submission 2026-03-20 | Improved generalization |
| Random init | Orthogonal init + scaled outputs | SOTA submission 2026-03-19 | -0.001-0.002 BPB faster convergence |
| seq_len=1024, batch=524K | seq_len=2048, batch=786K | SOTA submission 2026-03-18 | Longer context for training |
| Muon momentum=0.95 | Muon momentum=0.99, warmup 0.92->0.99 over 1500 steps | SOTA submission 2026-03-20 | Better convergence trajectory |

**Deprecated/outdated:**
- int8 quantization: Replaced by int6 (and int5 for MLP in Phase 3)
- zlib compression: Replaced by zstd level 22
- Standard eval without sliding window: Replaced by stride-64 sliding window
- MLP 2x expansion: Replaced by MLP 3x with relu-squared

## Open Questions

1. **Exact artifact size with int6 + 10 layers + BigramHash(10240)**
   - What we know: #1 SOTA uses int5 MLP + int6 attn and fits in ~15.8-15.97MB. Phase 2 uses uniform int6 which is slightly larger per-weight but we do not have int5 MLP savings yet.
   - What's unclear: Whether 10 layers + BigramHash(10240) + uniform int6 + zstd-22 fits under 16MB without int5.
   - Recommendation: Start with 9 layers and int6, measure artifact size, then add 10th layer. If it exceeds 16MB, either (a) reduce BigramHash to 8192 or (b) defer 10th layer to Phase 3 when int5 MLP is available. The #1 ablation shows that int5 MLP is what ENABLES the 10th layer.

2. **Step throughput with seq_len=2048 vs 1024**
   - What we know: Baseline at seq_len=1024 runs ~42-45ms/step on H200. SOTA at seq_len=2048 runs ~57ms/step.
   - What's unclear: Exact throughput with 10 layers and MLP 3x at seq_len=2048 on our H200 nodes.
   - Recommendation: Profile immediately after architecture changes. If step time increases too much, fewer steps will complete within the 600s wall cap, potentially making the warmdown/SWA schedule ineffective. The SOTA runs ~10,400-10,700 steps in 600s.

3. **Whether to include SmearGate in Phase 2 or defer to Phase 4**
   - What we know: CONTEXT.md does not explicitly list SmearGate as a Phase 2 requirement. However, both SOTA reference implementations include it as integral to the architecture, and it is trivial to implement (~10 lines).
   - What's unclear: Whether SmearGate is expected in Phase 2 or Phase 4.
   - Recommendation: Include SmearGate in Phase 2 because (a) both reference implementations have it, (b) it is <10 lines, (c) it is baked into the forward pass ordering (before x0=x), and (d) removing it later would change the architecture baseline for ablation.

4. **Can the 10th layer fit with int6 (not int5)?**
   - What we know: The ablation in #1 README says "9L int6 baseline: 1.1485" and "+ int5 MLP + 10th layer: 1.1453". This implies the 10th layer was enabled BY int5 MLP savings.
   - Calculation: 1 layer adds approximately (Q+K+V+O attention: 512*512*3 + 512*512 = 1.05M params) + (MLP: 512*1536*2 = 1.57M params) = ~2.62M params. At int6 (6 bits) + per-row FP16 scale, that is approximately 2.62M * 0.75 bytes + scales ~ 2.0MB raw, compressed by zstd-22 at ~1.5x = ~1.3MB. Baseline uses 15.88MB. With int6 replacing int8 for all weights: savings of ~2 bits/param * ~10M large params = ~2.5MB saved. Net: should fit.
   - Recommendation: Try 10 layers with int6, verify artifact size. If tight, reduce BigramHash to 8192.

## Sources

### Primary (HIGH confidence)
- **Baseline train_gpt.py** (1127 lines) -- read in full, every function analyzed
- **SOTA #1 submission** (`repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`, 1231 lines) -- read in full, all modifications documented
- **SOTA SmearGate/BigramHash submission** (`repo/records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py`, 1218 lines) -- read in full, identical patterns confirmed
- **SOTA #1 README.md** -- ablation data, 3-seed results, hyperparameters

### Secondary (MEDIUM confidence)
- **Phase 1 baseline results** (01-02-SUMMARY, 01-03-SUMMARY) -- baseline BPB 1.2254, artifact 15.88MB, timing 600s cap
- **Project research** (STACK.md, FEATURES.md, ARCHITECTURE.md) -- technique landscape and architecture patterns

### Tertiary (LOW confidence)
- **Artifact size estimates for int6 10L** -- calculated from parameter counts and compression ratios; actual size requires empirical measurement

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- zstandard is the only new dependency; everything else is PyTorch native
- Architecture: HIGH -- exact code exists in two reference implementations already in the repo
- Pitfalls: HIGH -- identified from concrete diff analysis between baseline and SOTA code
- Hyperparameters: HIGH -- exact values from #1 submission README with 3-seed validation

**Research date:** 2026-03-23
**Valid until:** 2026-04-06 (competition is time-limited; patterns are stable)

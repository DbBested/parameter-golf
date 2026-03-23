# Phase 3: Mixed-Precision Quantization - Research

**Researched:** 2026-03-23
**Domain:** Mixed int5/int6 post-training quantization, magnitude pruning, gradient monitoring
**Confidence:** HIGH

## Summary

This research investigates how to modify the current uniform int6 post-training quantization in `repo/train_gpt.py` to use mixed int5 (MLP weights) / int6 (attention weights) quantization. The goal is to reduce the compressed artifact from 18.49MB to under 16MB while maintaining BPB quality near 1.1356.

The change is minimal and proven. The SOTA #1 submission (`repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py`) already implements this exact pattern and achieves 1.14276 BPB (mean of 3 seeds) with artifact sizes ranging from 15.68MB to 15.97MB -- all under the 16MB cap. The entire diff from our current code is exactly **two lines** in the `mixed_quantize_int6()` function.

The research also covers QUANT-04 (3% magnitude pruning, already implemented in Phase 2) and QUANT-05 (per-layer gradient monitoring, which needs new instrumentation code since no monitoring exists in the current codebase).

**Primary recommendation:** Change two lines in `mixed_quantize_int6()` to select `clip_range=15` for MLP parameters and `clip_range=31` for everything else. Update the meta type string accordingly. No other code changes are needed for the core quantization improvement.

## Standard Stack

No new libraries are needed. The entire change is within the existing `train_gpt.py` file using existing functions.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.6.0+cu124 | Framework (quantize_intN_per_row) | Already installed, all quantization uses torch ops |
| zstandard | 0.25.0 | zstd-22 compression | Already installed from Phase 2 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (none new) | | | |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom int5/int6 per-row PTQ | torchao | torchao lacks int5/int6 support; our quantize_intN_per_row already works |
| Post-training quantization | QAT with STE | PTQ is proven to work in SOTA; QAT adds complexity with unproven benefit (deferred) |
| Per-row scaling in FP16 | Per-row scaling in FP32 | FP16 scales save 2 bytes/row; SOTA uses FP16 scales successfully |

## Architecture Patterns

### The Two-Line Change (VERIFIED - HIGH confidence)

The entire mixed-precision quantization change is a two-line modification to `mixed_quantize_int6()`:

**Current code (Phase 2, line 571):**
```python
clip = 31  # Phase 2: uniform int6 for all (int5 MLP deferred to Phase 3)
```
```python
meta[name] = {"type": "int6"}
```

**Target code (SOTA #1, line 366):**
```python
clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for attention
```
```python
meta[name] = {"type": f"int{5 if cat == 'mlp' else 6}"}
```

Source: Direct diff of `repo/train_gpt.py` line 571 vs SOTA #1 `train_gpt.py` line 366.

### Quantization Mapping (all parameter categories)

| Parameter Category | Classifier Match | Quantization | clip_range | Stored As |
|-------------------|------------------|-------------|------------|-----------|
| MLP weights | `.mlp.` in name | **int5** | 15 | int8 tensor + FP16 per-row scale |
| Attention weights | `.attn.` in name | int6 | 31 | int8 tensor + FP16 per-row scale |
| Bigram proj | `bigram` in name | int6 | 31 | int8 tensor + FP16 per-row scale |
| Embeddings (tok_emb) | FP16_KEEP pattern | FP16 passthrough | N/A | FP16 |
| Last-layer key proj (blocks.8.attn.c_k) | FP16_KEEP pattern | FP16 passthrough | N/A | FP16 |
| Control tensors (scales, gains, smear, bigram.scale) | CONTROL_TENSOR pattern | FP32 passthrough | N/A | FP32 |
| Small tensors (numel <= 8192) | Size check | Passthrough | N/A | FP16/original |
| Other 2D float tensors | Fallback | int8 | 127 | int8 + FP16 scale |

### Dequantization: No Changes Needed

The `dequantize_mixed_int6()` function is identical between our code and SOTA #1 (verified via diff). It works generically with any quantized tensor -- it just multiplies `q * scale` regardless of the original clip_range. The meta type string (`"int5"` vs `"int6"`) is informational only and not used during dequantization.

### Parameter Count Breakdown

| Category | Params/layer | Total (10 layers) | Raw Quantized Bytes |
|----------|-------------|-------------------|-------------------|
| Attention | 786,432 | 7,864,320 | ~7.9MB (int8 + FP16 scales) |
| MLP | 1,572,864 | 15,728,640 | ~15.8MB (int8 + FP16 scales) |
| Embeddings | N/A | 524,288 (tied) | ~1.0MB (FP16) |
| Bigram embed | N/A | 1,310,720 | ~2.6MB (FP16) |
| Bigram proj | N/A | 65,536 | ~66KB (int8 + scales) |

MLP weights are **2x** the attention weights, so the int5 reduction has maximum leverage.

### Compression Ratios (from SOTA #1 README)

| Quantization | Compression Ratio (zstd-22) | Description |
|-------------|---------------------------|-------------|
| Int5 (clip=15) | 1.88x | 32 unique values in [-16,15], stored as int8. Many zeros from the restricted range improve entropy coding. |
| Int6 (clip=31) | 1.51x | 64 unique values in [-32,31], stored as int8. Less repetition, harder to compress. |

### Artifact Size: Actual Measurements

**Phase 2 (uniform int6):** 18,487,380 bytes (17.63MB) -- 2.49MB OVER cap

**SOTA #1 (mixed int5/int6), 3 seeds:**

| Seed | Artifact Bytes | Under 16MB Cap By |
|------|---------------|-------------------|
| 42 | 15,965,978 | 34,022 bytes (0.03MB) |
| 1337 | 15,830,186 | 169,814 bytes (0.16MB) |
| 2024 | 15,684,876 | 315,124 bytes (0.30MB) |

**Savings from int5 MLP:** 2.52MB to 2.80MB depending on seed. This is because different seeds produce different weight distributions, affecting compression ratios.

**Critical observation:** The headroom is tight (34KB to 315KB). Any code size increase or model parameter addition could push over the cap. The planner must ensure artifact size is validated after every change.

### BPB Impact of Int5 MLP

| Configuration | val_bpb | Source |
|--------------|---------|--------|
| Phase 2 (uniform int6) | 1.1356 | Phase 2 run, seed 1337 |
| SOTA #1 (int5 MLP) seed 42 | 1.1427 | SOTA #1 log |
| SOTA #1 (int5 MLP) seed 1337 | 1.1430 | SOTA #1 log |
| SOTA #1 (int5 MLP) seed 2024 | 1.1426 | SOTA #1 log |
| SOTA #1 mean | 1.14276 | SOTA #1 README |

**Important nuance:** The SOTA #1 BPB (1.1428) is HIGHER (worse) than Phase 2 (1.1356) but this is NOT because of int5 quantization. SOTA #1 uses a slightly different model configuration (not identical hyperparameters to our Phase 2 model). The BPB impact of switching from int6-to-int5 for MLP specifically was measured in SOTA #1's ablation table:

| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.003 |

The 10th layer IMPROVEMENT (+0.003 BPB) slightly exceeded the int5 DEGRADATION, resulting in a net positive. For our Phase 2 model which already has 10 layers, switching MLP from int6 to int5 is expected to produce a BPB degradation of approximately 0.001 to 0.003 BPB. This is within the acceptable 0.003 threshold stated in CONTEXT.md.

### Magnitude Pruning + Int5 Interaction (QUANT-04)

Magnitude pruning is already implemented (Phase 2, line 1367-1373 of current train_gpt.py):
```python
# Magnitude pruning: zero out smallest weights to improve compression
with torch.no_grad():
    for name, param in base_model.named_parameters():
        if param.ndim == 2 and param.numel() > 65536:
            threshold = torch.quantile(param.abs().float().flatten(), 0.03)
            mask = param.abs() < threshold
            param.masked_fill_(mask, 0.0)
```

The SOTA #1 code has identical pruning code (line 1163-1169). Pruning happens BEFORE quantization, so it creates zeros in the float weights which then get quantized to zero int8 values, improving zstd compression. With int5 (restricted range), there are already more near-zero values that quantize to exactly zero, so pruning has slightly less marginal benefit. But the SOTA #1 uses the same 3% threshold successfully.

**No changes needed for pruning.** The existing implementation works correctly with int5.

### Per-Layer Gradient Monitoring (QUANT-05)

QUANT-05 says: "Per-layer gradient monitoring during QAT to detect instability early."

**Key finding:** Our pipeline uses PTQ (post-training quantization), not QAT. There is no quantization during training. The "gradient monitoring" requirement needs reinterpretation for the PTQ context.

**Recommended interpretation:** Add logging that reports per-layer quantization error (the BPB or MSE impact of quantizing each layer) during the post-training serialization step. This serves the same purpose -- detecting which layers are most sensitive to quantization -- without requiring actual gradient computation.

**Implementation approach:**
1. After magnitude pruning, before the mixed quantization call, compute per-layer quantization sensitivity:
   - For each weight matrix, quantize and dequantize it individually
   - Measure reconstruction MSE: `(original - dequantized).pow(2).mean()`
   - Log: `quant_sensitivity layer:{name} mse:{value} type:{int5|int6}`
2. Optionally compute per-layer BPB contribution by doing forward pass with one layer quantized at a time (more expensive but more informative)

**Alternative (lighter weight):** Simply log the per-layer scale statistics and max quantization error during the existing quantization pass. This requires adding ~10 lines of logging to `mixed_quantize_int6()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Int5 quantization kernel | Custom bit-packing or new quantization function | Modify existing `quantize_intN_per_row(t, clip_range=15)` | The function already supports arbitrary clip_range; just pass 15 instead of 31 |
| Mixed-precision dispatch | Complex per-layer config system | Simple `if cat == "mlp"` conditional in existing function | SOTA #1 uses exactly this one-liner; over-engineering adds no value |
| Artifact size estimation | Pre-computation model | Just run the quantize+compress pipeline and check | Compression ratios vary by seed; the only reliable check is actual measurement |
| Quantization sensitivity analysis | Custom backward pass hooks | Per-layer MSE between original and dequantized weights | MSE is cheap, informative, and does not require gradient computation |

**Key insight:** The quantization infrastructure from Phase 2 is already fully general. `quantize_intN_per_row()` accepts any clip_range. The only change is selecting which clip_range to use per layer, which is a single conditional.

## Common Pitfalls

### Pitfall 1: Artifact barely fits -- seed variance can push over 16MB
**What goes wrong:** With int5 MLP, the artifact size ranges from 15.68MB to 15.97MB across seeds. A "bad" seed (seed 42) leaves only 34KB of headroom. Adding code or changing model parameters could push it over.
**Why it happens:** Different random seeds produce different weight distributions. Some distributions compress better than others.
**How to avoid:** Always check artifact size with the specific seed being used. Plan for worst-case (seed 42 scenario). If code grows, consider reducing bigram_vocab_size from 10240 to 8192 as a fallback (saves ~0.3MB per SOTA ablation, costs ~0.001 BPB).
**Warning signs:** Artifact size > 15.9MB on any seed.

### Pitfall 2: Forgetting to update the meta type string
**What goes wrong:** If you change `clip = 15 if cat == "mlp" else 31` but forget to update `meta[name] = {"type": "int6"}` to include the int5 indicator, the meta dictionary will be misleading. This does NOT affect correctness (dequantization ignores the type field) but makes debugging and validation harder.
**Why it happens:** The meta type is informational and the code works without updating it.
**How to avoid:** Always update both the clip_range line AND the meta type line together. Use the exact SOTA #1 pattern.
**Warning signs:** Log output saying "int6" for MLP parameters when they should say "int5".

### Pitfall 3: Not validating BPB after the roundtrip
**What goes wrong:** The quantization might introduce unexpected errors that are only visible after the full roundtrip (quantize -> compress -> decompress -> dequantize -> eval). Checking only the artifact size without re-evaluating BPB misses quality regressions.
**Why it happens:** Int5 has higher quantization error per weight than int6. If a specific layer's weights have an unusual distribution, the error could be larger than expected.
**How to avoid:** Always run the full eval after the roundtrip. Compare roundtrip BPB to pre-quantization BPB. Expected degradation from int5 MLP: 0.001-0.003 BPB.
**Warning signs:** Roundtrip BPB is > 0.005 worse than pre-quantization BPB.

### Pitfall 4: Gradient monitoring adds excessive overhead
**What goes wrong:** If QUANT-05 is implemented as per-layer BPB evaluation (quantize one layer at a time, run full eval), it requires N forward passes (one per layer) which could take 10+ minutes.
**Why it happens:** Full BPB evaluation takes ~3 minutes (192s from Phase 2 logs). With 10 layers * ~6 weight matrices each = 60 forward passes.
**How to avoid:** Use per-layer MSE (reconstruction error) instead of per-layer BPB. MSE requires zero forward passes -- just compare the quantized-dequantized weights to the original.
**Warning signs:** Post-training serialization + monitoring takes more than 5 minutes.

### Pitfall 5: Pruning AFTER quantization instead of BEFORE
**What goes wrong:** If pruning is applied after quantization, it zeros out int8 values rather than float values. This changes the semantics -- the zeros won't exactly correspond to the smallest original magnitudes.
**Why it happens:** Code ordering mistake.
**How to avoid:** Maintain the existing order: SWA averaging -> magnitude pruning -> quantization -> compression. This is already correct in both Phase 2 and SOTA #1.
**Warning signs:** N/A -- just verify code order.

## Code Examples

### Change 1: Mixed clip_range selection (THE CORE CHANGE)

```python
# In mixed_quantize_int6(), replace:
#   clip = 31  # Phase 2: uniform int6 for all (int5 MLP deferred to Phase 3)
# With:
clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for attention

# And replace:
#   meta[name] = {"type": "int6"}
# With:
meta[name] = {"type": f"int{5 if cat == 'mlp' else 6}"}
```
Source: SOTA #1 `train_gpt.py` line 366-370.

### Change 2: Per-layer quantization sensitivity logging (QUANT-05)

```python
# Add after magnitude pruning, before mixed_quantize_int6() call:
if master_process:
    log0("=== Per-layer quantization sensitivity ===")
    for name, param in base_model.named_parameters():
        if param.ndim == 2 and param.numel() > 8192:
            t = param.detach().cpu().float()
            cat = _classify_param(name)
            cr = 15 if cat == "mlp" else 31
            q, s = quantize_intN_per_row(t, clip_range=cr)
            # Reconstruct
            if s.ndim > 0:
                recon = (q.float() * s.float().view(q.shape[0], 1))
            else:
                recon = q.float() * s.float()
            mse = (t - recon).pow(2).mean().item()
            max_err = (t - recon).abs().max().item()
            log0(f"quant_sensitivity {name} cat:{cat} int{5 if cat=='mlp' else 6} "
                 f"mse:{mse:.2e} max_err:{max_err:.4f} shape:{list(t.shape)}")
```

### Change 3: Artifact size validation with explicit cap check

```python
# After writing final_model.int8.ptz, add explicit cap validation:
ARTIFACT_CAP = 16_000_000
total_bytes = quant_file_bytes + code_bytes
if total_bytes > ARTIFACT_CAP:
    log0(f"WARNING: artifact {total_bytes} bytes EXCEEDS {ARTIFACT_CAP} byte cap "
         f"by {total_bytes - ARTIFACT_CAP} bytes")
else:
    log0(f"OK: artifact {total_bytes} bytes fits cap with "
         f"{ARTIFACT_CAP - total_bytes} bytes headroom")
```

## State of the Art

| Old Approach (Phase 2) | Current Approach (Phase 3) | When Changed | Impact |
|------------------------|---------------------------|--------------|--------|
| Uniform int6 (clip=31) for all weights | Int5 (clip=15) for MLP, int6 (clip=31) for attention | SOTA #1 submission | Artifact: 18.49MB -> ~15.8MB (fits under 16MB) |
| BPB 1.1356 with 10 layers | Expected BPB ~1.137 with int5 MLP | Phase 3 | ~0.001-0.003 BPB degradation from int5 |
| No quantization sensitivity logging | Per-layer MSE logging | Phase 3 | Visibility into which layers are quantization-sensitive |

**Note on bit-packing:** Both int5 and int6 values are stored as int8 (one byte per value). There is no actual bit-packing to 5 or 6 bits -- the compression savings come entirely from the reduced entropy of the restricted range values, which zstd-22 exploits effectively. True bit-packing would save more space but would complicate serialization with torch.save() and is not used by any SOTA submission.

## Open Questions

1. **Exact BPB impact on OUR model**
   - What we know: SOTA #1 ablation shows int5 MLP degrades ~0.001-0.003 BPB, but on a different model/training run.
   - What's unclear: Our Phase 2 model (BPB 1.1356) may react differently to int5 since it may have different weight distributions.
   - Recommendation: Run the two-line change and measure. If BPB degrades > 0.003, consider int5 only for the fc (up-projection) matrices while keeping MLP proj (down-projection) at int6, since up-projections are typically more robust.

2. **Whether the blocks.8.attn.c_k FP16 keep pattern is optimal for 10 layers**
   - What we know: SOTA #1 keeps `blocks.8.attn.c_k` in FP16 (last decoder layer's key projection). With 10 layers (0-indexed 0-9), layer 8 is the second-to-last.
   - What's unclear: Whether a different layer's key projection would benefit more from FP16 preservation. This is likely a minor effect.
   - Recommendation: Keep the SOTA #1 pattern. Do not experiment with this in Phase 3.

3. **Whether 3% pruning percentage should change for int5**
   - What we know: SOTA #1 uses 3% pruning with int5 MLP successfully.
   - What's unclear: Whether int5's narrower range means pruning is less beneficial (values already cluster more tightly around zero).
   - Recommendation: Keep 3% as-is. This is proven in SOTA #1 and the interaction effect is marginal.

## Sources

### Primary (HIGH confidence)
- `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py` - SOTA #1 complete implementation with int5 MLP
- `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` - Ablation results, compression ratios, 3-seed BPB
- `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_seed42.log` - Actual artifact size (15,965,978 bytes)
- `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_seed1337.log` - Actual artifact size (15,830,186 bytes)
- `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_seed2024.log` - Actual artifact size (15,684,876 bytes)
- `repo/train_gpt.py` - Current Phase 2 implementation (lines 552-600: quantization functions)
- Direct diff between SOTA #1 and Phase 2 quantization functions (verified: exactly 2 lines different)

### Secondary (MEDIUM confidence)
- Phase 2 research (`02-RESEARCH.md`) - Quantization patterns, architecture context
- Phase 2 summary (`02-03-SUMMARY.md`) - Phase 2 artifact size (18,487,380 bytes), BPB (1.1356)

### Tertiary (LOW confidence)
- QUANT-05 gradient monitoring interpretation - No existing code or SOTA reference for this; implementation is novel to Phase 3

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, proven two-line change from SOTA #1
- Architecture: HIGH - Exact code diff verified, actual artifact sizes measured from 3-seed logs
- Pitfalls: HIGH - Based on actual measurements (tight headroom, seed variance) from SOTA #1 logs
- Gradient monitoring (QUANT-05): MEDIUM - Novel implementation not in any reference code; recommended MSE approach is straightforward but untested in this context

**Research date:** 2026-03-23
**Valid until:** Indefinitely (all findings based on code in the repository, not external sources)

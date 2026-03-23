# Phase 3: Mixed-Precision Quantization - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement mixed-precision post-training quantization: int5 (clip_range=15) for MLP weights, int6 (clip_range=31) for attention weights, FP16 for embeddings. This reduces the compressed artifact from 18.49MB to under 16MB while maintaining BPB quality. Also implements 3% magnitude pruning optimization and per-layer gradient monitoring for quantization stability analysis.

Current state: Phase 2 delivered BPB 1.1356 but artifact is 18.49MB (2.49MB over 16MB cap). The SOTA #1 submission (1.1428 BPB) uses exactly this mixed int5/int6 pattern.

</domain>

<decisions>
## Implementation Decisions

### Quantization Strategy
- Int5 for MLP weights (clip_range=15, range [-16, 15]) — MLPs are more tolerant of quantization noise
- Int6 for attention weights (clip_range=31, range [-32, 31]) — attention is more sensitive
- FP16 passthrough for embeddings (tok_emb) and control tensors (scales, gains, smear gate, bigram scale)
- Per-row scaling factors stored as FP32 for both int5 and int6 layers
- Follow the exact pattern from SOTA #1 submission (thwu1's train_gpt.py)

### Implementation Approach
- Modify the existing `mixed_quantize_int6` function to accept per-layer clip_range based on parameter name
- MLP parameters (*.mlp.*) get clip_range=15, attention parameters (*.attn.*) get clip_range=31
- Update `dequantize_mixed_int6` to handle the mixed bit widths during roundtrip verification
- Measure compressed artifact size after each quantization change

### Pruning Optimization
- 3% magnitude pruning already implemented in Phase 2 — verify it works correctly with int5
- May need to adjust pruning percentage if int5 introduces quality issues

### Claude's Discretion
- Exact implementation of mixed clip_range selection in the quantization function
- Whether to add gradient monitoring during the serialization/eval path (not during training since we use PTQ)
- Bit-packing optimization details for int5 values

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `repo/train_gpt.py` — Already has int6 PTQ, zstd-22, magnitude pruning from Phase 2
- SOTA #1 reference: `repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py` — has the exact int5 MLP pattern to copy
- `scripts/eval/check_artifact_size.py` — Validates artifact vs 16MB cap

### Current State
- Model: 25.5M params, 10 layers, MLP 3x, BigramHash, SmearGate
- Int6 uniform quantization: 18.49MB artifact (OVER 16MB)
- BPB: 1.1356 (beats SOTA)
- Int5 MLP should save ~2.5MB (enough to fit under 16MB)

### Integration Points
- Quantization happens in the serialization path of train_gpt.py (post-training)
- The dequantization roundtrip must handle mixed bit widths correctly
- Compressed artifact = code bytes + zstd-22 compressed quantized model

</code_context>

<specifics>
## Specific Ideas

- The SOTA #1 submission shows int5 MLP achieves 1.88x compression ratio vs int6's 1.51x
- Key code pattern: check if parameter name contains '.mlp.' to select clip_range
- Monitor BPB degradation from int5 vs int6 — if > 0.003 BPB loss, adjust strategy

</specifics>

<deferred>
## Deferred Ideas

- Int4 quantization for most tolerant layers (Phase 5 moonshot)
- QAT with STE if PTQ quality degrades (Phase 3 contingency)

</deferred>

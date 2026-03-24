# 11L VRL + XSA-11 + Full GPTQ + LeakyReLU² + Score-First TTT

**3-seed mean val_bpb = FILL_IN (std FILL_IN)**

| Seed | val_bpb | Artifact Size |
|------|---------|---------------|
| 1337 | FILL_IN | FILL_IN |
| 42   | FILL_IN | FILL_IN |
| 7    | FILL_IN | FILL_IN |

Hardware: 8xH100 SXM, 600s training wall clock.

## Base

Built on merged SOTA #1 (signalrush, 1.1228 BPB, PR #414).

## Our Contributions

### 1. Full GPTQ (Hessian-Calibrated Quantization)
Replaces GPTQ-lite (per-row percentile clip search) with full Hessian-aware GPTQ following the IST-DASLab algorithm. Collects H=X^TX on 128 calibration batches, computes upper Cholesky factor of H^{-1}, then quantizes column-wise with block-128 error compensation and activation-order sorting (descending Hessian diagonal). 3% magnitude pruning post-GPTQ for artifact size compliance.

**Impact: ~-0.004 BPB from better quantization alone.**

### 2. Value Residual Learning (VRL)
Stores layer-0's attention V output and blends it into all subsequent layers (1-10) via per-layer learned sigmoid gates initialized at sigmoid(-1.5) ≈ 0.18. Preserves initial token information through depth. Only 10 additional scalar parameters.

**Impact: ~-0.001 BPB.**

### 3. XSA on All 11 Layers
Extended Exclusive Self Attention from the last 4 layers to all 11 layers. XSA subtracts the self-value component from attention output, forcing heads to model only contextual information.

**Impact: ~-0.0005 BPB.**

### 4. QAT-Export Alignment
Changed STE fake-quantization clipping from row-max (100th percentile) to quantile(0.9995) to match the export quantizer's percentile search. Ensures training-time quantization noise matches inference-time quantization.

**Impact: ~-0.0005 BPB.**

### 5. LeakyReLU(0.5)²
Replaces relu² with leaky_relu(x, 0.5)² in MLP layers. Prevents dead neurons, doubles effective MLP capacity.

**Impact: ~-0.0015 BPB.**

### 6. Score-First TTT (Test-Time Training)
Legal score-first adaptation during evaluation:
- 131K-token chunks, 3 epochs per chunk
- AdamW lr=1e-4, weight_decay=0.0, cosine LR decay
- Unfreezes last 2 blocks + norms + lm_head (~5.2M / 27M params)
- Post-TTT temperature calibration at T=0.98
- All tokens scored BEFORE any gradient update using them

**Impact: ~-0.003 BPB.**

## Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- MLP 3x (1536 hidden) with LeakyReLU(0.5)²
- SmearGate + BigramHash(2048) + U-Net skip connections
- Value Embeddings (VE128) on layers 9-10
- Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer+1))
- OrthoInit + logit softcap(30.0) + tied embeddings

## Training

- Muon (lr=0.025, momentum=0.99, WD=0.04) for 2D weights
- AdamW for embeddings/scalars
- EMA(0.997) weight averaging
- Late QAT at scale < 0.15 with quantile(0.9995) clipping
- Wallclock-based warmdown (3500 iterations)
- FlashAttention-3 via flash_attn_interface

## Quantization

- Int6 for attention + MLP weights (Full GPTQ with Hessian error compensation)
- Int8 for embeddings
- FP32 for control tensors (scales, gates, norms)
- 3% magnitude pruning post-quantization
- zstd level 22 compression

# Phase 6: Next-Gen Stack + Submission - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the complete next-generation technique stack that emerged from unmerged competition PRs (March 20-23), pushing BPB from 1.1421 toward < 1.12. Then prepare and validate the submission package.

The competition has advanced dramatically past the merged SOTA (1.14276). Unmerged PRs show:
- 1.0865 (EMA + LoRA TTT rank-8)
- 1.0916 (Shared Sparse Sidecar + EMA + TTT)
- 1.1179-1.1204 (GPTQ + Legal TTT + LeakyReLU(0.5)^2)

Our current model: BPB 1.1421, artifact 15.82MB, functionally identical to merged SOTA.

</domain>

<decisions>
## Implementation Decisions

### Wave 1: Quick Architecture Wins (modify train_gpt.py in-place)
1. **LeakyReLU(0.5)^2** — Replace relu(x).square() with leaky_relu(x, 0.5).square() in MLP forward. Expected -0.003 BPB. 1 line.
2. **Reduce BigramHash 10240→4096** — Free ~1.5MB for 11th layer. All new submissions use 2048-4096 buckets.
3. **Add 11th layer** — num_layers=9→11 with the freed artifact budget. Expected -0.003 to -0.005 BPB.
4. **Partial RoPE 16/64** — Apply RoPE to only 16 of 64 head dimensions. Expected -0.001 BPP. ~5 lines.
5. **Revert pruning from 5% to 3%** — Match the leader's proven optimal.

### Wave 2: Training Improvements (modify train_gpt.py)
6. **EMA(0.997)** — Replace SWA with EMA from step 0. Maintain CPU copy, update every step. Expected -0.002 to -0.004 BPB. ~20 lines.
7. **Late QAT** — Apply STE fake-quantization during last 10-15% of training (scale < 0.15). Expected -0.001 to -0.003 BPP. ~15 lines.
8. **GPTQ-lite** — At serialization, try 5 clip percentiles per row, pick lowest MSE. Expected -0.002 BPB. ~10 lines.
9. **LN Scale** — Scale each layer output by 1/sqrt(layer_idx+1). Trivial.

### Wave 3: Advanced Architecture (optional, if time permits)
10. **XSA** (Cross-layer Shared Attention) on last 4 layers — Remove redundant attention info. ~20 lines.
11. **Value Embeddings** (VE128) on last 2 layers — Additional position-aware info. ~15 lines.

### Wave 4: Legal TTT (the big gun)
12. **Score-first-then-train TTT** — After standard eval, do chunked TTT with full-weight SGD/AdamW on most blocks. Expected -0.005 to -0.020 BPP.
    - Chunk size: 32K-131K tokens
    - For each chunk: SCORE first (inference mode), then TRAIN (2-30 epochs, lr=0.002)
    - Freeze first 2 blocks, unfreeze rest
    - Reset between documents (BOS detection)
    - Must complete within 10-minute eval budget

### Wave 5: Submission Hardening
13. Run 3-seed validation
14. Package submission: README.md, submission.json, train logs, train_gpt.py
15. Verify artifact < 16MB, BPB statistically significant

### Claude's Discretion
- Exact LR for TTT (0.001-0.01 range)
- TTT chunk size (32K-131K)
- Number of TTT epochs per chunk (2-30)
- Whether to include XSA and VE128 (skip if artifact too tight)
- Exact BigramHash bucket count (2048 vs 4096)

</decisions>

<code_context>
## Existing Code Insights

### Base Model
- repo/train_gpt.py — 54KB, 1259 lines, fully working SOTA implementation
- BPP: 1.1421, artifact 15.82MB

### Reference Implementations to Study
- repo/records/ — all merged submissions
- Unmerged PR code accessible via GitHub API

### Compute
- H200 nodes: node4300, node4301 (8xH200 each)
- Each training run: ~10-15 min total
- Need ~5-10 training runs for this phase

</code_context>

<specifics>
## Specific Ideas

- Study PR #374 (unnir, 1.1246) and PR #529 (EthanYangTW, 1.1195) for the exact implementation patterns
- LeakyReLU(0.5)^2 is confirmed by 4+ independent submissions — highest confidence quick win
- EMA(0.997) from step 0 is strictly better than SWA for this training regime
- GPTQ-lite: try quantile [0.999, 0.9995, 0.9999, 0.99999, 1.0] per row, pick min MSE
- Legal TTT: the score-first protocol must be bulletproof — any data leakage = disqualification

</specifics>

<deferred>
## Deferred Ideas

- Shared Sparse Sidecar (too complex for time remaining)
- Arithmetic coding compression (marginal gains)
- Parameter banking / Parallel Muon (systems optimization)
- Non-uniform bit allocation across layers

</deferred>

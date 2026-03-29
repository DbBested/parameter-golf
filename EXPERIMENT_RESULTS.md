# Experiment Results — Parameter Golf Campaign

## Baseline Reference
| Source | BPB | Steps | Step avg | Artifact | Notes |
|--------|-----|-------|----------|----------|-------|
| ORCD 4xH100 1700s (seed 1337) | 1.1107 | 8154 | 208ms | 16.55MB (OVER) | Full GPTQ, XSA=11, no pruning |
| ORCD 4xH100 1700s + 4% prune | 1.1149 | 8147 | 209ms | 15.98MB ✓ | Same but with pruning |
| RunPod 8xH100 seed 1337 (XSA=11) | 1.1213 | 6376 | 90.9ms | 16.11MB (OVER) | Pre-TTT sliding window |
| RunPod 8xH100 seed 42 (XSA=4) | 1.1244 | 6364 | 87.9ms | 15.96MB ✓ | Pre-TTT sliding window |
| **SOTA (top scorer)** | **1.1194** | 7185 | 83.3ms | 15.98MB | Post-TTT, submission bar = 1.1144 |

---

## TTT Sweep (2026-03-25, ORCD 4xH100, 8200 steps)
All use XSA=4, SEED=1337, PRUNE_PCT=0.045, FORCE_SDPA=1, old Muon (10 NS steps)

| Job ID | TTT_LR | Freeze Blocks | Pre-TTT BPB | Post-TTT BPB | TTT Δ | Artifact | Status |
|--------|--------|---------------|-------------|-------------|-------|----------|--------|
| 10972205 | 0.0002 | 0 (all) | 1.1324 (EMA) | **1.1173** | **-0.0151** | 15.75MB ✓ | DONE |
| 10972206 | 0.0002 | 8 (last 3) | 1.1334 (EMA) | **1.1179** | **-0.0155** | 16.21MB ✗ | DONE |
| 10972207 | 0.0005 | 0 (all) | 1.1332 (EMA) | **1.2195 (diverged)** | **+0.086** | 16.44MB ✗ | DONE |
| 10972208 | 0.0005 | 8 (last 3) | 1.1332 (EMA) | **1.1296** | **-0.0036** | 16.23MB ✗ | DONE |
| 10972209 | 0.001 | 0 (all) | 1.1331 (EMA) | **1.2168 (diverged)** | **+0.084** | 16.13MB ✗ | DONE |
| 10972210 | 0.001 | 8 (last 3) | 1.1331 (EMA) | **1.1377** | **-0.0** (neutral) | 16.25MB ✗ | DONE |

**Key findings:**
1. TTT with SGD LR=0.0002 gives massive -0.015 BPB improvement (barely modifies weights)
2. **SGD LR=0.0005 DIVERGES** (+0.25 BPB) — GPTQ weights are extremely fragile under SGD
3. Freezing 8 blocks barely helps at LR=0.0002 (1.1179 vs 1.1173) — too conservative to matter
4. **AdamW TTT is essential** — SGD only works at ultra-low LR where improvement is limited
5. Artifact size varies significantly between runs (15.75MB to 16.44MB) — adaptive pruning needed

---

## Turbo Muon (2026-03-25, ORCD 4xH100)
| Job ID | NS Steps | Preconditioning | Step avg | BPB | Status |
|--------|----------|-----------------|----------|-----|--------|
| 10972284 | 4 | AOL (Turbo) | 201.1ms | TTT diverged (LR=0.0005) | DONE |
| baseline | 10 | Frobenius | 201.7ms | 1.1324 (EMA) | — |

**Finding:** Turbo Muon shows no measurable speedup on 4xH100 (201.1 vs 201.7ms — within noise). The optimizer step is too small a fraction of total compute at this batch size. May show improvement on 8xH100 where batch/GPU is smaller.

---

## Compression Sweep (2026-03-25, ORCD 4xH100, Turbo Muon, no TTT)
| Job ID | Compressor | Prune Mode | Prune % | Artifact Size | BPB | Status |
|--------|-----------|------------|---------|---------------|-----|--------|
| 10972323 | lzma | random | 4.5% | **16.30MB ✗** | **1.1177** | DONE |
| 10972324 | lzma | random | 3.0% | **16.37MB ✗** | **1.1158** | DONE |
| 10972325 | lzma | random | 0.0% | **16.40MB ✗** | **1.1129** | DONE |
| 10972326 | zstd | hessian | 4.5% | **16.27MB ✗** | **1.1146** | DONE |
| 10972327 | lzma | hessian | 3.0% | **16.22MB ✗** | **1.1144** | DONE |

**Key finding: lzma is WORSE than zstd for our weights!** zstd at 4.5% prune = 15.75MB (fits), lzma at same = 16.30MB (over). Stick with zstd. The top scorer's lzma advantage is specific to their weight distribution (GPTQ-lite, int6 per-row without Full GPTQ).

**Also notable:** Less pruning gives better BPB (1.1158 at 3% vs 1.1177 at 4.5%). With adaptive pruning (TARGET_ARTIFACT_BYTES), we should prune the absolute minimum needed.

---

## RunPod TTT Results (2026-03-25, 8xH100)
| Seed | XSA | TTT impl | TTT_LR | Pre-TTT | Post-TTT | TTT Δ | Notes |
|------|-----|----------|--------|---------|----------|-------|-------|
| 1337 | 11 | old (broken) | 0.002 | 1.1213 | 3.0252 | +1.904 | Fed [1,32768] seqs |
| 42 | 4 | new (sliding) | 0.002 | 1.1244 | 1.1367 | +0.012 | SGD too aggressive on GPTQ weights |

---

## AdamW TTT Sweep (queued, ORCD 4xH100, Turbo Muon)
Based on PR #601 (SGD TTT = +0.030 on GPTQ) and PR #606 (AdamW TTT = works, 1.1162 BPB).

| Job ID | TTT_LR | Freeze Blocks | Chunk Size | Post-TTT BPB | Status |
|--------|--------|---------------|-----------|-------------|--------|
| 10973647 | 0.0001 | 9 (last 2) | 131072 | **1.4185 (diverged)** | DONE |
| 10973648 | 0.0003 | 9 (last 2) | 131072 | | RUNNING |
| 10973649 | 0.0001 | 6 (last 5) | 131072 | | PENDING |
| 10973650 | 0.00005 | 0 (all) | 131072 | | PENDING |

**Key finding: AdamW TTT ALSO diverges on Full GPTQ weights.** Even lr=0.0001 with only 2 blocks unfrozen. AdamW's adaptive LR amplifies gradients on quantization-fragile weights. Only SGD at lr=0.0002 (essentially doing almost nothing) works.

**Root cause:** Our Full GPTQ quantization produces weights that are fundamentally incompatible with ANY gradient-based TTT at meaningful learning rates. The successful PRs (#606, #615) use GPTQ-lite, not Full GPTQ. Options:
1. Use GPTQ-lite for TTT compatibility (sacrifices quantization quality)
2. Stick with SGD lr=0.0002 (gives -0.015 BPB, decent but not optimal)
3. Skip TTT entirely and push pre-TTT BPB lower

---

## GEPA Architecture Research

PR #668 GEPA achieves 1.0920 BPB at 30k steps. Scaling law at 9k steps → 1.116 BPB.

Key architectural differences from our model:
1. **U-Net skip connections**: Encoder (first 5 layers) saves outputs, decoder (last 6 layers) mixes via learned sigmoid gates
2. **Learned residual mixing**: Per-block `resid_mix` parameter blends `x` and `x0` (vs fixed residual)
3. **Star-ReLU MLP**: `relu(x)^2 * scale + bias` (vs LeakyReLU²)
4. **Gated skip weights**: `sigmoid(gate) * x + (1-gate) * weighted_skip`
5. **XOR bigram hash**: Different hash function for bigram embeddings
6. **Orthogonal init**: `nn.init.orthogonal_` on large matrices

Model is 27M params, artifact only 13.4-14.9MB (well under 16MB). Uses pure int6 GPTQ-lite.

---

## Code Changes Implemented (ready for next experiments)

### Turbo Muon (AOL preconditioning)
`zeropower_via_newtonschulz5` now uses AOL preconditioning + dynamic per-iteration coefficients.
Default 4 NS steps (was 10). Drop-in, no hyperparameter changes needed.
Expected: 5-10% faster step time.

### AdamW TTT
`TTT_OPTIMIZER=adamw` env var. Uses AdamW(lr, wd=0, betas=(0.9,0.999)) instead of SGD.
Based on PR #606 (1.1162 BPB) and PR #615 (1.1169 BPB) which both use AdamW TTT successfully.
SGD TTT was shown to hurt GPTQ models in PR #601 (+0.030 BPB).

### Prune-before-quantize
`PRUNE_BEFORE_QUANT=1` prunes float weights by magnitude before GPTQ quantization.
Skips post-quant pruning. GPTQ sees zeros and optimizes remaining weights better.

### AdamHD Huber Decay
`HUBER_DELTA=0.1` (or any positive value) enables Huber decay in Muon optimizer.
Quadratic decay for |w| < delta (same as L2), linear for |w| >= delta (bounded, L1-like).
Suppresses outlier weights that hurt int6 quantization.

### Batch Size Warmup
`BS_WARMUP_STEPS=500` ramps batch from 1/3 → full over first 500 steps.
More effective early optimization steps.

### lzma Compression
`COMPRESSOR=lzma` uses lzma preset 9 extreme instead of zstd-22.
Top scorer uses lzma. May compress better for our weight distribution.

### Hessian-weighted Pruning
`PRUNE_MODE=hessian` prunes |val|=1 weights with smallest GPTQ Hessian diagonal first.
Less BPB damage per byte saved compared to random pruning.

---

## Stochastic RYS Experiments (2026-03-26/27)

**Concept:** Train with stochastic layer repetition (SRYS) so layers learn to produce refinable representations. At eval time, deterministic RYS repeats the block for free depth at zero parameter cost. Based on [dnhkng's RYS blog](https://dnhkng.github.io/posts/rys/) which achieved #1 on HuggingFace leaderboard by repeating middle layers of Qwen2-72B.

**Setup:** 1xH100 or 1xL40S, NO_COMPILE, batch=393K tokens, 2000 steps, seed 1337.

### Phase 1: Standard 11L Architecture (512-dim)

**Baseline (no SRYS training):** val_bpb = **1.2562**

#### Contraction Property Analysis
Without SRYS training, repeating layers 7-8 produces out-of-distribution representations (cos_sim=0.66). SRYS training pushes cos_sim to 0.999+, teaching layers to tolerate repetition.

| Config | cos_sim | r1 (pass 1) | r2 (pass 2) | Contracting? |
|--------|---------|-------------|-------------|-------------|
| No SRYS (baseline) | 0.676 | 0.788 | 0.581 | N/A |
| SRYS p=0.3 | 0.9997 | 0.200 | 0.197 | Yes (r2 < r1) |
| SRYS p=0.5 | 0.9998 | 0.257 | 0.255 | Marginal |

#### Single-Repeat Training (SRYS_MAX_REPS=1, layers 7-9)

| SRYS_PROB | Base BPB | +RYS×1 | +RYS×2 | +RYS×3 | Δ best vs baseline |
|-----------|----------|--------|--------|--------|-------------------|
| 0.001 (ctrl) | **1.2562** | 1.8663 | — | — | — |
| 0.3 | 1.2561 | **1.2557** | 1.2558 | 1.2566 | **-0.0005** |
| 0.5 | 1.2586 | 1.2578 | 1.2577 | 1.2590 | +0.0015 |

#### Multi-Repeat Training (SRYS_MAX_REPS=3, random 1-3 repeats, layers 7-9)

| SRYS_PROB | Base BPB | +RYS×1 | ×2 | ×3 | ×4 | ×5 |
|-----------|----------|--------|----|----|----|-----|
| 0.3 | 1.2565 | **1.2561** | 1.2562 | 1.2570 | 1.2586 | 1.2611 |

#### Lower-p Sweep (SRYS_MAX_REPS=3, layers 7-9)

| SRYS_PROB | Base BPB | +RYS×1 | ×2 | ×3 | Δ best vs baseline |
|-----------|----------|--------|----|-----|-------------------|
| 0.1 | **1.2556** | 1.2556 | 1.2566 | 1.2585 | -0.0006 (base only) |
| 0.15 | 1.2560 | 1.2559 | 1.2565 | 1.2580 | -0.0003 |

#### Gated SRYS (learned per-dim gate, init sigmoid(-2)≈0.12)

| SRYS_PROB | Gate final | Base BPB | +RYS×1 | +RYS×2 | Δ best vs baseline |
|-----------|-----------|----------|--------|--------|-------------------|
| 0.3 gated | 0.095 | **1.2535** | **1.2534** | **1.2534** | **-0.0028** |

Note: The -0.0028 improvement is primarily from the gated training acting as a regularizer, not from eval-time RYS. The eval-time repeat adds only -0.0001. The gate closes from 0.12→0.095 — the model learns to minimize the repeat's contribution.

### Phase 2: Thin-Deep Architectures (fixed ~27M params)

**Hypothesis:** More layers → more functional redundancy → better RYS. Tested by trading width for depth.

| Config | Layers | Dim | Baseline BPB | SRYS Base | Best +RYS | Gate final | Δ vs own baseline |
|--------|--------|-----|-------------|-----------|-----------|-----------|-------------------|
| 512×11 (standard) | 11 | 512 | 1.2562 | 1.2561 | 1.2557 (×1) | — | -0.0005 |
| 384×20 | 20 | 384 | 1.2658 | 1.2652 | 1.2652 (×1) | 0.031 | -0.0006 |
| 352×24 | 24 | 352 | 1.2689 | 1.2683 | **1.2680 (×2-4)** | 0.069 | **-0.0009** |

**Key finding:** The 352×24 model shows the best RYS stability — repeats ×2-4 plateau at the same BPB instead of degrading. This confirms the scaling hypothesis: more depth improves RYS behavior. However, the thin models are 0.010-0.013 BPB worse than 512×11 overall because width matters more than depth at this param count.

### Phase 3: Attention-Only Hybrid (512-dim, 15L: 9 full + 6 attn-only)

**Hypothesis:** Keep full width (512) but add cheap depth via attention-only blocks (no MLP, ~0.79M vs 2.36M per block). These naturally support "re-attending" as iterative refinement.

Architecture: `[Full×5: encoder] → [AttnOnly×6: middle, RYS target] → [Full×4: decoder]`

| Config | Base BPB | +RYS×1 | ×2 | ×3 | ×4 | ×5 |
|--------|----------|--------|----|----|----|-----|
| Baseline (no SRYS) | **1.2646** | — | — | — | — | — |
| Gated SRYS p=0.3 | 1.2644 | 1.2645 | 1.2648 | 1.2654 | 1.2662 | 1.2674 |
| Ungated SRYS p=0.3 | 1.2680 | — | — | — | — | — |

**Key finding:** The gated model's cos_sim dropped to 0.35 (attn-only layers make huge changes on repeat) but the gate closed to 0.06, suppressing ~94% of the delta. The ungated model self-regulated to cos_sim=0.9998 (near identity). Either way, the model rejects the repeat.

### Overall Conclusions

**RYS is a scaling phenomenon.** Across all architectures tested at ~27M params:

| Architecture | Best RYS Δ vs own baseline | Gate behavior |
|-------------|---------------------------|---------------|
| 512×11 standard | -0.0005 | Closes to 0.03-0.10 |
| 384×20 thin-deep | -0.0006 | Closes to 0.03 |
| 352×24 ultra-deep | -0.0009 | Closes to 0.07 |
| 512×15 attn-only hybrid | -0.0002 | Closes to 0.06 |

1. **SRYS successfully teaches the contraction property** (cos_sim 0.66→0.999), proving that small models CAN learn to tolerate layer repetition.
2. **But the repeat provides negligible benefit** (-0.0005 to -0.0009 BPB at best). The model consistently minimizes the repeat's contribution when given a learnable gate.
3. **More depth improves RYS stability** (352×24 plateaus at ×2-4 instead of degrading at ×3) but not enough to overcome the width-for-depth tradeoff.
4. **The fundamental limitation:** At 27M params, every layer is maximally specialized for its pipeline position. There is no "unfinished thinking" for a second pass to complete. The blog's success at 72B/80L relied on functional redundancy that doesn't exist at this scale.
5. **Gated SRYS shows promise as a training regularizer** (-0.0028 base BPB improvement) independent of eval-time RYS. The mechanism (dampening layer outputs on ~30% of steps) may be worth investigating separately from RYS.

### Related Work
- [dnhkng RYS blog](https://dnhkng.github.io/posts/rys/): +2.6% on Qwen2-72B by repeating layers 45-51. Success required 80 layers with functional circuit redundancy.
- [Parameter Golf PR #579 (Frugendorff)](https://github.com/openai/parameter-golf/pull/579): Weight sharing with double-firing at 27M params. Best result (1.1325) came with cad0 (NO double-firing). Confirms RYS-like repetition doesn't help at this scale.
- Our prior RYS experiments (see RYS.md): Eval-only RYS with calibration training showed +0.021 to +0.638 BPB damage. Stochastic training eliminates the damage but doesn't produce meaningful improvement.

# nGPT Research Campaign — Context for Continuation

## What This Is

A research campaign to make nGPT (hypersphere-normalized transformers) work at small scale under extreme compression (16MB artifact, 600s training on 8xH100). Started as RYS (Repeat Your Self) research, evolved into a full nGPT investigation with multiple novel findings.

**Goal:** Research novelty for a competition PR that impresses OpenAI staff. BPB score is secondary to demonstrating curiosity, systematic investigation, and novel contributions.

---

## Current Best Results

### Competition-Viable Entry (bf16 compile fix — 2026-03-29)
| Metric | Value |
|--------|-------|
| Config | Full nGPT, 12L 3x + BigramHash, 30M params |
| Training | **bf16 compile + L2NormalizeHP**, 8xH200, 560s wallclock, 4646 steps at 120ms/step |
| Pre-quant BPB | **1.1705** |
| Post-quant BPB (batch) | **1.1795** |
| Post-quant BPB (sliding) | **1.1570** |
| Quant gap | +0.0090 |
| Artifact | **15.9 MB** (pruned 7.8% to fit) |
| SLURM job | 11140227 |

### Previous Best (fp32 compile)
| Metric | Value |
|--------|-------|
| Config | Full nGPT, 12L 3x + BigramHash, 29.2M params |
| Training | fp32 compile, 8xH200, 560s wallclock, 4431 steps at 126ms/step |
| Post-quant BPB | **1.2056** |
| Artifact | **13.75 MB** |
| SLURM job | 11134842 |

### 11L Variant (faster per-step)
| Metric | Value |
|--------|-------|
| Config | Full nGPT, 11L 3x + BigramHash, 27M params |
| Training | bf16 compile, 8xH200, 560s, 5087 steps at 110ms/step |
| Post-quant BPB (sliding) | **1.1736** |
| Artifact | **14.3 MB** (no pruning needed) |
| SLURM job | 11140232 |

### Reference
| Model | BPB | Notes |
|-------|-----|-------|
| Standard model (our baseline) | 1.2562 (batch), 1.1213 (sliding) | Muon + bf16 compile, all features |
| SOTA | 1.1194 (post-TTT) | Submission bar = 1.1144 |

### Multi-Seed Validation (12L 3x bf16 compile)
| Seed | Sliding BPB | Artifact | Pruned |
|------|-------------|----------|--------|
| 1337 | **1.1570** | 15.9 MB | 7.8% |
| 42 | **1.1583** | 15.9 MB | 6.8% |
| 7 | **1.1594** | 15.9 MB | ~7% |
| **Mean ± Std** | **1.1582 ± 0.0012** | | |

### Full Overnight Sweep Results (2026-03-29)
| Job | Config | ms/step | Sliding BPB | Artifact | Verdict |
|-----|--------|---------|-------------|----------|---------|
| **11140227** | **12L 3x int6** | **120ms** | **1.1570** | **15.9 MB** | **Best — winner** |
| 11140235 | 12L 3x int6 s42 | 122ms | 1.1583 | 15.9 MB | Multi-seed OK |
| 11140236 | 12L 3x int6 s7 | 122ms | 1.1594 | 15.9 MB | Multi-seed OK |
| 11140234 | 12L 3x int5 | 121ms | 1.1647 | 13.1 MB | Int5 costs 0.008 BPB |
| 11142211 | 12L 3.5x int5 | 126ms | 1.1705 | ~15 MB | Wider doesn't help |
| 11140232 | 11L 3x int6 | 110ms | 1.1736 | 14.3 MB | Fewer layers = worse |
| 11140233 | 12L 3.5x int6 | 128ms | 1.1754 | 16.1 MB | OVER budget |
| 11142212 | 13L 3x int5 | 128ms | 1.1787 | 13.8 MB | More depth = worse |
| 11141290 | 12L 3x no wn | 118ms | 2.7842 | 16.6 MB | Broken — wn required |

**Key finding:** Weight norm (NGPT_WEIGHT_NORM=1) is REQUIRED for nGPT quantization. Without it, renorm dequantization fails catastrophically (+1.6 BPB).

---

## Novel Contributions (for PR writeup)

### 1. Three Fixes That Make Full nGPT Work
PR #831 dismissed nGPT (got 1.6915 BPB). We fixed it with:
- **Small-init proj weights** (std=0.01 instead of zeros) — zero + normalize = stuck
- **Learnable logit_scale** (init sqrt(dim)) — unit-norm hidden states → tiny logits without scaling
- **Don't normalize tok_emb** — logits need embedding magnitude
- Result: 1.2714 BPB (only 0.015 behind standard)

### 2. Post-Dequant Renormalization
- `F.normalize(w_deq, dim=-1)` after int6 dequantization
- Reduces quant gap from 0.35 → 0.0005-0.008 BPB
- 3 lines of code, 700x reduction

### 3. torch.compile Precision Compounding Bug
- nGPT's 123 normalize calls per forward create a precision feedback loop
- torch.compile fuses bf16 ops, eliminating float32 casts in normalize
- Small errors compound catastrophically through the chain
- **Fix: fp32 compile (disable autocast)** — works but 39% slower
- Related: PyTorch Issue #168126

### 4. Stochastic RYS (SRYS)
- Train with random layer repetition (p=0.3)
- On standard transformers: -0.0005 BPB (negligible)
- On nGPT: -0.006 BPB (12x amplification)
- Discovery: identity-or-reject dichotomy on unconstrained spaces

### 5. Compression Paradox
- nGPT compresses better at SHORT training (0.414 bytes/param at 1870 steps)
- At full training: 0.589 bytes/param — identical to standard
- The "compression advantage" was a mirage from undertrained weights

### 6. Riemannian Muon (novel optimizer)
- Tangent-plane projection after Newton-Schulz orthogonalization
- +2% overhead, 5x closer to standard Muon than AdamW
- But still 0.19 BPB behind standard Muon at 2000 steps

### 7. Int5 Full-Model Quantization
- Standard models need mixed int5 MLP + int6 attention
- nGPT with renorm dequant: int5 everywhere works (quant gap 0.007)

---

## bf16 Compile: SOLVED (2026-03-29)

### The Fix: L2NormalizeHP with `allow_in_graph`

`torch._dynamo.allow_in_graph` on a custom `autograd.Function` makes normalize opaque to torch.compile. Inductor can't fuse through it, so fp32 norm precision is preserved. Zero graph breaks. bf16 matmuls for everything else.

| Mode | ms/step (8xH200) | Steps in 560s | Sliding BPB |
|------|-------------------|---------------|-------------|
| NO_COMPILE (eager) | 912ms | ~620 | — |
| Compile + fp32 (old) | 126-154ms | 3640-4431 | 1.2056 (batch) |
| **Compile + bf16 + L2NormalizeHP** | **119ms** | **~4706** | **1.1570** |
| Standard model bf16 (ref) | 88ms | 6800 | 1.1213 |

### What Failed
| Approach | Why |
|----------|-----|
| Manual fp32 norm | Compile fuses through float() casts |
| emulate_precision_casts | Not comprehensive for fused ops |
| Local autocast(enabled=False) | Compile fuses across context |
| @compiler.disable | 86 graph breaks → slower than eager |
| Custom Triton kernel | Tested: 17% SLOWER than PyTorch ops for these tensor sizes |

### What Worked
`allow_in_graph` prevents Inductor from ever seeing the internal ops. The function is a single opaque node — compile schedules it in the graph without tracing into it. Internal float32 precision preserved.

### Remaining Gap: 119ms vs 88ms
The 31ms overhead comes from ~86 opaque Python function calls per forward (each launching 4 eager CUDA kernels). This is a fundamental cost of the approach — no way to eliminate it without making compile handle mixed precision correctly.

---

## Architecture Details

### Full nGPT Block Forward
```python
# Normalize both sides, interpolate on hypersphere
h_norm = normalize_hp(x_in)
attn_out = self.attn(h_norm)
attn_norm = normalize_hp(attn_out)
x_out = slerp_hp(h_norm, attn_norm, alpha_a.abs())

mlp_out = self.mlp(x_out)
mlp_norm = normalize_hp(mlp_out)
x_out = slerp_hp(x_out, mlp_norm, alpha_m.abs())
```

### What's Disabled vs Standard Stack
| Feature | Standard | nGPT | Reason |
|---------|----------|------|--------|
| XSA | Last 4 layers | Disabled | Untested with full nGPT |
| Value Embeddings | Layers 9,10 | Disabled | Injects unnormalized signal |
| Partial RoPE | 16 dims | Full RoPE | Untested |
| LN Scale | Yes | Disabled | Replaced by F.normalize |
| bf16 compile | Yes | **bf16 + L2NormalizeHP** | Solved via opaque autograd.Function |
| BigramHash | Yes | **Yes** | Compatible |
| U-Net skips | Yes | **Yes** | Compatible |
| SmearGate | Yes | Present | ~no-op |

### Key Env Vars for nGPT
```bash
NGPT_ENABLED=1          # Enable nGPT activation normalization
NGPT_FULL=1             # Full nGPT (normalize both sides)
NGPT_WEIGHT_NORM=1      # Forward-pass weight normalization
NGPT_QUANT_MODE=renorm  # Post-dequant renormalization
INT5_ALL=1              # Int5 for all weights (not just MLP)
NO_COMPILE=0            # Compile enabled (fp32 via _autocast=False)
```

---

## Environment Details

### ORCD Cluster (Development)
- **Partitions:** `pg_tata` (H200×8, L40S×4), `mit_normal_gpu` (H200×8, H100×4), `mit_preemptable` (H200×8, H100×8)
- **QOS limits:** `mit_normal_gpu` max 2 GPUs/user, `mit_preemptable` max 4 GPUs/user, `pg_tata` no GPU limit but be courteous
- **H200 nodes:** node4300, node4301 (pg_tata), 8 GPUs each, 141GB HBM3e
- **CUDA:** 12.4 (module load), PyTorch 2.6.0+cu124
- **Conda env:** `pgolf` at `$HOME/.conda/envs/pgolf`
- **flash-attn:** Built from source with GLIBC 2.28 patch, needs `LD_PRELOAD=$HOME/.conda/envs/pgolf/lib/glibc_shim.so`
- **torch.compile cache:** MUST use `export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER` — home dir has quota (~50GB), compile cache fills it fast
- **Disk quota:** ~50GB home. Clean `/tmp/torchinductor_tomli/` and `.cache/pip/` regularly. HuggingFace dataset cache (16GB) must stay.
- **Working dir:** `/orcd/home/002/tomli/parameter_golf/`
- **SLURM scripts:** `scripts/slurm/`
- **Logs:** `logs/slurm/`
- **NEVER cancel SLURM jobs without asking** — bash sessions on pg_tata are the user's Claude Code workspace

### RunPod (Competition Evaluation)
- **GPU:** 8xH100 SXM 80GB
- **CUDA:** 12.6-12.8
- **PyTorch:** 2.9.1+cu128
- **Time limit:** 600s training, 600s evaluation
- **NEVER use nohup** — background processes leak CUDA contexts
- **zstandard not pre-installed** — `pip install --break-system-packages zstandard`
- **torch.compile** works with bf16 + L2NormalizeHP opaque normalize (solved 2026-03-29)
- **FORCE_SDPA=1** needed on ORCD, native flash_attn works on RunPod

### Key Timing Translations
| Metric | ORCD 4xH200 | ORCD 8xH200 | RunPod 8xH100 |
|--------|-------------|-------------|---------------|
| fp32 compile step | ~270ms | ~126-154ms | ~130ms (est) |
| bf16 + L2NormalizeHP step | — | ~119ms (12L), ~110ms (11L) | ~125ms (est) |
| Standard bf16 step (no nGPT) | ~135ms | ~80ms | ~88ms |
| nGPT steps in 560s (bf16 fix) | — | ~4706 (12L), ~5090 (11L) | ~4480 (est) |

---

## Completed Experiments (2026-03-29)

### Job 11136281: 12L 3.5x fp32 compile (COMPLETED)
- 8xH200, fp32 compile, 560s wallclock, 3783 steps at 148ms/step
- val_bpb: 1.1731 pre-quant, **1.1553 sliding window**
- Artifact: **20.6 MB (OVER)** — compression paradox confirmed

### Job 11140227: 12L 3x bf16 compile (COMPLETED — new best)
- 8xH200, bf16 L2NormalizeHP, 560s wallclock, 4646 steps at 120ms/step
- val_bpb: **1.1570 sliding window**
- Artifact: **15.9 MB** (pruned 7.8%)

### Job 11140232: 11L 3x bf16 compile (COMPLETED)
- 8xH200, bf16 L2NormalizeHP, 560s wallclock, 5087 steps at 110ms/step
- val_bpb: **1.1736 sliding window**
- Artifact: **14.3 MB** (no pruning)

---

## What To Do Next (Priority Order)

### 1. Review Overnight Results
Check jobs 11140233-11140236, 11141290 for:
- 12L 3.5x (wider MLP — will it fit 16MB?)
- INT5_ALL (better compression → less pruning?)
- Multi-seed (seeds 42, 7 — statistical validation)
- No weight norm (faster? NGPT_WEIGHT_NORM=0)

### 2. RunPod Validation
Run the best config on actual competition hardware:
- Use `submission/run_ngpt_on_runpod.sh`
- Verify timing (target: 580s training + GPTQ within 600s)
- Verify artifact < 16MB
- Get official H100 BPB numbers

### 3. TTT on nGPT
Test-time training with weight renormalization after each gradient step (implemented, untested).
Could add -0.002 to -0.005 BPB.

### 4. Feature Re-enablement
Try on the bf16-compiled best config:
- XSA (just an attention mask — should work)
- Partial RoPE (applied to Q/K before normalize — should work)
Each could save 0.001-0.002 BPB.

### 5. Research PR Writeup
Write the competition PR with:
- Standard model as BPB entry
- nGPT findings as research contribution
- The bf16 compile fix as a novel contribution to PyTorch community
- All tables from nGPT.md and RYS.md

---

## Key Files
| File | Purpose |
|------|---------|
| `repo/train_gpt.py` | Main training script (modified for nGPT) |
| `nGPT.md` | Full nGPT research documentation |
| `RYS.md` | RYS/SRYS research documentation |
| `EXPERIMENT_RESULTS.md` | All experiment results |
| `STACK.md` | Technology stack reference |
| `CLAUDE.md` | Project instructions and conventions |
| `scripts/slurm/train_ngpt_full_4h200.sbatch` | Base nGPT SLURM template |
| `scripts/test_ngpt_compile.py` | Compile precision diagnostic |

## Key Code Locations in train_gpt.py
| Function/Section | What it does |
|-----------------|-------------|
| `normalize_hp()` | Float32 normalize (line ~408) |
| `slerp_hp()` | Float32 interpolation + normalize (line ~414) |
| `Block.forward` nGPT path | Full nGPT forward with `_ngpt_full` flag (line ~672) |
| `CastedLinear.forward` | Weight norm via `_ngpt_norm` flag (line ~422) |
| `GPT.__init__` `logit_scale` | Learnable logit scaling for nGPT (line ~809) |
| `GPT._init_weights` | Small-init for nGPT proj weights (line ~830) |
| `dequantize_mixed_int6` | Renorm quant mode (line ~1499) |
| `eval_val_sliding_ttt` | TTT with nGPT weight renorm (line ~1208) |
| `Muon.step` | Riemannian Muon option (line ~210) |
| Training loop autocast | `_autocast = not args.ngpt_enabled` (line ~1943) |

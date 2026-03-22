# Stack Research

**Domain:** Parameter-efficient language model training (OpenAI Parameter Golf competition)
**Researched:** 2026-03-22
**Confidence:** HIGH (stack verified against competition repo, leading submissions, and official PyTorch docs)

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PyTorch | 2.10.x (match RunPod) | Deep learning framework | Competition baseline uses PyTorch; `torch.optim.Muon` is native in 2.10; `torch.compile` matured for Hopper GPUs; `torch.distributed` DDP is the competition standard |
| CUDA Toolkit | 12.4 (cluster) / match RunPod | GPU compute runtime | H100/H200 both require sm_90; CUDA 12.4 available on pg_tata cluster; RunPod likely uses 12.6-12.8; forward-compatible within 12.x |
| Python | 3.11 | Runtime | RunPod templates standardize on 3.11; good balance of performance and library compatibility; avoids 3.14 bleeding edge |
| Triton | bundled with PyTorch | Custom GPU kernel compiler | Included with PyTorch 2.x; powers `torch.compile` Inductor backend; needed if writing custom quantization or attention kernels |

**Confidence: HIGH** -- PyTorch 2.10 confirmed released 2026-01-21 with Muon optimizer. Competition requirements.txt pins no PyTorch version (uses latest). RunPod templates available up to PyTorch 2.8 + CUDA 12.8.

**Critical note on environment parity:** The competition evaluates on RunPod 8xH100 SXM. Development on pg_tata 8xH200 is fine (H200 is a superset of H100 with more VRAM and bandwidth), but final timing validation MUST happen on RunPod 8xH100. The key risk is that H200 trains faster, so a run that fits in 10 minutes on H200 may exceed 10 minutes on H100. Always leave timing margin.

### Optimizer Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `torch.optim.Muon` | PyTorch 2.10 built-in | Optimizer for 2D weight matrices | Native PyTorch implementation; used by ALL top-5 submissions; 1.35x faster convergence than Adam for small transformers; Newton-Schulz orthogonalization in bf16 |
| AdamW | PyTorch built-in | Optimizer for non-2D params | Muon only works on 2D weight matrices; embeddings, biases, layer norms need AdamW; standard combination pattern |

**Muon details (HIGH confidence, verified against official PyTorch 2.10 docs and KellerJordan/Muon repo):**
- Only optimizes parameters with `ndim >= 2` (weight matrices in hidden layers)
- Uses Newton-Schulz iteration (5 steps, coefficients 3.4445, -4.7750, 2.0315) for orthogonalization
- Runs Nesterov momentum + orthogonalization post-processing
- Embeddings, biases, gains, classifiers handled by separate AdamW
- Key hyperparams from leading submission: `matrix_lr=0.02, weight_decay=0.04, momentum=0.99`
- Alternative: KellerJordan/Muon standalone (`pip install git+https://github.com/KellerJordan/Muon`) if PyTorch < 2.10

### Attention & Kernel Libraries

| Library | Version | Purpose | Why Recommended |
|---------|---------|---------|-----------------|
| `flash-attn` | 2.8.3 | FlashAttention-2 for Hopper GPUs | 75% H100 utilization; prebuilt wheels available; competition explicitly permits it; saves memory for larger batch sizes |
| `kernels` (PyPI) | latest | Modular kernel loader | Listed in competition requirements.txt; provides `get_kernel("kernels-community/flash-attn2")` and `flash-attn3` access |
| `torch.nn.functional.scaled_dot_product_attention` | PyTorch built-in | SDPA with automatic backend selection | Zero-install fallback; auto-selects FlashAttention or efficient attention backend; `torch.compile`-friendly |

**Confidence: HIGH** -- flash-attn 2.8.3 confirmed on PyPI (Jan 2026), supports PyTorch 2.9+, CUDA 12.8 wheels. The `kernels` package is in the official parameter-golf requirements.txt.

**Recommendation:** Use `flash-attn` directly via pip for simplicity. The `kernels` package is an alternative loader but adds abstraction. For this model size (~10-15M params), FlashAttention's primary benefit is memory savings (enabling larger batch sizes / longer sequences), not raw speed -- the attention matrices are small.

### Quantization Stack

| Technology | Version | Purpose | When to Use |
|------------|---------|---------|-------------|
| Custom int5/int6 QAT (in train_gpt.py) | N/A (hand-written) | Quantization-aware training | ALWAYS -- this is the core technique for fitting in 16MB; top submission uses int5 for MLP, int6 for attention |
| Straight-Through Estimator (STE) | Hand-implemented | Gradient flow through quantization | Required for QAT; fake-quantize in forward pass, pass gradients through in backward |
| `torchao` | 0.16.x | PyTorch-native quantization toolkit | OPTIONAL -- provides int4/int8 building blocks but NOT int5/int6; useful for reference implementations only |

**Quantization is custom, not library-driven (HIGH confidence):**
The competition demands non-standard bit widths (int5 = 5-bit, int6 = 6-bit) that no standard library supports. Every top submission implements quantization manually:
- **Int5 [-16, 15]:** 32 levels, best compression ratio with zstd (~1.88x), used for MLP weights by leader
- **Int6 [-32, 31]:** 64 levels, better precision for attention weights (~1.51x zstd ratio)
- **FP16:** Kept for tied embeddings and critical projection layers
- **QAT with STE:** Train with fake-quantize nodes; `round()` in forward, straight-through in backward

**This means:** No need to learn torchao deeply. The quantization code lives in train_gpt.py and is ~50 lines of custom PyTorch. Focus effort on understanding the int5 vs int6 tradeoff and which layers tolerate lower precision.

### Compression Stack

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `zstandard` (python-zstandard) | 0.25.0 | Zstd compression for model artifacts | Top submissions use zstd level 22; dramatically better compression ratio than zlib for quantized weights; ~1.88x ratio on int5 weights |
| `zlib` | Python stdlib | Baseline compression | Used in official baseline (int8 + zlib); strictly worse than zstd for this use case; only use for compatibility testing |

**Confidence: HIGH** -- Leading submissions explicitly document zstd-22 compression. The baseline uses zlib, but every competitive submission has switched to zstd.

**Key detail:** The compression level matters enormously. zstd level 22 is the maximum standard level and provides the best ratio at the cost of compression time (which is fine -- compression happens once after training, not in the hot loop). The artifact size budget is 16,000,000 bytes = code + compressed weights. Every byte saved by better compression = more parameters = lower BPB.

```python
import zstandard as zstd
cctx = zstd.ZstdCompressor(level=22)
compressed = cctx.compress(quantized_weights_bytes)
```

### Data Loading & Tokenization

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `datasets` (HuggingFace) | latest | FineWeb dataset access | Competition standard; downloads/caches FineWeb shards; binary shard format (100M tokens each) |
| `huggingface-hub` | latest | Dataset download | Required dependency for FineWeb access |
| `tiktoken` | 0.12.0 | BPE tokenization | Listed in competition requirements; OpenAI's fast BPE tokenizer |
| `sentencepiece` | latest | Alternative tokenizer (sp1024/sp4096) | Baseline uses 1024-vocab sentencepiece; some submissions explore larger vocab |
| Custom `DistributedTokenLoader` | in train_gpt.py | Multi-GPU data distribution | Implemented in baseline; distributes binary shards across ranks; handles gradient accumulation |

**Confidence: HIGH** -- All from official competition requirements.txt.

**Tokenization strategy note:** The competition metric is bits-per-byte (BPB), which is tokenizer-agnostic. This means tokenizer choice affects how efficiently the model can compress text, making it a meaningful optimization axis. The baseline uses sp1024 (1024 vocab), but BigramHash (hashing consecutive token pairs into a large embedding table) has emerged as the dominant approach among leaders.

### Distributed Training

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `torch.distributed` + DDP | PyTorch built-in | Multi-GPU data parallelism | Competition baseline uses DDP; model is small (~10-15M params) so FSDP sharding is unnecessary; DDP replicates model on each GPU, syncs gradients |
| `torchrun` | PyTorch built-in | Process launcher for distributed | Standard launcher; `torchrun --nproc_per_node=8 train_gpt.py`; handles rank/world_size env vars |

**Use DDP, NOT FSDP (HIGH confidence):**
- Model is ~10-15M parameters = ~20-30MB in fp16 = trivially fits in GPU memory
- FSDP is designed for models too large for single GPU memory -- complete overkill here
- DDP overhead is minimal for small models; FSDP's shard/unshard overhead would hurt
- Every competition submission uses DDP via `torchrun`
- The 8-GPU parallelism is purely for batch throughput, not model sharding

**Training config from leader:**
```bash
torchrun --nproc_per_node=8 train_gpt.py
# seq_len=2048, global_batch=786K tokens, gradient accumulation as needed
```

### Compilation & Performance

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `torch.compile` | PyTorch 2.10 | JIT compilation of model forward | Fuses ops, generates Triton kernels; important for small-model throughput; use `mode="reduce-overhead"` for inference |
| `torch.amp` (autocast) | PyTorch built-in | Mixed precision training | bf16 on H100/H200 for matmuls; fp32 for reductions; standard practice |
| `torch.backends.cuda.matmul.allow_tf32 = True` | PyTorch setting | TF32 tensor core usage | Free 3x speedup for matmuls on H100; no accuracy loss for this model scale |

**Confidence: HIGH** -- These are standard PyTorch performance settings used in the baseline.

### Experiment Tracking & Utilities

| Tool | Purpose | Notes |
|------|---------|-------|
| `tqdm` | Progress bars | In competition requirements.txt; use for training loop progress |
| `numpy` | Array operations | In competition requirements.txt; weight manipulation, statistics |
| Custom logging | Training metrics | Write to CSV/JSON; competition requires training logs in submission |
| `setuptools` | Package utilities | In competition requirements.txt; likely for build tooling |

## Installation

### On pg_tata cluster (H200 development)

```bash
# Load CUDA module
module load cuda/12.4.0 cudnn/9.8.0.87-cuda12

# Create conda environment
conda create -n pgolf python=3.11 -y
conda activate pgolf

# Core PyTorch (match version to what RunPod uses)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu124

# FlashAttention (prebuilt wheel)
pip install flash-attn==2.8.3 --no-build-isolation

# Competition dependencies (from requirements.txt)
pip install numpy tqdm huggingface-hub datasets tiktoken sentencepiece setuptools kernels
pip install typing-extensions==4.15.0

# Compression
pip install zstandard==0.25.0

# Muon standalone (only if PyTorch < 2.10)
# pip install git+https://github.com/KellerJordan/Muon
```

### On RunPod (H100 evaluation)

```bash
# RunPod template has pre-installed deps
# Just verify versions match:
python -c "import torch; print(torch.__version__)"
pip install zstandard==0.25.0  # May not be pre-installed
```

### SLURM job script template

```bash
#!/bin/bash
#SBATCH --job-name=pgolf
#SBATCH --partition=gpu          # adjust to your cluster partition with H200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00          # 30 min for safety margin (10 min training + overhead)
#SBATCH --output=logs/%j.out

module load cuda/12.4.0 cudnn/9.8.0.87-cuda12
source activate pgolf

# Training
torchrun --nproc_per_node=8 train_gpt.py

# Post-training: compress and measure artifact
python compress.py --level 22
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PyTorch 2.10 | PyTorch 2.6-2.8 | If RunPod pins an older version; Muon available standalone from KellerJordan/Muon repo |
| DDP | FSDP | Never for this competition -- model is <30MB, fits trivially in single GPU memory |
| DDP | DeepSpeed ZeRO | Never -- same reasoning as FSDP; adds complexity for zero benefit at this scale |
| flash-attn (pip) | `kernels` package loader | If you need to switch between FA2 and FA3 easily; slight abstraction overhead |
| flash-attn | PyTorch SDPA | If flash-attn installation fails; SDPA is zero-install but slightly less optimized |
| zstd level 22 | zlib level 9 | Never -- zstd strictly dominates zlib for quantized weight compression |
| zstd level 22 | zstd level 19-21 | If compression time matters (it shouldn't -- compression is post-training) |
| Custom int5/int6 QAT | torchao int4/int8 | Never for competition -- torchao doesn't support int5/int6; competition needs custom bit widths |
| Muon + AdamW | Pure AdamW | Never -- every top submission uses Muon; the convergence advantage is proven |
| Muon (PyTorch 2.10) | KellerJordan/Muon standalone | If stuck on older PyTorch; standalone has same algorithm |
| sentencepiece | tiktoken | Both available; tokenizer choice is an optimization dimension, not a fixed decision |
| Custom BigramHash | Standard token embeddings | BigramHash is strictly better for this competition -- captures bigram statistics in embedding layer |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| FSDP / DeepSpeed ZeRO | Model is 10-15M params; sharding overhead exceeds any benefit; adds debugging complexity | DDP via torchrun |
| torchao for quantization | Does not support int5/int6 bit widths needed for competition; only int4/int8 | Custom QAT with STE in train_gpt.py |
| PyTorch Lightning / HuggingFace Trainer | Abstraction layers add overhead and obscure optimization; competition needs bare-metal PyTorch | Raw PyTorch training loop |
| Weights & Biases / MLflow | Adds dependency complexity; competition evaluates reproducibility via logs, not dashboards | Simple CSV/JSON logging |
| zlib compression | Strictly dominated by zstd at every compression level for quantized weight data | zstandard with level 22 |
| Post-training quantization (PTQ) | QAT is strictly better -- model learns to compensate for quantization noise during training | QAT with Straight-Through Estimator |
| FP8 training | H100 supports FP8 but model is too small to benefit from reduced precision matmuls; int5/int6 weight quantization is the real win | bf16 training with int5/int6 QAT |
| torch.distributed.launch | Deprecated in favor of torchrun | torchrun |
| Apex (NVIDIA) | Functionality absorbed into PyTorch core (AMP, fused optimizers); adds unnecessary dependency | PyTorch native equivalents |
| Custom CUDA kernels | PROJECT.md explicitly marks out of scope; Triton via torch.compile handles kernel fusion; dev time better spent on architecture/quantization | torch.compile + Triton + flash-attn |

## Stack Patterns by Competition Phase

**Phase 1: Baseline reproduction**
- Use exact competition requirements.txt
- PyTorch + DDP + Muon + int8 + zlib (match baseline)
- Goal: reproduce 1.2244 BPB baseline

**Phase 2: Architecture optimization**
- Add flash-attn for memory savings (enables larger batch / longer seq)
- Add torch.compile for throughput
- Switch to zstd-22 compression
- Goal: implement SmearGate, BigramHash, U-Net skip connections

**Phase 3: Quantization push**
- Implement custom int5/int6 QAT with STE
- Mixed precision: int5 for MLP, int6 for attention, FP16 for embeddings
- Goal: maximize params-per-byte within 16MB budget

**Phase 4: Training optimization**
- Tune Muon hyperparams (lr, WD, momentum)
- Implement SWA (last 40% of warmdown, 24 checkpoints)
- Sliding window eval (stride=64)
- Goal: squeeze every 0.001 BPB

## Version Compatibility Matrix

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PyTorch 2.10 | CUDA 12.4-12.8 | sm_90 support for H100/H200; use cu124 wheels on cluster |
| flash-attn 2.8.3 | PyTorch 2.9+, CUDA 12.8 | Prebuilt wheels for py3.10-3.13; may need `--no-build-isolation` for source build |
| zstandard 0.25.0 | Python 3.8+ | Pure Python fallback available; C extension preferred |
| tiktoken 0.12.0 | Python 3.8+ | Rust-based; fast tokenization |
| Muon (torch.optim) | PyTorch 2.10+ only | Use KellerJordan/Muon for PyTorch < 2.10 |
| kernels (PyPI) | PyTorch 2.x | Provides FA2/FA3 loaders; used in competition requirements |

**CRITICAL compatibility note:** The RunPod evaluation environment likely runs a specific PyTorch + CUDA combination. Before final submission, verify your code runs correctly on RunPod by launching a test pod. The `requirements.txt` in the competition repo does NOT pin PyTorch, meaning RunPod installs whatever is latest in their template. If RunPod uses PyTorch 2.8 and you develop with 2.10, test for API differences (especially torch.optim.Muon availability -- fall back to standalone implementation).

## Competition-Specific Considerations

### Artifact Size Budget (16MB)

The 16MB limit covers code + compressed model. Typical budget allocation:
- **train_gpt.py:** ~10-20KB (negligible)
- **Compressed weights:** ~15.9MB
- **Strategy:** Maximize weight budget by minimizing code size

### Reproducibility Requirements

- Must provide training logs
- Must beat SOTA by >= 0.005 nats at p < 0.01
- Run 3+ seeds (1337, 42, 7 are conventional)
- Report mean and std of val_bpb across seeds

### Development vs Evaluation Hardware

| Aspect | pg_tata (dev) | RunPod (eval) |
|--------|---------------|---------------|
| GPU | H200 SXM (141GB) | H100 SXM (80GB) |
| Count | 8 per node | 8 per pod |
| Bandwidth | Higher (HBM3e) | Lower (HBM3) |
| CUDA | 12.4 (module) | 12.6-12.8 (template) |
| Training speed | Faster | Slower (this is the target) |
| VRAM | Not a constraint | Not a constraint (model is tiny) |

**Timing calibration:** Expect ~10-20% slower training on H100 vs H200. If a run takes 8.5 minutes on H200, it may take 9.5-10.5 minutes on H100. Target 8-minute training on H200 to leave margin.

## Sources

- [OpenAI Parameter Golf Repository](https://github.com/openai/parameter-golf) -- competition rules, baseline code, requirements.txt (HIGH confidence)
- [Leading submission: 10L Int5-MLP + BigramHash, 1.1428 BPB](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md) -- architecture and hyperparameter details (HIGH confidence)
- [Second place: Sliding Window + FP16 Emb + Muon WD](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md) -- alternative approach details (HIGH confidence)
- [DeepWiki: Parameter Golf Technical Overview](https://deepwiki.com/openai/parameter-golf) -- aggregated technical analysis (MEDIUM confidence)
- [PyTorch 2.10 Release Blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) -- Muon optimizer, combo kernels, varlen_attn (HIGH confidence)
- [torch.optim.Muon Documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html) -- official API (HIGH confidence)
- [KellerJordan/Muon GitHub](https://github.com/KellerJordan/Muon) -- standalone implementation, algorithm details (HIGH confidence)
- [Muon Blog Post](https://kellerjordan.github.io/posts/muon/) -- Newton-Schulz iteration details (HIGH confidence)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2407.08608) -- Hopper-optimized attention (HIGH confidence)
- [flash-attn PyPI](https://pypi.org/project/flash-attn/) -- version 2.8.3, compatibility (HIGH confidence)
- [python-zstandard Docs](https://python-zstandard.readthedocs.io/en/latest/) -- zstd compression API (HIGH confidence)
- [Zstandard Official](https://facebook.github.io/zstd/) -- compression algorithm details (HIGH confidence)
- [RunPod PyTorch Templates](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8) -- RunPod environment details (MEDIUM confidence)
- [PyTorch DDP vs FSDP](https://www.jellyfishtechnologies.com/ddp-vs-fsdp-in-pytorch-unlocking-efficient-multi-gpu-training/) -- parallelism strategy rationale (MEDIUM confidence)
- [torchao GitHub](https://github.com/pytorch/ao) -- quantization toolkit capabilities and limitations (MEDIUM confidence)
- [tiktoken GitHub](https://github.com/openai/tiktoken) -- tokenizer library (HIGH confidence)

---
*Stack research for: Parameter Golf -- small LM training competition*
*Researched: 2026-03-22*

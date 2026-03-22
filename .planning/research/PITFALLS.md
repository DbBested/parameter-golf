# Pitfalls Research

**Domain:** Parameter-constrained language model competition (OpenAI Parameter Golf)
**Researched:** 2026-03-22
**Confidence:** HIGH (competition rules verified via official repo and DeepWiki; quantization/training pitfalls verified via multiple academic sources)

---

## Critical Pitfalls

These cause wasted GPU hours, blown submissions, or disqualification.

### Pitfall 1: Artifact Size Miscalculation (16MB != 16MiB)

**What goes wrong:**
The artifact limit is 16,000,000 bytes (16 MB decimal), NOT 16,777,216 bytes (16 MiB). Participants build models targeting 16 MiB, submit, and exceed the limit by ~750KB. The artifact includes BOTH code bytes (train_gpt.py) AND compressed model bytes. Teams forget to count code size or miscalculate compression ratios, ending up over budget.

**Why it happens:**
Confusion between MB and MiB is endemic in computing. Additionally, teams add features to train_gpt.py (lookup tables, custom tokenizer logic, hardcoded constants) without tracking how those code bytes eat into the 16,000,000 total. Compression ratios are estimated rather than measured, and small changes to quantization (e.g., int5 vs int6) have nonlinear effects on compressed size.

**How to avoid:**
- Build artifact size checking into every experiment run. After training, always compute `code_bytes + compressed_model_bytes` and assert it is under 16,000,000.
- Track code size growth as features are added. Large lookup tables or embedding dictionaries in code eat into budget.
- Use the exact compression pipeline (zstd-22 or zlib-9) that will be used in submission -- never estimate.
- Keep a ~500KB safety margin for code growth during development.

**Warning signs:**
- Artifact size above 15.5MB without margin budgeted
- Code file growing beyond 50KB without accounting
- Adding new embedding tables or lookup arrays to the code

**Phase to address:**
Phase 1 (Baseline) -- establish artifact size validation as part of every training run from day one. The size check must be automated and fail-fast.

---

### Pitfall 2: H200-to-H100 Timing Miscalibration

**What goes wrong:**
Development on H200 GPUs (141GB HBM3e, 4.8 TB/s bandwidth) produces a model that trains in 8 minutes locally but takes 12+ minutes on the competition's 8xH100 SXM (80GB HBM3, 3.35 TB/s bandwidth). The submission fails the 10-minute wall-clock requirement. H200 and H100 share identical compute (FP16/BF16 FLOPS), but H200 has 43% more memory bandwidth and 76% more VRAM. Memory-bandwidth-bound operations (attention with long sequences, embedding lookups, large batch data loading) run significantly faster on H200.

**Why it happens:**
Teams optimize on the hardware they have. The H200's extra bandwidth makes memory-bound operations deceptively fast. Operations that are compute-bound scale identically, but operations that are memory-bound (which dominate small model training) are 30-45% faster on H200. The gap varies per operation, making simple scaling factors unreliable.

**How to avoid:**
- Budget for no more than 7 minutes of training on H200, providing ~30% margin for H100.
- Profile which operations are memory-bound vs compute-bound using torch.profiler. Memory-bound ops need the largest timing correction.
- Validate on actual RunPod 8xH100 before any submission. Budget for at least 3-5 RunPod validation runs.
- Track per-operation timing breakdowns, not just total wall-clock.

**Warning signs:**
- Training takes more than 7 minutes on H200
- Sequence length or batch size increased to fill H200's extra VRAM (these changes will OOM or slow down on H100)
- Large embedding tables that are memory-bandwidth-bound during lookups

**Phase to address:**
Phase 1 (Baseline) -- establish H100 timing validation as a gate for all experiments. Every promising configuration must be validated on RunPod before being considered final.

---

### Pitfall 3: QAT Training Instability at Int5 and Below

**What goes wrong:**
Training with int5 quantization-aware training (QAT) using the standard Straight-Through Estimator (STE) becomes unstable: loss spikes, diverges, or oscillates without converging. This is particularly severe when combining with aggressive learning rates (needed for the 10-minute training budget) and the Muon optimizer. The STE provides biased gradient estimates, and at int5 (only 32 quantization levels), the mismatch between the continuous surrogate and the discrete rounding operator is large enough to cause gradient explosion or stalling.

**Why it happens:**
The STE approximates the gradient of the rounding function with the identity, which is increasingly wrong as bit-width decreases. At int5, each quantization bin spans a wide range, so weights oscillate between bins without converging. The Muon optimizer's orthogonal update steps can exacerbate this by pushing weights through multiple bin boundaries in a single step. StableQAT research (2025-2026) confirms that standard STE exhibits "exploding variance" at ultra-low bit-widths.

**How to avoid:**
- Start with int6 QAT (64 levels), which is stable and well-proven by multiple leaderboard entries. Only attempt int5 after int6 is solid.
- Use smaller learning rates during QAT fine-tuning to avoid destabilizing quantized weights.
- Consider the Rotated Damped Fourier Surrogate (RDFS) from StableQAT, which bounds gradient variance and strictly generalizes STE.
- Monitor per-layer gradient norms during QAT training -- diverging norms in early layers signal instability.
- Use gradient clipping as a safety net, not a primary solution.

**Warning signs:**
- Loss spikes after initial convergence during QAT
- Per-layer gradient norm ratios exceeding 10x between layers
- Weights clustering at quantization bin boundaries
- Training loss oscillating without decreasing over 100+ steps

**Phase to address:**
Phase 3 (Quantization) -- implement int6 QAT first, then carefully ablate int5 with stability monitoring. Do not skip to int5.

---

### Pitfall 4: Quantized Weights Compress Poorly with Zstd/Zlib

**What goes wrong:**
Teams quantize weights to int5 or int6 expecting smaller artifact size, but discover that zstd/zlib compression achieves poor ratios on quantized weights. The compressed artifact is larger than expected, pushing past the 16MB limit. Quantized weights have high entropy and near-uniform bit distributions, which are exactly the patterns that dictionary-based compression algorithms like zstd and zlib handle worst.

**Why it happens:**
Generic compression (zstd, zlib, gzip) works by finding repeated byte patterns. Quantized neural network weights have high entropy in their value distribution -- the values are pseudo-random within the quantization range. Zstd achieves only 1.0-1.1x compression on high-entropy quantized data. Recent research (ZipNN, QStore 2025) confirms that "general-purpose lossless compressors achieve suboptimal compression ratios on model weights due to their high entropy." The irony: more aggressive quantization (lower bits) may increase per-bit entropy, making compression worse.

**How to avoid:**
- Measure actual compressed sizes after quantization, not theoretical sizes. `int5_params * 5/8` is NOT the actual compressed size.
- Consider entropy-aware quantization: structure the quantization to produce more compressible patterns (e.g., encourage weight clustering).
- Evaluate alternative compression: QStore achieves 2.2x better compression than raw Safetensors by using byte-level entropy coding tuned for quantized weights.
- Use zstd at level 22 (maximum compression) rather than default levels -- the leading submission uses zstd-22.
- Consider weight ordering and layout optimizations that improve compression (group similar magnitude weights together).

**Warning signs:**
- Zstd compression ratio below 1.05x on quantized weights
- Artifact size growing when switching from int8 to int6 (counter-intuitively)
- Large per-row scale arrays adding overhead that offsets quantization gains

**Phase to address:**
Phase 3 (Quantization) -- always measure compressed artifact size as part of quantization experiments. Build a size tracking dashboard.

---

### Pitfall 5: BPB Calculation Bugs When Changing Tokenizer

**What goes wrong:**
Teams implement a custom tokenizer (different vocab size, BPE rules, or byte-level encoding) but miscalculate BPB. The BPB formula is `val_bpb = (val_loss / ln(2)) * (tokens / bytes)`. If the tokenizer changes, the `tokens/bytes` ratio changes, and the SentencePiece lookup tables (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut) must be rebuilt correctly. Common bugs include double-counting space bytes, miscounting multi-byte UTF-8 characters, or forgetting that boundary tokens contribute zero bytes.

**Why it happens:**
The BPB calculation is subtle. SentencePiece uses a special `_` prefix to mark word boundaries, which represents a space byte. If the LUT doesn't correctly handle this convention for a new tokenizer, space bytes get double-counted or missed. The competition explicitly warns: "Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score." A tokenizer bug can make a model appear better than it is, leading to submission rejection.

**How to avoid:**
- Use the provided eval_val() function exactly as-is. The evaluation function is designed to be immutable.
- If using a custom tokenizer, verify BPB against a known-correct implementation on a small validation subset.
- Cross-check: compute BPB manually on a 10-document sample with known byte counts.
- Keep float64 accumulators for val_loss_sum, val_token_count, and val_byte_count (as the baseline does) to prevent numerical drift.
- Test with the default sp1024 tokenizer first; only then introduce custom tokenizers.

**Warning signs:**
- BPB improves dramatically (>0.05 improvement) when switching tokenizer -- suspect calculation bug
- BPB differs from val_loss in unexpected ways (the ratio should be predictable)
- Boundary tokens contributing non-zero byte counts

**Phase to address:**
Phase 1 (Baseline) -- lock down BPB evaluation as trusted infrastructure before any experiments. Phase 5 (Tokenizer) -- re-validate BPB calculation when introducing custom tokenizers.

---

### Pitfall 6: Non-Reproducibility Across Seeds (p < 0.01 Requirement)

**What goes wrong:**
A configuration scores 1.1420 BPB on one seed but 1.1445 on another, failing the statistical significance test. The competition requires beating SOTA by at least 0.005 nats at p < 0.01. With high variance, you cannot distinguish real improvements from noise. Teams waste GPU hours chasing phantom improvements that are actually within the noise floor.

**Why it happens:**
Small models (10-11 layers, 512 dim) are particularly sensitive to initialization, data ordering, and CUDA nondeterminism. PyTorch CUDA operations using atomicAdd produce different floating-point rounding across runs. cuDNN benchmark mode selects different convolution algorithms nondeterministically. The Muon optimizer with orthogonal projections amplifies small initialization differences. With only 10 minutes of training, there is insufficient time for results to stabilize.

**How to avoid:**
- Run 3+ seeds for every promising configuration from the start. Never make decisions based on single-seed results.
- Set torch.backends.cudnn.deterministic = True and benchmark = False for reproducibility (at some speed cost).
- Use torch.manual_seed() consistently and log seeds with every experiment.
- Track standard deviation across seeds. If std > 0.002 BPB, the configuration is too noisy for meaningful comparison.
- Consider whether a 0.002 improvement is real by computing confidence intervals, not just point estimates.

**Warning signs:**
- Single-seed BPB improvement that reverses on re-run
- Standard deviation across 3 seeds > 0.003 BPB
- Inconsistent ranking of configurations across different seeds

**Phase to address:**
Phase 1 (Baseline) -- establish multi-seed evaluation protocol. Every experiment must report mean +/- std across at least 3 seeds before claiming improvement.

---

### Pitfall 7: Test-Time Training Rule Violation

**What goes wrong:**
Test-time training (TTT) is allowed in Parameter Golf, but with a critical constraint: you can only train on validation tokens that have already been evaluated (already graded). Teams implement TTT that inadvertently trains on tokens before they are scored, which constitutes training on the test set and will be disqualified. The causal masking must be strictly enforced -- you cannot use future tokens to improve predictions on past tokens.

**Why it happens:**
The boundary between "already evaluated" and "not yet evaluated" is subtle in sliding-window evaluation. If you process a batch of tokens [t1...t512] and then do a TTT update, you can only use t1...t512 for the TTT update that improves predictions on t513+. But if you re-evaluate t1...t512 with the updated model, you've violated the rules. The sliding window stride means some tokens appear in multiple evaluation windows, creating confusion about which tokens are "already evaluated."

**How to avoid:**
- Maintain a strict pointer: tokens at positions < pointer have been evaluated and can be used for TTT. Tokens at positions >= pointer cannot be influenced by TTT.
- Log which tokens were used for TTT training vs. evaluation. This audit trail protects against accidental violations.
- Implement TTT as a separate post-evaluation step, not interleaved with evaluation.
- Review the competition rules document carefully: "test-time training is allowed only on validation tokens already evaluated."

**Warning signs:**
- BPB improves more than expected from TTT (>0.02 improvement suggests possible leakage)
- TTT update happening before evaluation of the same tokens
- No clear separation between "graded" and "ungraded" token sets in the code

**Phase to address:**
Phase 5 (Evaluation Optimization) -- implement TTT only after the evaluation pipeline is fully validated and understood. Build with explicit auditability.

---

## Moderate Pitfalls

These cause wasted time and suboptimal results but don't cause disqualification.

### Pitfall 8: SWA Start Fraction Misconfiguration

**What goes wrong:**
Stochastic Weight Averaging (SWA) is a critical technique used by all top entries (e.g., start_frac=0.4 in the leading submission). But choosing the wrong start fraction wastes the benefits: starting too early averages over poorly-trained weights, starting too late doesn't average enough checkpoints. The standard recommendation (start at 75% of training) doesn't apply in this 10-minute budget regime.

**Why it happens:**
SWA literature recommends starting at 70-75% of training. But the competition leader uses 0.4 (40%), which is unusual. In the constrained 10-minute budget, the model underfits relative to a full training run, so starting SWA earlier captures more diverse weight snapshots that improve generalization. Teams who follow standard SWA guidance start too late and get minimal benefit.

**How to avoid:**
- Treat SWA start fraction as a critical hyperparameter to sweep: test 0.3, 0.4, 0.5, 0.6, 0.7.
- The leading entry uses 0.4, which is a strong prior for this regime.
- Monitor the SWA averaged model's BPB vs. the final model's BPB. If SWA hurts, the start fraction is wrong.
- If using batch normalization, ensure at least a few hundred steps after SWA starts to update BN running statistics.

**Warning signs:**
- SWA model performs worse than non-SWA model (start fraction too early or too late)
- Minimal BPB improvement from SWA (<0.005) -- likely misconfigured
- BN statistics not updated after SWA replacement

**Phase to address:**
Phase 2 (Training Optimization) -- include SWA start fraction in hyperparameter sweep grid.

---

### Pitfall 9: Multi-GPU Synchronization Overhead Eating Training Budget

**What goes wrong:**
With 8 GPUs and a small model (~15M parameters), the communication overhead of DDP (DistributedDataParallel) becomes a significant fraction of total training time. AllReduce gradient synchronization across 8 GPUs for a small model takes nearly as long as the forward+backward pass itself. Teams lose 20-30% of their 10-minute budget to communication, reducing effective training steps.

**Why it happens:**
DDP amortizes communication cost well for large models (gradients are large, compute is proportionally larger). For small models (~15M params), gradient tensors are small, but the overhead per AllReduce call is fixed (PCIe/NVLink latency). With ~20,000 training steps in 10 minutes, that is 20,000 AllReduce operations, each with fixed latency overhead. FSDP is even worse for small models (6x overhead vs DDP).

**How to avoid:**
- Use DDP, not FSDP, for these small models. FSDP's memory savings are irrelevant when the model fits easily in one GPU.
- Overlap computation with communication using DDP's gradient bucketing.
- Increase gradient accumulation steps to reduce AllReduce frequency while maintaining effective batch size.
- Profile communication vs. computation ratio using torch.profiler. Target < 15% communication overhead.
- Consider whether 8 GPUs actually helps -- sometimes fewer GPUs with larger per-GPU batch is faster for small models.

**Warning signs:**
- GPU utilization below 70% during training
- torch.profiler showing AllReduce as top time consumer
- Increasing GPU count doesn't proportionally increase throughput

**Phase to address:**
Phase 1 (Baseline) -- profile DDP overhead early and tune gradient accumulation/bucketing before optimizing the model itself.

---

### Pitfall 10: Optimizing One Dimension While Ignoring Interactions

**What goes wrong:**
A team spends two weeks perfecting int5 quantization, achieving great compression, but then discovers that their quantized weights don't work well with the BigramHash embedding layer they planned to add. Or they optimize the architecture for minimum parameters but then find SWA doesn't work well with their custom layer types. The dimensions of optimization (architecture, quantization, training, evaluation) interact nonlinearly.

**Why it happens:**
Each dimension is complex enough to absorb all available time. It's natural to go deep on one dimension. But this competition rewards full-stack optimization -- the leaderboard leaders combine ALL techniques: architecture (MLP3x, SmearGate), quantization (int5/int6 QAT), training (Muon, SWA, WD=0.04), and evaluation (sliding window). Optimizing one dimension to 95% while leaving others at 50% loses to someone at 80% on all dimensions.

**How to avoid:**
- Build a "technique stack" incrementally: start with the baseline, add one technique at a time, measuring marginal BPB improvement.
- Use the leaderboard leaders as a template: implement their full stack first, then innovate beyond it.
- Run interaction tests: when adding technique B to a model that already has technique A, measure both A+B and B alone. If A+B < A + B (individually), there is negative interaction.
- Allocate time budget across dimensions: ~25% architecture, ~25% quantization, ~25% training, ~25% evaluation.

**Warning signs:**
- Spending more than 40% of total project time on a single dimension
- No integration testing between independently developed components
- Marginal returns on the current dimension dropping below 0.001 BPB per day of effort

**Phase to address:**
Phase 2 (Training) through Phase 5 -- maintain a cross-dimension integration test suite throughout development.

---

### Pitfall 11: Learning Rate Sensitivity at Extreme Quantization

**What goes wrong:**
The learning rate that works for float32/bfloat16 training causes divergence or severe underfitting when QAT is enabled. At int5/int6 quantization, gradients through the STE are biased, and the effective landscape changes dramatically. The Muon optimizer, already aggressive with its orthogonal Newton-like updates, becomes even more sensitive when combined with quantized gradients.

**Why it happens:**
QAT changes the effective loss landscape. The STE passes gradients as if the rounding didn't happen, but the forward pass uses discrete values. This mismatch means the gradient points in a slightly wrong direction, and larger learning rates amplify this error. The Muon optimizer compounds this because it uses matrix orthogonalization (Newton-Schulz iterations), which can project QAT gradients into unhelpful directions.

**How to avoid:**
- When enabling QAT, start with a learning rate 2-5x lower than the non-QAT baseline.
- Use a separate learning rate for the quantization scale parameters vs. the weights themselves.
- Implement learning rate warmup specifically for the QAT phase.
- Test the Muon+QAT combination carefully; Adam may be more stable for the QAT phase even if Muon is better for full-precision training.

**Warning signs:**
- Loss diverges within first 100 steps of QAT
- Loss flatlines (gradients too small to change quantization bins)
- Large discrepancy between full-precision loss and quantized-inference loss

**Phase to address:**
Phase 3 (Quantization) -- dedicated learning rate sweep when introducing QAT.

---

### Pitfall 12: Data Ordering Effects Amplified in Small Models

**What goes wrong:**
Small models (14M-70M parameters) are disproportionately sensitive to data ordering. Random shuffling produces high gradient noise, slowing convergence. With only 10 minutes of training, every step counts, and noisy gradients from random ordering waste precious steps.

**Why it happens:**
Research (2025) confirms that "curriculum gains are scale-dependent and most pronounced for capacity-constrained models." Small models have less capacity to absorb batch-level variance, so gradient noise from random data ordering is proportionally larger. The gradient noise scale for smaller models (14M-70M) is measurably higher with random ordering vs. curriculum-based ordering. In the 10-minute regime, 18-45% fewer steps to reach baseline performance translates to meaningful BPB improvement.

**How to avoid:**
- Experiment with curriculum learning: start with shorter, simpler documents and progress to longer, more complex ones.
- Use compression ratio, lexical diversity, or readability as difficulty signals for ordering.
- Apply curriculum as a warmup strategy: ordered data for the first 30% of training, then switch to random.
- At minimum, ensure consistent shuffling across seeds (use seeded RNG for data ordering).

**Warning signs:**
- High variance in per-step loss (gradient noise)
- Slow convergence in early training steps
- Large BPB differences across seeds with identical configurations (data ordering as the only variable)

**Phase to address:**
Phase 2 (Training Optimization) -- include data ordering experiments in the training pipeline sweep.

---

### Pitfall 13: Sliding Window Eval Implementation Errors

**What goes wrong:**
Sliding window evaluation (used by top entries with stride=64) is implemented incorrectly, leading to either inflated BPB scores (some tokens evaluated with too little context) or deflated BPB scores (some tokens counted multiple times or context leaking). The stride, context length, and target masking must be precisely coordinated.

**Why it happens:**
In sliding window evaluation, tokens near the start of each window have less context than tokens near the end. The correct approach is to only score tokens in the latter portion of each window (setting targets for context tokens to -100 so they are ignored in the loss). Implementation errors include: scoring all tokens in every window (double-counting), using the wrong stride (missing tokens or excessive overlap), and not properly handling the final partial window.

**How to avoid:**
- Use the Hugging Face documentation pattern: set targets to -100 for tokens that serve only as context (not scored).
- Verify that every token in the validation set is scored exactly once.
- Cross-check: sum of scored tokens across all windows should equal total validation tokens.
- Compare sliding window BPB against non-sliding evaluation -- sliding should be better (more context), not dramatically different (implementation bug if >0.05 difference).

**Warning signs:**
- Total scored tokens != total validation tokens
- BPB dramatically better with sliding window (>0.03 improvement per stride reduction)
- Different strides giving inconsistent results

**Phase to address:**
Phase 4 (Evaluation Optimization) -- implement and validate sliding window with extensive assertion checking.

---

## Minor Pitfalls

These cause annoyance and small efficiency losses.

### Pitfall 14: torch.compile Warmup Eating Training Time

**What goes wrong:**
torch.compile's first execution triggers compilation, which can take 30-60 seconds for a complex model. In a 10-minute training budget, losing 60 seconds to compilation is a 10% training time loss. The baseline handles this with explicit warmup steps, but teams adding custom layers or changing compilation flags may trigger recompilation.

**Why it happens:**
torch.compile traces the computation graph on first execution. Graph breaks (caused by data-dependent control flow, unsupported operations, or Python-level logic) cause multiple separate compilations. Custom operations not supported by the compiler trigger fallbacks or recompilation. Adding `torch.compile(mode="reduce-overhead")` changes the compilation strategy and may increase warmup time.

**How to avoid:**
- Include explicit warmup steps (as the baseline does) and measure their cost.
- Avoid graph breaks: no data-dependent control flow in the model, no Python print statements in the forward pass.
- Cache compiled models across runs during development (but note: competition evaluation starts fresh).
- Profile compilation time separately from training time.

**Warning signs:**
- First training step taking 10x+ longer than subsequent steps
- torch._dynamo warnings about graph breaks in logs
- Training wall-clock significantly exceeding expected time

**Phase to address:**
Phase 1 (Baseline) -- measure and budget for compilation overhead from the start.

---

### Pitfall 15: NFS Storage Filling Up During Experiments

**What goes wrong:**
Home directory on the SLURM cluster fills up (currently 69GB used), causing training jobs to crash with I/O errors. Checkpoints, datasets, wandb logs, and intermediate artifacts accumulate silently. NFS near quota causes serialized writes, which can hang the entire NFS server and affect other users.

**Why it happens:**
ML experiments generate large artifacts: model checkpoints (tens of MB each), training logs, downloaded datasets (FineWeb is multi-GB), cached tokenized data, and experiment tracking files. With aggressive experimentation (dozens of runs per day), storage accumulates quickly. NFS doesn't provide immediate feedback about approaching quota -- the failure is sudden.

**How to avoid:**
- Set up automatic checkpoint cleanup: keep only the best and most recent 3 checkpoints.
- Use local scratch storage (not NFS) for intermediate data and checkpoints during training. Copy only final artifacts to NFS.
- Monitor disk usage daily: `du -sh ~/parameter_golf/*/` and `quota -s`.
- Use hard-linking for dataset files when cache and working directory are on the same filesystem (the download script supports this).
- Clean up wandb/experiment tracking directories regularly.

**Warning signs:**
- `du -sh ~` showing >80% of quota
- Training jobs failing with "No space left on device" or "Disk quota exceeded"
- NFS performance degrading (slow `ls`, slow file operations)

**Phase to address:**
Phase 1 (Baseline) -- establish storage management protocol and monitoring from the start. Add cleanup to experiment scripts.

---

### Pitfall 16: Script Doesn't Run in Evaluation Environment

**What goes wrong:**
The training script works on the local SLURM cluster (H200, specific CUDA version, specific Python packages) but fails on RunPod's 8xH100 evaluation environment. Missing dependencies, wrong PyTorch version, CUDA version mismatch, or hardcoded paths cause the submission to fail.

**Why it happens:**
The competition requires torch==2.10 specifically. The local cluster may have a different version. Custom packages installed in the local environment may not be available on RunPod. Hardcoded paths (like `/orcd/home/002/tomli/...`) or cluster-specific environment variables break on a different machine. The RunPod template (y5cejece4j) has a specific set of pre-installed packages.

**How to avoid:**
- Pin all dependencies in requirements.txt with exact versions, including torch==2.10.
- Test early on RunPod (not just at submission time).
- Avoid hardcoded paths; use relative paths or environment variables.
- Test with a clean virtual environment locally before submitting.
- Keep the package list minimal -- fewer dependencies means fewer compatibility issues.

**Warning signs:**
- import errors when running in a different environment
- Code using features from torch > 2.10 or < 2.10
- Hardcoded absolute paths in the training script

**Phase to address:**
Phase 1 (Baseline) -- test on RunPod early to establish environment compatibility.

---

### Pitfall 17: Not Doing Proper Ablations (Can't Tell What Helps)

**What goes wrong:**
Teams implement 5 techniques simultaneously, achieve a good BPB, but cannot determine which techniques actually helped. When a later change regresses BPB, they cannot identify the interaction. They end up unable to make informed decisions about what to keep, what to remove, and what to improve.

**Why it happens:**
In a time-pressured competition, it feels wasteful to spend GPU hours on ablation studies rather than trying new ideas. But without ablations, every decision is a guess. The interaction effects between techniques (quantization x architecture x training schedule) are not additive -- some combinations help while others cancel out.

**How to avoid:**
- Implement a systematic ablation framework: one baseline, add one technique at a time, measure marginal contribution.
- Log every experiment with full configuration (hyperparameters, techniques enabled/disabled, seed, BPB result).
- Budget 30% of GPU hours for ablation studies, 70% for exploration.
- Use the L40S nodes for parallel ablation sweeps (less powerful but more available).
- Maintain a "technique contribution table": technique name, marginal BPB improvement, confidence level.

**Warning signs:**
- Cannot answer "how much does technique X contribute?" for any technique in the stack
- Removing a technique causes unexpected improvement (negative interaction was masked)
- Different team members have conflicting beliefs about what works

**Phase to address:**
Phase 2 (Training) -- establish ablation protocol as part of the experiment framework. Every new technique must be ablated before being included in the main stack.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip multi-seed validation | Saves 2x GPU time | False confidence in improvements; wasted effort on noise | Never for submission candidates; OK for early exploration |
| Hardcode hyperparameters | Faster iteration | Cannot sweep or reproduce; locks in suboptimal values | Never; always use env vars or config |
| Estimate compression ratio | Skip compression step | Artifact size surprise at submission | Never; always measure exact sizes |
| Skip RunPod validation | Save $20-50 in RunPod costs | Submission fails timing check | OK during early exploration; never for submission candidates |
| Copy-paste from leaderboard code without understanding | Quick baseline | Cannot modify or debug; black-box behavior | OK as starting point; must understand before modifying |
| Train with bfloat16, quantize at end (no QAT) | Simpler training | ~0.01-0.03 BPB loss from post-training quantization vs QAT | Only for initial baseline; switch to QAT for competitive results |

## Integration Gotchas

Common mistakes when combining components of the competition stack.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Muon + QAT | Using Muon's default learning rate with QAT, causing instability | Reduce Muon LR by 2-5x when QAT is active; consider Adam for QAT phase |
| SWA + Custom Layers | SWA averages all parameters including non-learnable ones (e.g., quantization scales) | Explicitly specify which parameters to average; exclude fixed scales |
| BigramHash + Custom Tokenizer | BigramHash trained on sp1024 tokenizer's bigram distribution; changing tokenizer invalidates hash table | Retrain BigramHash when changing tokenizer; verify hash collision rates |
| Sliding Window Eval + TTT | TTT updates model during sliding window, changing predictions for overlapping tokens | TTT must only update after tokens are scored; sliding window must not re-score |
| Int5 + Zstd Compression | Assuming 5/8 ratio (5 bits per weight vs 8) after compression | Int5 weights may compress to >5 bits effective due to entropy; measure actual compressed size |
| DDP + torch.compile | Compilation happens per-rank, wasting time if all ranks compile the same graph | Ensure compilation cache is shared or compile only on rank 0 and broadcast |

## Performance Traps

Patterns that seem efficient but waste the 10-minute training budget.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Too-large batch size | GPU memory full but throughput low; each step slow | Profile tokens/second at different batch sizes; optimal is usually 50-70% memory utilization | When batch size exceeds L2 cache working set |
| Excessive validation frequency | Losing 5-10% of training time to eval | Validate every 1000-2000 steps, not every 100; final full eval at end | When validation takes >2% of total training time |
| Unnecessary gradient accumulation | Same effective batch as fewer accumulation steps with larger per-step batch | Calculate: if per-GPU batch can be increased, reduce accumulation steps | When gradient accumulation steps > 4 with unused GPU memory |
| Logging overhead | Wandb/tensorboard sync blocking training | Use async logging; log scalars only, not histograms; log every N steps | When logging frequency > 1% of step time |
| Re-downloading dataset each run | Network I/O wasting training time | Cache dataset on local scratch; verify cache before download | When dataset not pre-cached in evaluation environment |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Artifact size check:** Often missing code bytes in the total -- verify code_bytes + model_bytes < 16,000,000
- [ ] **BPB after roundtrip:** Often reporting pre-quantization BPB -- verify BPB AFTER quantize + compress + decompress + dequantize
- [ ] **Multi-seed validation:** Often reporting single best seed -- verify mean BPB across 3+ seeds with p < 0.01
- [ ] **H100 timing:** Often reporting H200 timing -- verify actual wall-clock on RunPod 8xH100
- [ ] **SWA BN update:** Often enabling SWA but not updating batch norm stats -- verify BN running mean/var are recalculated
- [ ] **Submission format:** Often missing required files -- verify README.md, submission.json, train.log, train_gpt.py, final_model artifact all present
- [ ] **requirements.txt:** Often missing pinned versions -- verify all imports are listed with exact version pins
- [ ] **TTT compliance:** If using TTT, verify no tokens are influenced by future token information
- [ ] **Statistical significance:** Often reporting point estimate -- verify p < 0.01 with >=0.005 nat improvement over SOTA

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Artifact over 16MB | LOW | Reduce model dim by 16-32; remove one layer; switch to more aggressive quantization |
| H100 timing > 10min | MEDIUM | Reduce iterations; increase learning rate; reduce sequence length; cut one layer |
| QAT instability | MEDIUM | Revert to int6; reduce LR by 5x; add gradient clipping; try Adam instead of Muon for QAT |
| BPB calculation bug | LOW | Revert to baseline eval_val(); compare against known-correct BPB values |
| Non-reproducible results | MEDIUM | Enable deterministic mode; fix all seeds; run 5 seeds; if still noisy, the technique is unstable |
| TTT rule violation | HIGH | Remove TTT entirely; redesign with strict token accounting; get external review |
| Disk full | LOW | Clean checkpoints, caches; use scratch storage; compress logs |
| RunPod incompatibility | MEDIUM | Test on RunPod immediately; pin all deps; remove hardcoded paths |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Artifact size miscalculation | Phase 1: Baseline | Automated size check in every training run |
| H200/H100 timing gap | Phase 1: Baseline | RunPod validation run before any submission |
| QAT instability | Phase 3: Quantization | Per-layer gradient norm monitoring; stability metrics |
| Poor compression of quantized weights | Phase 3: Quantization | Compressed artifact size measured for every quantization experiment |
| BPB calculation bugs | Phase 1: Baseline (lock down), Phase 5: Tokenizer (re-validate) | Cross-check BPB against manual byte counts |
| Non-reproducibility | Phase 1: Baseline | Multi-seed protocol; std < 0.002 across seeds |
| TTT rule violation | Phase 5: Evaluation | Token accounting audit trail |
| SWA misconfiguration | Phase 2: Training | SWA vs non-SWA comparison; start fraction sweep |
| Multi-GPU overhead | Phase 1: Baseline | torch.profiler showing < 15% communication overhead |
| Single-dimension optimization | Phase 2-5: All | Cross-dimension integration tests; technique contribution table |
| LR sensitivity with QAT | Phase 3: Quantization | LR sweep specific to QAT; divergence detection |
| Data ordering effects | Phase 2: Training | Curriculum vs random comparison; gradient noise measurement |
| Sliding window eval errors | Phase 4: Evaluation | Token count assertions; comparison with non-sliding eval |
| torch.compile warmup | Phase 1: Baseline | Compilation time measurement and budgeting |
| NFS storage overflow | Phase 1: Baseline | Daily quota monitoring; automatic cleanup |
| Environment incompatibility | Phase 1: Baseline | Early RunPod test; requirements.txt pinning |
| Missing ablations | Phase 2: Training | Technique contribution table maintained throughout |

## Sources

- [OpenAI Parameter Golf GitHub Repository](https://github.com/openai/parameter-golf) -- competition rules, baseline code, submission format [HIGH confidence]
- [DeepWiki - Parameter Golf](https://deepwiki.com/openai/parameter-golf) -- comprehensive competition documentation [HIGH confidence]
- [DeepWiki - Evaluation Metrics](https://deepwiki.com/openai/parameter-golf/3.2-evaluation-metrics) -- BPB calculation details [HIGH confidence]
- [StableQAT: Stable Quantization-Aware Training at Ultra-Low Bitwidths](https://arxiv.org/abs/2601.19320) -- STE gradient instability, RDFS solution [HIGH confidence]
- [Understanding Straight-Through Estimator](https://arxiv.org/abs/1903.05662) -- STE bias analysis [HIGH confidence]
- [ZipNN: Lossless Compression for AI Models](https://arxiv.org/html/2411.05239v2) -- compression challenges for quantized weights [HIGH confidence]
- [QStore: Quantization-Aware Compressed Model Storage](https://arxiv.org/pdf/2505.04081) -- entropy coding for quantized weights [MEDIUM confidence]
- [PyTorch Reproducibility Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html) -- CUDA nondeterminism, deterministic mode [HIGH confidence]
- [SWA in PyTorch Blog](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/) -- SWA scheduling, BN update requirements [HIGH confidence]
- [Strategic Data Ordering for LLMs](https://arxiv.org/abs/2405.07490) -- curriculum learning benefits for small models [MEDIUM confidence]
- [Curriculum Learning for LLM Pretraining](https://arxiv.org/html/2601.21698v1) -- scale-dependent curriculum effects [MEDIUM confidence]
- [HuggingFace Perplexity Documentation](https://huggingface.co/docs/transformers/perplexity) -- sliding window evaluation methodology [HIGH confidence]
- [Practical Efficiency of Muon for Pretraining](https://arxiv.org/pdf/2505.02222) -- Muon + weight decay interaction [HIGH confidence]
- [NVIDIA H100 vs H200 Benchmarks](https://greennode.ai/blog/compare-h100-vs-h200) -- performance difference analysis [MEDIUM confidence]
- [Parameter Golf Leaderboard](https://parameter-golf.github.io/) -- current SOTA techniques and scores [HIGH confidence]

---
*Pitfalls research for: OpenAI Parameter Golf -- parameter-constrained LM competition*
*Researched: 2026-03-22*

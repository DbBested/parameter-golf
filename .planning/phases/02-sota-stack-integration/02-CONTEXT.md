# Phase 2: SOTA Stack Integration - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement the full proven SOTA technique stack into a single train_gpt.py, achieving ~1.15 BPB on FineWeb validation. This includes: 10-layer transformer (512 dim, MLP 3x relu-squared, GQA 8Q/4KV), BigramHash(10240) embeddings, Muon+AdamW optimizer, SWA, int6 QAT with STE, zstd-22 compression, sliding window evaluation, and weight decay 0.04. Each technique's marginal BPB contribution must be measured via ablation.

</domain>

<decisions>
## Implementation Decisions

### Model Architecture
- Modify baseline train_gpt.py in-place (competition requires single script submission)
- Implement BigramHash as a separate module class integrated into the model for clean ablation toggling
- Incremental implementation order: architecture changes first (10L, GQA, MLP3x relu-squared), then training improvements (Muon, SWA, WD), then quantization (int6 QAT), then eval (sliding window) — each verified via ablation
- Use the standalone Muon optimizer already in train_gpt.py (PyTorch 2.6 lacks native torch.optim.Muon)

### Quantization & Compression
- Int6 QAT with per-row scaling (matches SOTA submissions for better accuracy at small model size)
- zstd compression level 22 (all submissions use this — known optimal)
- QAT from training start (model learns to compensate for quantization noise)

### Training & Evaluation Pipeline
- Custom SWA: save 20-24 checkpoints from last 40% of warmdown, average offline (matches competition pattern)
- Sliding window eval integrated into train_gpt.py's eval function (competition evaluates via training script)
- Everything in single train_gpt.py — use clean function organization internally

### Claude's Discretion
- Exact learning rate schedule and warmup steps
- Specific batch size tuning for H200 memory optimization
- RoPE frequency base selection
- Initialization scheme details (before OrthoInit in Phase 4)
- Exact SWA checkpoint collection interval

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `repo/train_gpt.py` — Official baseline with 9L/512dim/1024vocab, Muon optimizer already implemented, DDP training, int8 quantization + zlib compression
- `scripts/eval/check_artifact_size.py` — Artifact size validation
- `scripts/eval/validate_bpb.py` — BPB validation
- `scripts/eval/ablation_runner.py` — Ablation framework for toggling techniques
- `scripts/slurm/train_baseline.sbatch` — SLURM job template for 8xH200
- `scripts/tracking/log_experiment.py` — Experiment logging

### Established Patterns
- DDP via torchrun --standalone --nproc_per_node=8
- BPB calculated via official eval_val function with byte-level accounting
- Model serialization: int8 + zlib/zstd compression
- Experiment tracking via JSON-lines in experiments/results/

### Integration Points
- Baseline BPB reference: 1.2254 ± 0.0007 (3-seed mean)
- Artifact size budget: 16,000,000 bytes (baseline uses 15.88MB)
- H200 timing: 600s wall cap, step avg 44.83ms
- conda env pgolf with PyTorch 2.6.0+cu124

</code_context>

<specifics>
## Specific Ideas

- Start from the existing SOTA submission code on the leaderboard (e.g., thwu1's 1.1428 submission) as reference, not the naive baseline
- The baseline train_gpt.py already has Muon implemented — extend it rather than replacing
- Key SOTA hyperparameters from leaderboard: matrix_lr=0.02, weight_decay=0.04, momentum=0.99, SWA start at 40% warmdown
- BigramHash(10240) hashes consecutive token pairs into 10240 buckets with dim=128, projects via learned linear to model dim

</specifics>

<deferred>
## Deferred Ideas

- Int5 quantization (Phase 3)
- SmearGate, U-Net skips, OrthoInit (Phase 4)
- Depth recurrence, TTT, curriculum learning (Phase 5)

</deferred>

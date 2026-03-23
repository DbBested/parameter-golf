# Phase 5: Moonshot Exploration - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Explore 4 independent moonshot directions on top of the current competitive base model (BPB 1.1421, artifact 15.82MB). Each moonshot is an independent experiment with a clear go/no-go gate. Target: at least one technique delivers measurable improvement, pushing BPB below 1.135. The best configuration becomes the submission model.

Current base model:
- BPB: 1.1421 (beats current SOTA 1.1428)
- Artifact: 15,824,167 bytes (176KB headroom under 16MB)
- Architecture: 10L/512d, MLP 3x relu², GQA 8Q/4KV, SmearGate, BigramHash(10240), U-Net skips, OrthoInit
- Training: Muon+AdamW WD=0.04, SWA 24 checkpoints, 600s wall cap, ~86ms/step
- Quantization: int5 MLP, int6 attention, FP16 embeddings, 5% pruning, zstd-22

</domain>

<decisions>
## Implementation Decisions

### Moonshot 1: Depth Recurrence (RECUR-01 through RECUR-04)
- Implement layer reuse: 5-6 unique transformer layers looped 2-3x for 15-18 effective layers
- Per-recurrence-step adaptation via small learned scaling factors
- Key benefit: shared weights compress much better (identical params → perfect zstd compression)
- Risk: training stability through loops, quality at this small scale
- Go/no-go: must beat non-recurrent model at same artifact size

### Moonshot 2: Test-Time Training (TTT-01 through TTT-04)
- LoRA adaptation during evaluation on already-evaluated tokens
- Rules explicitly allow TTT on tokens already graded
- Apply LoRA to attention layers, fine-tune with next-token prediction loss
- Key benefit: adapts to validation data distribution at eval time
- Risk: eval time budget (must complete within 10 minutes)
- Go/no-go: must provide >0.002 BPB improvement

### Moonshot 3: Curriculum Learning (CURR-01 through CURR-03)
- Order training data by difficulty (compression ratio as difficulty metric)
- Progress from easy to hard during training
- Key benefit: faster convergence → more effective steps in 600s budget
- Risk: precomputing difficulty metrics adds preprocessing time
- Go/no-go: must provide measurable BPB improvement vs random ordering

### Moonshot 4: Sub-5-bit Quantization (LOWBIT-01 through LOWBIT-03)
- Int4 for MLP layers (clip_range=7), keep int5/int6 for sensitive layers
- Key benefit: saves ~1-2MB artifact space → can add more parameters
- Risk: quality degradation at int4
- Go/no-go: must achieve lower BPB than int5 model at same or smaller artifact

### Execution Strategy
- Each moonshot is independent — can be developed and tested in parallel
- Use separate branches or experiment IDs for each
- Run 1-seed quick tests first, 3-seed validation only on promising results
- Time-box each moonshot: if no improvement after 2-3 runs, move to go/no-go

### Claude's Discretion
- Exact LoRA rank and learning rate for TTT
- Depth recurrence layer grouping strategy
- Curriculum difficulty metric implementation details
- Int4 layer selection criteria

</decisions>

<code_context>
## Existing Code Insights

### Base Model
- repo/train_gpt.py — fully optimized SOTA implementation (54KB, 1259 lines)
- All techniques from Phases 2-4 integrated

### Experiment Infrastructure
- scripts/eval/ablation_runner.py — technique toggle framework
- scripts/eval/multiseed_eval.py — multi-seed evaluation
- scripts/tracking/log_experiment.py — experiment logging
- scripts/slurm/ — SLURM job templates

### Compute Resources
- H200 nodes: node4300, node4301 (8xH200 each)
- L40S nodes: 15 nodes × 4xL40S for parallel experiments
- Max job time: 48 hours

</code_context>

<specifics>
## Specific Ideas

- Depth recurrence: Huginn-3.5B shows 4 recurrent blocks matching 12+ unique layers. At our scale, try 5 unique layers × 2 loops = 10 effective layers with ~half the params
- TTT: LoRA rank 4-8 should be sufficient. Apply to Q/K projections only. Learning rate ~1e-4 for eval-time adaptation
- Curriculum: Use zlib compression ratio of each training sequence as difficulty metric — precomputable, cheap
- Sub-5-bit: Start with int4 for MLP only (most tolerant), measure BPB impact

</specifics>

<deferred>
## Deferred Ideas

- MoE with shared parameters (v2 — too complex for time remaining)
- Neural Architecture Search (v2 — search space too narrow at this point)
- Novel tokenizer design (v2 — tokenizer changes are risky per competition rules)

</deferred>

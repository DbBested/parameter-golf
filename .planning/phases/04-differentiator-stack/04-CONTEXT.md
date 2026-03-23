# Phase 4: Differentiator Stack - Context

**Gathered:** 2026-03-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Apply architecture differentiators (U-Net skip connections — SmearGate and OrthoInit already implemented in Phase 2), ablate each technique individually, and critically: **get the artifact under the 16MB cap** (currently 173KB over at 16,173,399 bytes). The best combination of differentiators should achieve BPB <= 1.145 within the 16MB artifact budget.

Current state:
- BPB: 1.1424 (from Phase 3 int5 MLP run)
- Artifact: 16,173,399 bytes (173KB over 16MB cap)
- SmearGate: already implemented (Phase 2)
- OrthoInit: already implemented (Phase 2)
- U-Net skips: NOT yet implemented
- Code size: 62,489 bytes (potentially shrinkable)

</domain>

<decisions>
## Implementation Decisions

### Artifact Size Fix (CRITICAL — must fit under 16MB)
- Increase magnitude pruning from 3% to 5% (saves compression budget)
- Minify/strip comments from train_gpt.py code (reduce 62KB code size)
- If still over: reduce BigramHash buckets from 10240 to 8192 (saves ~200KB compressed)
- Target: artifact < 15,500,000 bytes (500KB safety margin)

### U-Net Skip Connections
- Add cross-layer residual connections (early layers → late layers)
- Follow the pattern from SOTA #1 reference if present, or implement standard U-Net skip
- Small parameter overhead for connection weights

### Ablation Framework
- Use the ablation runner from Phase 1 to toggle each differentiator
- Measure marginal BPB contribution of: SmearGate, OrthoInit, U-Net skips, pruning level
- Record results in experiment tracker

### Claude's Discretion
- Exact U-Net skip connection architecture (which layers connect to which)
- Code minification strategy (strip comments, shorten variable names, or external minifier)
- Whether to run ablation on all differentiators or prioritize the artifact size fix

</decisions>

<code_context>
## Existing Code Insights

### Current Architecture in repo/train_gpt.py
- SmearGate: class at ~line 798, called in forward at ~line 933
- BigramHash: class at ~line 810, called in forward at ~line 931
- OrthoInit: in _init_weights method
- Mixed int5/int6 PTQ: in mixed_quantize_int6 (clip=15 for MLP, 31 for attn)
- 3% magnitude pruning: in serialization path
- SWA: 24 checkpoints from warmdown
- Sliding window eval: stride=64

### SOTA Reference
- repo/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py

### Integration Points
- Artifact budget: 16,000,000 bytes total (code + compressed model)
- H200 training: 600s wall cap, ~86ms/step
- Ablation framework: scripts/eval/ablation_runner.py

</code_context>

<specifics>
## Specific Ideas

- The SOTA #1 reference shows the leader's code is ~61KB — ours at 62KB is comparable, but stripping docstrings could save 5-10KB
- Increasing pruning from 3% to 5% should save ~100-200KB compressed with minimal BPB impact
- If U-Net skips don't help BPB enough, don't include them (they add params)

</specifics>

<deferred>
## Deferred Ideas

- Depth recurrence, TTT, curriculum learning (Phase 5)
- Sub-5-bit quantization (Phase 5)

</deferred>

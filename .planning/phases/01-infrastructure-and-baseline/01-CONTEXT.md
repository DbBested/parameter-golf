# Phase 1: Infrastructure and Baseline - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — discuss skipped)

<domain>
## Phase Boundary

Trustworthy measurement infrastructure exists and the official baseline (1.2244 BPB) is reproduced on pg_tata, so every subsequent experiment has a reliable reference point. This phase delivers: working training pipeline on pg_tata H200 cluster, automated artifact size checking, multi-seed evaluation protocol, H200-to-H100 timing calibration, experiment tracking, ablation framework, and storage management.

</domain>

<decisions>
## Implementation Decisions

### Claude's Discretion
All implementation choices are at Claude's discretion — pure infrastructure phase. Use ROADMAP phase goal, success criteria, and codebase conventions to guide decisions.

Key infrastructure decisions:
- Clone the official parameter-golf repo and adapt the baseline train_gpt.py
- Use SLURM job scripts for pg_tata with `--partition=pg_tata --gres=gpu:h200:8`
- Experiment tracking via lightweight JSON/CSV logs (no heavyweight MLflow/W&B to avoid storage bloat)
- Ablation framework as shell scripts or Python wrappers that toggle technique flags
- Storage management: auto-cleanup of checkpoints older than N runs, keep only best + latest

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Official parameter-golf repo baseline: train_gpt.py, data loading scripts, tokenizer
- FineWeb cached dataset download scripts
- Official evaluation methodology in train_gpt.py

### Established Patterns
- Competition uses torchrun for DDP, PyTorch native training loop
- BPB calculation: total_loss / ln(2) / total_bytes
- Artifact size: code bytes + zstd/zlib compressed model bytes

### Integration Points
- pg_tata SLURM cluster (H200 nodes: node4300, node4301)
- Data stored in home directory with storage constraints (~69GB used)
- Git for version control of experiments and planning docs

</code_context>

<specifics>
## Specific Ideas

No specific requirements — infrastructure phase. Refer to ROADMAP phase description and success criteria.

</specifics>

<deferred>
## Deferred Ideas

None — infrastructure phase.

</deferred>

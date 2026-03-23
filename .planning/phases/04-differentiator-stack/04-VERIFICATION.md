---
phase: 04-differentiator-stack
verified: 2026-03-23T07:44:07Z
status: gaps_found
score: 2/3 must-haves verified
gaps:
  - truth: "Each differentiator (SmearGate, U-Net skips, OrthoInit) has a measured marginal BPB contribution"
    status: failed
    reason: "No ablation runs were executed. The ablation_runner.py exists and supports toggling techniques via environment variables, but train_gpt.py has no ENABLE_* toggle flags, and experiments/results/experiments.jsonl contains zero ablation entries. ROADMAP success criteria 1 and 2 both require individual ablation measurements that do not exist."
    artifacts:
      - path: "scripts/eval/ablation_runner.py"
        issue: "Runner exists and is substantive (369 lines) but is wired via ENABLE_<TECHNIQUE> env vars that train_gpt.py does not honour — no toggle mechanism implemented in the model code"
      - path: "experiments/results/experiments.jsonl"
        issue: "Contains only 3 baseline entries (Phase 1); no ablation entries for SmearGate, U-Net skips, or OrthoInit"
    missing:
      - "Toggle flags in train_gpt.py (e.g. --no-smear, --no-skip, --no-ortho CLI args or ENABLE_* env checks) so individual differentiators can be disabled"
      - "At minimum 3 ablation runs: baseline-without-SmearGate, baseline-without-U-Net-skips, baseline-without-OrthoInit"
      - "Recorded marginal BPB contribution for each differentiator in experiments.jsonl"
      - "ROADMAP success criterion 1: SmearGate BPB improvement measured > 0.001"
      - "ROADMAP success criterion 2: per-differentiator marginal contribution measured and documented"
---

# Phase 4: Differentiator Stack Verification Report

**Phase Goal (ROADMAP):** Verify all differentiator techniques (SmearGate, U-Net skips, OrthoInit) are present, fix the 173KB artifact size overage, and validate the model fits under 16MB with BPB <= 1.145

**Verified:** 2026-03-23T07:44:07Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

The ROADMAP specifies three success criteria for Phase 4. These are treated as the authoritative must-haves.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SmearGate provides measurable BPB improvement (>0.001) when ablated individually | FAILED | No ablation run exists. experiments.jsonl has zero ablation entries. train_gpt.py has no toggle mechanism. |
| 2 | Each differentiator has a measured marginal BPB contribution and artifact size impact | FAILED | Same root cause: no ablation infrastructure wired to train_gpt.py, no ablation runs executed. |
| 3 | Best combination achieves BPB <= 1.145 within artifact size budget | VERIFIED | SLURM log confirms: val_bpb=1.1421, artifact=15,824,167 bytes (176KB headroom). PASS in SLURM output. |

**Score:** 1/3 success criteria verified per ROADMAP.

The PLAN frontmatter defines a separate, narrower set of must-haves focused on code size and differentiator presence (not ablation). Against those plan-level truths: 5/5 verified. The gap is between the PLAN scope and the ROADMAP's stated success criteria.

### Required Artifacts

| Artifact | Expected | Exists | Substantive | Wired | Status |
|----------|----------|--------|-------------|-------|--------|
| `repo/train_gpt.py` | Complete training script with all differentiators | YES (53,995 B, 1,259 lines) | YES | YES | VERIFIED |
| `scripts/slurm/train_phase4.sbatch` | SLURM job script for Phase 4 validation | YES | YES (46 lines) | YES | VERIFIED |
| `logs/phase4_seed1337_20260323_032137.log` | Training log from Phase 4 run | YES | YES | YES | VERIFIED |
| `logs/slurm/10822629-phase4.out` | SLURM output with BPB + artifact size | YES | YES | YES | VERIFIED |
| Ablation results in `experiments/results/experiments.jsonl` | Per-differentiator BPB measurements | NO ENTRIES | — | — | MISSING |
| Toggle flags in `repo/train_gpt.py` | ENABLE_*/--no-X flags for ablation | NOT IMPLEMENTED | — | — | MISSING |

### Key Link Verification

| From | To | Via | Status | Evidence |
|------|----|-----|--------|----------|
| SmearGate class (line 697) | GPT.forward() (line 822) | `self.smear(x)` at line 827 | WIRED | `grep "self.smear("` returns lines 827, 853 |
| SmearGate class (line 697) | GPT.forward_logits() (line 848) | `self.smear(x)` at line 853 | WIRED | Both forward paths confirmed |
| skip_weights parameter (line 787) | GPT.forward() decoder loop | `skip_weights[i] * skips.pop()` at line 835 | WIRED | encoder push at 832, decoder pop at 835 |
| skip_weights parameter (line 787) | GPT.forward_logits() decoder loop | `skip_weights[i] * skips.pop()` at line 861 | WIRED | encoder push at 858, decoder pop at 861 |
| _init_weights (line 808) | nn.init.orthogonal_ | `module.weight.shape[0] >= 64 and shape[1] >= 64` at line 816 | WIRED | Line 817: `nn.init.orthogonal_(module.weight, gain=1.0)` |
| pruning threshold | zstd compression | `torch.quantile(..., 0.05)` at line 1201 | WIRED | Confirmed 5% threshold |
| ablation_runner.py | train_gpt.py differentiator toggles | ENABLE_* env vars | NOT WIRED | train_gpt.py has no ENABLE_SMEAR/ENABLE_SKIP/ENABLE_ORTHO env var checks |

### Requirements Coverage

All three requirement IDs from the PLAN frontmatter are mapped to Phase 4 in REQUIREMENTS.md (traceability table, lines 128-130).

| Requirement | Description | Status | Blocking Issue |
|-------------|-------------|--------|----------------|
| ARCH-05 | SmearGate gating mechanism integrated after attention | SATISFIED — code verified | Class exists (line 697), called in both forward paths (lines 827, 853), substantive implementation (sigmoid gate + causal smear) |
| ARCH-06 | U-Net skip connections linking early layers to late layers | SATISFIED — code verified | `num_encoder_layers`, `num_decoder_layers`, `skip_weights` in __init__ (lines 784-787); push/pop in both forward paths (lines 830-835, 856-861) |
| ARCH-07 | Orthogonal initialization for all weight matrices | SATISFIED — code verified | `_init_weights` applies `nn.init.orthogonal_` to all `nn.Linear` layers with both dimensions >= 64 (lines 816-817) |

ARCH-05, ARCH-06, ARCH-07 requirements as stated in REQUIREMENTS.md are all met at the code level. The gap is at the ROADMAP success-criteria level (ablation measurements), which is a stronger claim than the requirement text.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `repo/train_gpt.py` | — | No ablation toggle flags | Blocker (for ROADMAP SC 1-2) | Cannot disable individual differentiators to measure marginal contribution without modifying the file |

No TODO/FIXME/placeholder stubs found in the core implementation files. The file is syntactically valid Python (`ast.parse` confirms).

### Human Verification Required

None beyond the automated checks — BPB and artifact size results come from a completed SLURM run with clear numeric output.

## Gaps Summary

**Root cause:** Phase 4 was scoped in the PLAN to verify presence and fix artifact size. The plans did not include tasks to execute ablation runs. The ROADMAP success criteria, however, require individual ablation measurements that were never taken.

**What is solid:**
- All three differentiators (ARCH-05, ARCH-06, ARCH-07) are implemented correctly and wired in both forward paths
- BPB 1.1421 beats SOTA 1.1428 by 0.0007
- Artifact 15,824,167 bytes is under the 16MB cap with 176KB headroom
- Code is at 53,995 bytes (under 54,000 byte target), 1,259 lines (under 1,500 hard cap)
- Pruning threshold is correctly set to 5%
- File is syntactically valid Python

**What is missing:**
- ROADMAP SC 1: No measured proof that SmearGate alone contributes >0.001 BPB improvement
- ROADMAP SC 2: No per-differentiator marginal BPB contribution table exists anywhere
- The ablation infrastructure (ablation_runner.py) exists but is not wired to train_gpt.py because train_gpt.py has no ENABLE_*/toggle flags

**Severity assessment:** The competition outcome (BPB 1.1421, artifact under cap) is achieved. The ablation gap is a documentation and rigor gap, not a quality regression. However, the ROADMAP goal as literally stated ("individually ablated") is not met. For downstream phases (Phase 5 moonshot decisions depend on knowing which differentiators are load-bearing), the lack of ablation data is a real risk.

---

_Verified: 2026-03-23T07:44:07Z_
_Verifier: Claude (gsd-verifier)_

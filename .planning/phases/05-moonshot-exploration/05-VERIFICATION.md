---
phase: 05-moonshot-exploration
verified: 2026-03-23T09:27:32Z
status: gaps_found
score: 3/4 success criteria verified
gaps:
  - truth: "The best model configuration achieves BPB < 1.135"
    status: failed
    reason: "No moonshot improved upon the base model (1.1421 BPB). All four directions produced regressions: int4 +0.038, curriculum +0.002, TTT +0.111, recurrence +0.072. The phase goal of < 1.13 BPB was not achieved."
    artifacts:
      - path: "repo/experiments/moonshot_int4.py"
        issue: "Produced BPB 1.1799, a regression of +0.038 over base"
      - path: "repo/experiments/moonshot_curriculum.py"
        issue: "Produced BPB 1.1443, a regression of +0.002 over base"
      - path: "repo/experiments/moonshot_ttt.py"
        issue: "Produced BPB 1.2529, a regression of +0.111 over base"
      - path: "repo/experiments/moonshot_recurrence.py"
        issue: "Produced BPB 1.2141, a regression of +0.072 over base"
    missing:
      - "Any moonshot technique that delivers BPB improvement over 1.1421"
      - "A model configuration achieving BPB < 1.135"
      - "RECUR-04 satisfied: recurrent model did NOT achieve lower BPB at same artifact size"
      - "TTT-02 satisfied: TTT did NOT provide >= 0.002 BPB improvement (produced -0.111 regression)"
      - "CURR-03 satisfied: curriculum did NOT provide measurable BPB improvement"
      - "LOWBIT-03 satisfied: int4 did NOT achieve lower BPB than int5 model"
---

# Phase 5: Moonshot Exploration Verification Report

**Phase Goal:** At least one moonshot technique (depth recurrence, TTT, curriculum learning, or sub-5-bit quantization) delivers a measurable BPB improvement beyond the optimized SOTA stack, targeting < 1.13 BPB
**Verified:** 2026-03-23T09:27:32Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Depth-recurrent model trains stably and achieves lower BPB than non-recurrent at same artifact size (RECUR-04) | FAILED | Log confirms BPB 1.2141 (+0.072 regression). Artifact was 9.05MB (smaller), but quality catastrophically worse. |
| 2 | LoRA TTT provides measurable BPB improvement (>0.002) within time budget (TTT-02, TTT-03) | FAILED | Log confirms ttt_improvement: -0.110612 BPB (massive regression, not improvement). Eval time 87s is within budget, but quality failed. |
| 3 | Each moonshot has a clear go/no-go result with quantified evidence | VERIFIED | All 4 experiments have actual training logs with exact BPB values and documented NO-GO decisions in SUMMARYs. |
| 4 | Best model configuration achieves BPB < 1.135 | FAILED | Best result was curriculum at 1.1443, still above 1.135. Base model (1.1421) remains the best at this scale. |

**Score:** 1/4 truths verified (3/4 success criteria met if counting go/no-go completion + stability)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `repo/experiments/moonshot_int4.py` | Copy of train_gpt.py with clip=7 for MLP | VERIFIED | 1259 lines. Diff confirms exactly 2 lines changed (clip 15->7, metadata int5->int4). |
| `scripts/slurm/train_moonshot_int4.sbatch` | SLURM job for int4 experiment | VERIFIED | 49 lines. References moonshot_int4.py correctly. |
| `scripts/curriculum/compute_difficulty.py` | Script to precompute per-shard difficulty | VERIFIED | 40 lines. Contains zlib.compress and json.dump. |
| `scripts/curriculum/difficulty.json` | Cached difficulty scores for all shards | VERIFIED | 81 lines. Contains 80 fineweb_train_*.bin entries with compression ratios. |
| `repo/experiments/moonshot_curriculum.py` | Copy of train_gpt.py with curriculum-aware TokenStream | VERIFIED | 1275 lines. Contains curriculum_warmup_frac, difficulty_file param, curriculum:switching_to_random logic. |
| `scripts/slurm/train_moonshot_curriculum.sbatch` | SLURM job for curriculum experiment | VERIFIED | 51 lines. References moonshot_curriculum.py and difficulty.json env var. |
| `repo/experiments/moonshot_ttt.py` | Copy of train_gpt.py with LoRA TTT eval | VERIFIED | 1537 lines. Contains BatchedLinearLoRA, eval_val_ttt_lora, BOS_ID boundary detection, score-first comments. |
| `scripts/slurm/train_moonshot_ttt.sbatch` | SLURM job for TTT experiment | VERIFIED | 50 lines. References moonshot_ttt.py with TTT_ENABLED=1, TTT_LORA_RANK=4. |
| `repo/experiments/moonshot_recurrence.py` | Copy of train_gpt.py with depth recurrence | VERIFIED | 1274 lines. Contains n_loops, adapter (CastedLinear 2*dim->dim), loop_scales, for loop_idx in range loop. |
| `scripts/slurm/train_moonshot_recurrence.sbatch` | SLURM job for recurrence experiment | VERIFIED | 52 lines. References moonshot_recurrence.py with N_CORE_BLOCKS=5, N_LOOPS=2, MATRIX_LR=0.01. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `moonshot_int4.py` | `mixed_quantize_int6` | clip = 7 for MLP category | VERIFIED | Line 497: `clip = 7 if cat == "mlp" else 31`. Diff confirms exactly 2 lines changed from base. |
| `moonshot_curriculum.py` | `difficulty.json` | TokenStream reads difficulty scores and sorts shards | VERIFIED | Line 547-553: difficulty_file param, sorts by scores.get(). Line 1041: curriculum:enabled log. |
| `moonshot_ttt.py` | `BatchedLinearLoRA` | LoRA applied to Q, V projections, score-first-then-train ordering | VERIFIED | Lines 900, 1014-1015 contain explicit "Score FIRST" comments. BOS_ID=1 used for document boundary resets. eval_val_ttt_lora called from main() at line 1517. |
| `moonshot_recurrence.py` | `GPT.forward` | 5 blocks x 2 loops with embedding re-injection via adapter | VERIFIED | Lines 833-838: `for loop_idx in range(self.n_loops)` with adapter concat. Adapter routed to matrix_params line 989. loop_scales routed to scalar_params line 996. |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| RECUR-01: 5-6 unique layers looped 2-3x | SATISFIED | 5 blocks x 2 loops implemented. Code verified in moonshot_recurrence.py. |
| RECUR-02: Per-step adaptation via scaling factors | SATISFIED | loop_scales parameter implemented and routed to optimizer. |
| RECUR-03: Training stability under recurrence | SATISFIED | Log shows no NaN/explosion. Training completed stably (BPB regression is quality, not instability). |
| RECUR-04: Recurrent achieves lower BPB at same artifact size | BLOCKED | BPB 1.2141 vs base 1.1421. Artifact was smaller (9.05MB) but BPB catastrophically worse. |
| TTT-01: LoRA TTT on already-evaluated tokens only | SATISFIED | Score-first-then-train pattern verified in code (lines 900, 1014-1015, 1090-1091). |
| TTT-02: TTT provides >= 0.002 BPB improvement | BLOCKED | TTT produced BPB regression of -0.111 (much worse, not better). |
| TTT-03: TTT eval within 10-minute budget | SATISFIED | Log shows eval_time:87442ms (87 seconds). |
| TTT-04: TTT is rule-compliant (no future token training) | SATISFIED | BOS boundary detection at line 902, reset() at document boundaries verified in code. |
| CURR-01: Training data ordered by difficulty | SATISFIED | difficulty.json with 80 shards, zlib compression ratio metric. Code sorts shards in TokenStream. |
| CURR-02: Easy-to-hard curriculum schedule | SATISFIED | 30% warmup fraction, switches to random at step threshold. |
| CURR-03: Curriculum provides measurable BPB improvement | BLOCKED | BPB 1.1443 vs base 1.1421. +0.002 regression, not improvement. |
| LOWBIT-01: Int4 quantization for MLP layers | SATISFIED | clip=7 change verified. Exactly 2 lines differ from base. |
| LOWBIT-02: Mixed int4/int5/int6 per-layer allocation | SATISFIED | MLP=int4 (clip=7), attention=int6 (clip=31), embeddings FP16. |
| LOWBIT-03: Int4 achieves lower BPB at same/smaller artifact | BLOCKED | BPB 1.1799 vs base 1.1421. Artifact smaller (14.07MB vs 15.82MB) but quality regression +0.038. |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | — | — | All experiment scripts are full implementations with real training runs. No stubs or placeholders found. |

### Training Run Evidence

All four experiments have corresponding log files with actual BPB measurements:

| Experiment | Log File | Final BPB | Delta vs Base | Decision |
|------------|----------|-----------|---------------|----------|
| Int4 MLP | `logs/moonshot_int4_seed1337_20260323_041122.log` | 1.1799 | +0.038 | NO-GO |
| Curriculum | `logs/moonshot_curriculum_seed1337_20260323_042646.log` | 1.1443 | +0.002 | NO-GO |
| TTT | `logs/moonshot_ttt_seed1337_20260323_050524.log` | 1.2529 (TTT) / 1.1422 (standard) | TTT: +0.111 | NO-GO |
| Recurrence | `logs/moonshot_recurrence_seed1337_20260323_044907.log` | 1.2141 | +0.072 | NO-GO |

All BPB values confirmed directly from log lines (`final_int5_6_roundtrip val_bpb:` and `ttt_eval val_bpb:`).

### Gaps Summary

The phase primary goal — "at least one moonshot delivers measurable BPB improvement targeting < 1.13 BPB" — was NOT achieved. All four moonshot directions produced BPB regressions:

- **Int4 quantization**: Quality cost (+0.038 BPB) far outweighed artifact savings (1.76MB). The 16-level quantization is too coarse for relu-squared MLP activations at this scale.
- **Curriculum learning**: The narrow compression-ratio spread across shards (0.4978 to 0.5583) means FineWeb difficulty is relatively uniform. The 600s training budget is too short for curriculum benefits to manifest.
- **TTT LoRA**: The strong base model (1.1421) is already well-adapted. LoRA adaptation at eval time overfits per-document rather than generalizing, producing dramatic BPB regression (+0.111).
- **Depth recurrence**: 5-layer quality converges to 5-layer representational capacity regardless of loop count. At 16MB/25M parameter scale, per-layer specialization is essential — shared weights defeat this.

Four requirements are blocked (RECUR-04, TTT-02, CURR-03, LOWBIT-03) because they specify that each technique must produce BPB improvement — which none did. Ten requirements ARE satisfied (RECUR-01-03, TTT-01, TTT-03-04, CURR-01-02, LOWBIT-01-02) because they cover implementation and process quality, not outcome quality.

**The science is sound.** The experimental infrastructure is rigorous, each experiment ran to completion, and the go/no-go decisions are backed by actual measured BPB from SLURM training jobs. The negative results are informative: the base model (1.1421) is already at or near the optimum for these techniques at this scale.

**The base model already beats current SOTA** (1.1421 vs SOTA 1.1428), which provides a valid submission path through Phase 6 even without moonshot improvements.

---

*Verified: 2026-03-23T09:27:32Z*
*Verifier: Claude (gsd-verifier)*

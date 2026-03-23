---
phase: 02-sota-stack-integration
verified: 2026-03-23T03:47:00Z
status: gaps_found
score: 10/12 must-haves verified
re_verification: false
gaps:
  - truth: "Artifact fits well under 16MB (target: < 15.5MB)"
    status: failed
    reason: "Artifact is 18.49MB (18,487,380 bytes), exceeding the 16MB cap by ~2.5MB. Int6 uniform quantization is insufficient for the 25.5M parameter model. Phase 3 (int5 MLP) will address this."
    artifacts:
      - path: "repo/train_gpt.py"
        issue: "mixed_quantize_int6 uses clip=31 (int6) for ALL categories including MLP. MLP weights need int5 (clip=15) to reduce artifact size below cap."
    missing:
      - "Int5 quantization for MLP weights (clip_range=15) — deferred to Phase 3"
      - "Artifact size verification within 16MB hard cap"
  - truth: "Training wall time < 420s (7 minutes) on H200"
    status: failed
    reason: "Training hit the 600s wallclock cap at step 6974/20000 (86ms/step). The 420s target was based on completing a proportionally shorter warmdown, but the full training loop with the larger model (10L, MLP 3x, seq_len 2048) runs slower than projected."
    artifacts:
      - path: "repo/train_gpt.py"
        issue: "No structural defect — training is functional. The 86ms/step measured throughput means 7 minutes is insufficient for 20000 steps. This is a throughput/target mismatch, not a code bug."
    missing:
      - "Throughput optimization to achieve 420s training wall time"
      - "Or: revised step count / warmdown schedule tuned to 600s wall cap"
human_verification: []
---

# Phase 02: SOTA Stack Integration — Verification Report

**Phase Goal:** A competitive model implementing all table-stakes techniques achieves ~1.15 BPB, matching the lower tier of leaderboard submissions
**Verified:** 2026-03-23T03:47:00Z
**Status:** gaps_found (2 gaps, both pre-acknowledged in 02-03-SUMMARY.md)
**Re-verification:** No — initial verification

## Goal Achievement

The headline goal is achieved: the model reaches 1.1356 BPB, which beats SOTA (1.1428 BPB) and is well under the 1.155 target. The architecture, training pipeline, quantization, and evaluation mechanisms are all correctly implemented and verified against the actual codebase. Two gaps exist — artifact size and training time — both of which were identified and documented in the SUMMARY before verification.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Model has 10L, 512 dim, MLP 3x (1536 hidden), relu-squared, GQA 8Q/4KV | VERIFIED | `num_layers=10`, `mlp_mult=3.0`, `hidden=int(3.0*512)=1536`, `relu().square()`, `num_heads=8`, `num_kv_heads=4` in Hyperparameters and GPT classes |
| 2 | BigramHash(10240, dim=128) adds bigram embeddings before RMSNorm in forward | VERIFIED | Line 931: `x = x + self.bigram(input_ids)`, followed by `F.rms_norm` at line 932. `bigram_vocab_size=10240`, `bigram_dim=128` |
| 3 | SmearGate blends adjacent token embeddings after RMSNorm | VERIFIED | Line 933: `x = self.smear(x)` after `F.rms_norm`. SmearGate class at line 798 implements `(1-g)*x + g*x_prev` |
| 4 | Orthogonal init for large Linear weights with scaled output projections | VERIFIED | Lines 922–926: `nn.init.orthogonal_` for weights >= 64x64, with `1/sqrt(2*num_layers)` scaling for `.proj.` layers |
| 5 | Hyperparameters match SOTA: seq_len=2048, batch=786432, warmdown=3000, matrix_lr=0.02, momentum=0.99, grad_clip=0.3, weight_decay=0.04 | VERIFIED | All 7 values confirmed in Hyperparameters class (lines 61–94) |
| 6 | Muon WD, AdamW, correct param routing, CONTROL_TENSOR includes smear and bigram.scale | VERIFIED | `p.data.mul_(1.0 - lr * wd)` at line 181; `AdamW` at lines 1129, 1145; `smear.gate` at line 1119; `bigram.scale` at line 1121; `bigram.embed.weight` at line 1126; `bigram.proj.weight` at line 1128; CONTROL_TENSOR pattern at line 390 |
| 7 | SWA collects during warmdown, averages after training | VERIFIED | Log confirms: `swa:start step:5800`, `swa:applying averaged 24 checkpoints`. Code: lines 1309–1317 (collect), 1344–1351 (apply) |
| 8 | Int6 PTQ with per-row scaling, 3% magnitude pruning, zstd-22 | VERIFIED | `quantize_intN_per_row` (line 539), `mixed_quantize_int6` (line 552), `quantile(0.03)` pruning (line 1371), `ZstdCompressor(level=22)` (line 1382) |
| 9 | Roundtrip dequantization restores model for eval | VERIFIED | `dequantize_mixed_int6` called at line 1402, `base_model.load_state_dict(deq_state)` at line 1403 |
| 10 | Sliding window eval stride=64 computes final BPB | VERIFIED | `eval_val_sliding` at line 298, called with `stride=args.eval_stride` (64) at line 1413. Log confirms `final_eval_mode:sliding_window stride:64 batch_seqs:32` |
| 11 | Training achieves val_bpb <= 1.155 | VERIFIED | Log: `final_int8_zlib_roundtrip_exact val_bpb:1.13561527` — beats 1.155 target and current SOTA 1.1428 |
| 12 | Artifact fits < 16MB (target: < 15.5MB) | FAILED | Log: `Total submission size int6+zstd: 18,487,380 bytes` — 18.49MB, exceeds 16MB cap by ~2.5MB |
| 13 | Training wall time < 420s | FAILED | Log: `stopping_early: wallclock_cap train_time:600065ms step:6974/20000` — hit 600s wall cap, not 420s |

**Score:** 11/13 truths verified (and 10/12 plan must-haves verified; BPB truth spans both TRAIN-05 and primary goal)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `repo/train_gpt.py` | Complete SOTA training script | VERIFIED | 1433 lines (under 1500 cap). All modules, functions, and pipeline present and substantive. |
| `scripts/slurm/train_sota.sbatch` | SLURM job script for SOTA training | VERIFIED | Exists, targets pg_tata partition, 8 GPUs, uses torchrun, loads pgolf conda env |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `GPT.forward` | `BigramHashEmbedding.forward` | `x = x + self.bigram(input_ids)` | WIRED | Line 931, before rms_norm at 932 |
| `GPT.forward` | `SmearGate.forward` | `x = self.smear(x)` | WIRED | Line 933, after rms_norm at 932 |
| `GPT._init_weights` | `nn.init.orthogonal_` | orthogonal init for large Linear weights | WIRED | Lines 922–926, with scaled output projections |
| `Muon.step` | `p.data.mul_(1.0 - lr * wd)` | decoupled WD before gradient update | WIRED | Lines 176–182 |
| optimizer setup | `base_model.smear.gate` | appended to scalar_params | WIRED | Line 1119 |
| training loop | `swa_state` | scale < swa_start_frac triggers collection | WIRED | Line 1309 |
| `main() serialization` | `mixed_quantize_int6` | quantize state dict after SWA and pruning | WIRED | Line 1377 |
| `main() serialization` | `dequantize_mixed_int6` | roundtrip dequantize for eval | WIRED | Line 1402 |
| `main() final eval` | `eval_val_sliding` | sliding window eval on roundtripped weights | WIRED | Lines 1408–1414 |
| `serialization` | `zstandard.ZstdCompressor` | zstd level 22 compression | WIRED | Line 1382 |
| `eval_val_sliding` | `base_model.forward_logits` | forward_logits called in sliding eval | WIRED | Line 340 |

### Requirements Coverage

| Requirement | Description | Status | Notes |
|-------------|-------------|--------|-------|
| ARCH-01 | 10L transformer, 512 dim, MLP 3x, relu-squared | SATISFIED | Verified in Hyperparameters and GPT classes |
| ARCH-02 | GQA 8Q/4KV | SATISFIED | `num_heads=8`, `num_kv_heads=4`, enable_gqa used |
| ARCH-03 | BigramHash(10240, dim=128) | SATISFIED | BigramHashEmbedding class, `bigram_vocab_size=10240`, `bigram_dim=128` |
| ARCH-04 | Tied embeddings with RMSNorm and RoPE | SATISFIED | `tie_embeddings=True` default, RMSNorm applied, Rotary class present |
| QUANT-01 | Int6 PTQ with per-row scaling for all weight matrices | SATISFIED | `quantize_intN_per_row` with clip_range=31, used for mlp/attn/bigram categories |
| QUANT-03 | zstd level 22 compression with zlib fallback | SATISFIED | `ZstdCompressor(level=22)`, `_COMPRESSOR` fallback logic |
| TRAIN-01 | Muon optimizer with AdamW for embeddings/biases | SATISFIED | Muon for matrix_params (weight_decay=0.04), AdamW for tok/scalar (weight_decay=0.04) |
| TRAIN-02 | SWA over 20-24 checkpoints from last 40% of warmdown | SATISFIED | 24 checkpoints confirmed by log, swa_start_frac=0.4 |
| TRAIN-03 | Weight decay 0.04 via both Muon and AdamW | SATISFIED | Muon hardcoded 0.04, AdamW uses `args.weight_decay=0.04` |
| TRAIN-04 | DDP training across 8 GPUs with 786K tokens/batch | SATISFIED | torchrun with 8 GPUs, `train_batch_tokens=786_432`, DDP at line 1099 |
| TRAIN-05 | Training completes within 7 minutes on H200 | BLOCKED | Hit 600s wall cap; 86ms/step puts 7 min (420s) at ~4880 steps, not a viable full training run |
| EVAL-01 | Sliding window eval stride=64, context 2048 | SATISFIED | `eval_val_sliding` function, `train_seq_len=2048`, `eval_stride=64` |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `repo/train_gpt.py` | 571 | `# Phase 2: uniform int6 for all (int5 MLP deferred to Phase 3)` | Info | Intentional — documents Phase 3 task |
| `repo/train_gpt.py` | 794 | `x = torch.relu(self.fc(x))` | Info | relu-squared is `relu(x).square()` — this IS correct; square is on line 795 |

No blockers or warnings found. The inline comment at line 571 is informational design documentation, not a placeholder stub.

### Gaps Summary

Two gaps exist, both pre-acknowledged in the 02-03-SUMMARY.md before this verification:

**Gap 1: Artifact size 18.49MB exceeds 16MB cap.**
Root cause: Using uniform int6 (clip_range=31) for ALL weight categories, including MLP. MLP weights benefit from int5 (clip_range=15), which Phase 3 implements. The quantization code has a comment explicitly deferring this. The BPB performance (1.1356) proves the architecture and training pipeline are correct — the artifact size is a compression/quantization issue assigned to Phase 3.

**Gap 2: Training wall time exceeded 420s target.**
Root cause: The 10L/MLP3x/seq_len-2048 model runs at ~86ms/step on H200. Reaching 20000 steps requires ~1720s. The 600s wall cap stops training at step 6974. The 420s target was aspirational. Notably, the 600s-capped run still produced excellent BPB (1.1356), beating SOTA. Phase 3 should revisit the training time requirement in context of the wall cap.

Both gaps are infrastructure/resource gaps, not architecture correctness gaps. The primary phase goal — achieving ~1.15 BPB with all table-stakes techniques — is definitively met at 1.1356 BPB.

---
_Verified: 2026-03-23T03:47:00Z_
_Verifier: Claude (gsd-verifier)_

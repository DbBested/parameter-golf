---
phase: 01-infrastructure-and-baseline
verified: 2026-03-22T00:00:00Z
status: gaps_found
score: 7/8 must-haves verified
re_verification: false
gaps:
  - truth: "H200-to-H100 timing ratio is empirically measured for our specific model workload (not assumed)"
    status: partial
    reason: "The 1.35x ratio in timing_calibration.py is a literature estimate (middle of 1.3-1.45 range), not measured from actual H100 runs. The SUMMARY explicitly notes: 'actual ratio unknown until RunPod validation'. The H200 wall time is empirically measured (600s), but the H100 ratio itself is assumed. ROADMAP SC #4 states 'empirically measured (not assumed)'."
    artifacts:
      - path: "scripts/eval/timing_calibration.py"
        issue: "estimate_h100_time(h200_seconds, ratio=1.35) defaults to literature estimate; no H100 measurement in codebase"
    missing:
      - "Actual H100 training run to empirically measure the slowdown ratio"
      - "OR: explicit documentation that empirical H100 measurement is deferred to Phase 6 (RunPod validation)"
human_verification:
  - test: "Confirm that the partial INFRA-04 gap is intentionally deferred"
    expected: "H100 timing will be empirically measured in Phase 6 (RunPod), and Phase 1 tooling (timing_calibration.py) is designed to accept the measured ratio as input. If this is the intended interpretation, INFRA-04 can be considered satisfied by the tooling existing even without an H100 run."
    why_human: "Requires judgment on whether 'empirically measured' means (a) infrastructure for future measurement exists or (b) the measurement has been taken. The SUMMARY self-declares success but acknowledges the ratio is unknown."
---

# Phase 1: Infrastructure and Baseline Verification Report

**Phase Goal:** Trustworthy measurement infrastructure exists and the official baseline (1.2244 BPB) is reproduced on pg_tata, so every subsequent experiment has a reliable reference point
**Verified:** 2026-03-22
**Status:** gaps_found (1 partial gap)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|---------|
| 1  | conda environment 'pgolf' exists with PyTorch, scipy, zstandard, and competition deps | ? HUMAN | Cannot verify conda env from filesystem; SUMMARY confirms install; login node shows CUDA=False (expected) |
| 2  | parameter-golf repo is cloned to repo/ and FineWeb sp1024 dataset is cached | VERIFIED | `repo/train_gpt.py` exists; `repo/data/datasets/fineweb10B_sp1024/` contains 81 .bin shards |
| 3  | experiment logging appends JSON-lines records with run_id, config, and metrics fields | VERIFIED | `scripts/tracking/log_experiment.py` writes to `experiments.jsonl`; 3 real records exist with all required fields |
| 4  | storage cleanup script removes old checkpoints keeping N most recent plus best | VERIFIED | `scripts/storage/cleanup.py` globs `*.ptz`/`*.pt`, sorts by mtime, keeps N recent + best by val_bpb, calls `.unlink()` |
| 5  | project directory structure matches research-recommended layout | VERIFIED | `scripts/`, `experiments/results/`, `experiments/logs/`, `checkpoints/` all exist |
| 6  | Baseline training on pg_tata H200 produces BPB within 0.005 of 1.2244 | VERIFIED | experiments.jsonl: BPB [1.22592, 1.22558, 1.22459], mean 1.22536; all within 0.005 of 1.2244 |
| 7  | Artifact size validation reports code bytes + model bytes against 16,000,000 byte cap | VERIFIED | `check_artifact_size.py` reports model/code/total/budget/status; exits 1 on FAIL |
| 8  | Running 3+ seeds produces mean, std, and p-value output with std < 0.003 BPB | VERIFIED | multiseed_eval.py uses scipy.stats.ttest_ind; 3 seed records logged; computed std = 0.000692 (< 0.003) |
| 9  | H200 training wall-clock is measured and extrapolated to H100 using a conservative ratio | PARTIAL | H200 time empirically measured (600s all seeds); H100 ratio 1.35x is a literature estimate not measured from H100 runs |
| 10 | Ablation framework can toggle technique flags and measure marginal BPB contribution | VERIFIED | `ablation_runner.py` sets `ENABLE_{TECHNIQUE}=1/0` env vars, generates modified sbatch scripts, supports dry_run, calls ttest_ind for comparison |
| 11 | Experiment tracking captures BPB, artifact size, training time, hyperparameters for every run | VERIFIED | All 3 records in experiments.jsonl have val_bpb, artifact_bytes, wall_time_seconds, seed, n_layers, n_dim, vocab_size, optimizer, quantization |
| 12 | BPB calculation validated against official evaluation script with exact byte-level accounting | VERIFIED | process_baseline_results.py extracts `final_int8_zlib_roundtrip_exact val_bpb:` from train_gpt.py logs (the official eval function `eval_val` is used, which performs exact byte-level BPB via base_bytes_lut); validate_bpb.py compares measured against 1.2244 with 0.005 tolerance |

**Score:** 11/12 truths verified (1 partial on timing ratio empiricism)

### Required Artifacts

| Artifact | Expected | Exists | Lines | Stubs | Wired | Status |
|----------|----------|--------|-------|-------|-------|--------|
| `scripts/tracking/log_experiment.py` | JSON-lines logger | YES | 145 | NONE | Called by train_multiseed.sbatch via process_baseline_results.py | VERIFIED |
| `scripts/tracking/query_experiments.py` | Experiment display | YES | 144 | NONE | Standalone CLI | VERIFIED |
| `scripts/storage/cleanup.py` | Checkpoint cleanup | YES | 162 | NONE | Called by train_multiseed.sbatch | VERIFIED |
| `scripts/slurm/train_baseline.sbatch` | SLURM baseline job | YES | 63 | NONE | Working (jobs 10812326, 10812368) | VERIFIED |
| `scripts/eval/check_artifact_size.py` | Artifact size check | YES | 109 | NONE | Called by train_baseline.sbatch and train_multiseed.sbatch | VERIFIED |
| `scripts/eval/validate_bpb.py` | BPB validation | YES | 81 | NONE | Called by process_baseline_results.py | VERIFIED |
| `scripts/eval/multiseed_eval.py` | Multi-seed stats | YES | 198 | NONE | Called by train_multiseed.sbatch | VERIFIED |
| `scripts/eval/timing_calibration.py` | H200/H100 timing | YES | 237 | NONE | Called by train_multiseed.sbatch and timing_profile.sbatch | VERIFIED (partial gap) |
| `scripts/eval/ablation_runner.py` | Ablation framework | YES | 369 | NONE | Standalone CLI, references train_baseline.sbatch as template | VERIFIED |
| `scripts/slurm/train_multiseed.sbatch` | Multi-seed SLURM job | YES | 145 | NONE | Used for seeds 42 and 7 (job 10813207) | VERIFIED |
| `scripts/slurm/timing_profile.sbatch` | Timing profile job | YES | 93 | NONE | Created, calls timing_calibration.py | VERIFIED |
| `experiments/results/experiments.jsonl` | Experiment log | YES | 3 records | NONE | Written by process_baseline_results.py; read by multiseed_eval and timing_calibration | VERIFIED |
| `.gitignore` | Git exclusions | YES | 9 | NONE | Root of repo | VERIFIED |
| `repo/train_gpt.py` | Official baseline | YES | - | N/A | 81 FineWeb shards present | VERIFIED |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `scripts/tracking/log_experiment.py` | `experiments/results/experiments.jsonl` | JSON-lines append | WIRED | Line 48: `log_path = result_path / "experiments.jsonl"`; line 50: `f.write(json.dumps(record) + "\n")` |
| `scripts/storage/cleanup.py` | `checkpoints/*.ptz` | glob and unlink | WIRED | Lines 49-50: globs `*.ptz` and `*.pt`; line 106: `ckpt.unlink()` |
| `scripts/eval/check_artifact_size.py` | `repo/train_gpt.py` | measures code file size | WIRED | Line 36: `os.path.getsize(model_path)`; reads code paths via UTF-8 encode |
| `scripts/eval/validate_bpb.py` | `experiments/results/experiments.jsonl` | reads BPB from log (via process_baseline_results.py) | WIRED | process_baseline_results.py imports validate_baseline_bpb and reads experiments.jsonl |
| `scripts/slurm/train_baseline.sbatch` | `scripts/eval/check_artifact_size.py` | post-training artifact check | WIRED | Line 54: `python scripts/eval/check_artifact_size.py --model repo/final_model.int8.ptz --code repo/train_gpt.py` |
| `scripts/eval/multiseed_eval.py` | `experiments/results/experiments.jsonl` | reads BPB values | WIRED | collect_seed_results reads jsonl, filters by prefix, extracts val_bpb |
| `scripts/eval/multiseed_eval.py` | `scipy.stats.ttest_ind` | statistical significance testing | WIRED | Line 20: `from scipy.stats import ttest_ind`; line 70: `ttest_ind(values, baseline_values, alternative="less")` |
| `scripts/eval/timing_calibration.py` | `experiments/logs/` | parses wall time from SLURM logs | WIRED | parse_timing_from_log reads log file, regex extracts `train_time:XXXms` |
| `scripts/eval/ablation_runner.py` | `scripts/slurm/train_baseline.sbatch` | submits SLURM jobs with technique flags | WIRED | Default `--sbatch-template scripts/slurm/train_baseline.sbatch`; runs `subprocess.run(["sbatch", ...])`; dry_run prevents actual submission |
| `scripts/slurm/train_multiseed.sbatch` | `scripts/eval/multiseed_eval.py` | post-training analysis | WIRED | Lines 121-123: `python scripts/eval/multiseed_eval.py --result-dir experiments/results --run-id-prefix baseline_seed` |

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|---------|
| INFRA-01 | Training pipeline reproduces official baseline (1.2244 BPB) on pg_tata H200 | SATISFIED | 3 seeds: BPB 1.22592, 1.22558, 1.22459 — all within 0.005 of 1.2244 |
| INFRA-02 | Automated artifact size validation reports code + model bytes against 16MB cap after every run | SATISFIED | check_artifact_size.py wired into train_baseline.sbatch and train_multiseed.sbatch; all 3 runs under 16MB |
| INFRA-03 | Multi-seed evaluation: 3+ seeds, mean, std, p-value for BPB differences | SATISFIED | multiseed_eval.py implements analyze_seeds with ttest_ind; 3 records logged; std=0.000692 |
| INFRA-04 | H200-to-H100 timing calibration empirically measures slowdown factor | PARTIAL | H200 time is empirically measured (600s/run); H100 ratio is a literature estimate (1.35x, described as "middle of 1.3-1.45 range; actual ratio unknown until RunPod validation") |
| INFRA-05 | Experiment tracking records BPB, artifact size, training time, hyperparameters | SATISFIED | log_experiment.py writes all fields; 3 complete records in experiments.jsonl |
| INFRA-06 | Ablation framework enables toggling individual techniques on/off | SATISFIED | ablation_runner.py sets ENABLE_{TECHNIQUE} env vars, generates sbatch scripts, supports dry_run and compare modes |
| INFRA-07 | Storage management automatically cleans checkpoints within disk quota | SATISFIED | cleanup.py keeps N recent + best-BPB, called by train_multiseed.sbatch; 2 baseline checkpoints exist |
| EVAL-02 | BPB calculation validated against official evaluation script with exact byte-level accounting | SATISFIED | process_baseline_results.py extracts BPB from `final_int8_zlib_roundtrip_exact` line (train_gpt.py's official eval_val function with base_bytes_lut byte accounting); validate_bpb.py checks against 1.2244 with 0.005 tolerance |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | — | — | No TODO/FIXME/placeholder/stub patterns in any script |

### Human Verification Required

#### 1. Confirm INFRA-04 partial gap is intentionally deferred

**Test:** Review whether the Phase 1 ROADMAP success criterion "H200-to-H100 timing ratio is empirically measured (not assumed)" is satisfied by the tooling infrastructure alone, or requires an actual H100 run.

**Expected:** Two acceptable resolutions:
1. The 1.35x ratio is acknowledged as a placeholder estimate pending RunPod H100 validation in Phase 6, and INFRA-04 is re-scoped to mean "tooling for timing calibration exists and H200 time is measured" — in which case this gap can be closed without running on H100.
2. A timing run on RunPod H100 is performed and the actual ratio is plugged into timing_calibration.py, which then genuinely satisfies the requirement.

**Why human:** The SUMMARY simultaneously claims INFRA-04 is "MET" and states "actual ratio unknown until RunPod validation." This is a scope interpretation requiring human judgment, not a code defect.

#### 2. Conda environment functional on compute nodes

**Test:** Run `sbatch scripts/slurm/train_baseline.sbatch` and verify the job starts, activates pgolf env, and reports `CUDA: True, GPUs: 8`.

**Expected:** Job output shows PyTorch 2.6.0+cu124, CUDA available, 8xH200 detected.

**Why human:** Login nodes have CUDA=False; environment correctness on compute nodes can only be confirmed by running a job.

### Gaps Summary

One gap was found, marked as partial rather than failed because the underlying infrastructure is complete and working:

**INFRA-04 (Partial):** The H200-to-H100 timing calibration tooling is fully built (`timing_calibration.py` with `estimate_h100_time`, `calibrate_from_experiments`, and `parse_timing_from_log`). The H200 training time is empirically measured at 600s per seed across 3 runs. However, the H100 slowdown ratio (1.35x) is a literature estimate, not measured from actual H100 hardware. The SUMMARY explicitly notes "actual ratio unknown until RunPod validation." The ROADMAP success criterion specifically says "empirically measured (not assumed)."

This is unlikely to block Phase 2 work — all Phase 1 tooling is complete and the baseline is reproducibly measured. The timing gap will naturally be resolved in Phase 6 (RunPod validation).

All other requirements (INFRA-01, INFRA-02, INFRA-03, INFRA-05, INFRA-06, INFRA-07, EVAL-02) are fully satisfied with substantive, wired implementations and real experimental results.

---

_Verified: 2026-03-22_
_Verifier: Claude (gsd-verifier)_

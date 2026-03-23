# Phase 3 Plan 2: Training Results (SLURM Job 10821205)

## Job Status
- **Job ID:** 10821205
- **State:** COMPLETED
- **Exit Code:** 0:0
- **Elapsed:** 00:15:13
- **Node:** node4300 (8x H200)
- **Seed:** 1337

## Results Comparison

| Metric | Phase 2 (int6 uniform) | Phase 3 (int5 MLP + int6 attn) | Delta |
|--------|------------------------|--------------------------------|-------|
| Compressed artifact (bytes) | 18,487,380 | 16,173,399 | -2,313,981 (-12.5%) |
| Compressed model only (bytes) | 18,425,879 | 16,110,910 | -2,314,969 (-12.6%) |
| Code size (bytes) | 61,501 | 62,489 | +988 |
| val_bpb (roundtrip, sliding) | 1.1356 | 1.1424 | +0.0068 |
| Under 16MB cap | NO | NO (over by 173,399) | -- |
| Steps completed | 6,974 | 6,983 | +9 |
| Wall time (train) | 600s | 600s | same |
| Step avg | 86.04ms | 85.93ms | -0.11ms |
| SWA checkpoints | 24 | 24 | same |
| Peak memory | 18,967 MiB | 18,968 MiB | +1 MiB |

## Critical Findings

### 1. Artifact Size: OVER 16MB cap
- Total submission: 16,173,399 bytes (exceeds 16,000,000 by 173,399 bytes)
- Model compressed: 16,110,910 bytes
- Code: 62,489 bytes
- Reduction from Phase 2: 2,313,981 bytes (12.5% smaller)
- Still needs ~174KB reduction to fit

### 2. BPB Quality: Degradation exceeds 0.005 threshold
- Phase 3 roundtrip BPB: 1.14244159
- Phase 2 roundtrip BPB: 1.13561527
- Delta: +0.00683 BPB (exceeds 0.005 tolerance)
- However, at step 6983, pre-quantization BPB was 1.1523 -- quantization + SWA may be responsible

### 3. Per-Layer Sensitivity: All layers logged successfully
- 10 blocks x (4 attn + 2 MLP) + embeddings + bigram = 63 sensitivity lines
- MLP int5 MSE range: 6.76e-05 to 1.78e-04
- Attention int6 MSE range: 2.26e-05 to 9.35e-05
- MLP layers show ~2-3x higher MSE than attention layers (expected for int5 vs int6)
- No layer exceeds MSE > 1e-3 threshold

### 4. Training Stability: Clean
- No NaN/inf warnings
- No gradient explosions
- Normal loss progression
- SWA collection started at step 5800, applied 24 checkpoints

## Validation BPB Progression

| Step | Phase 2 BPB | Phase 3 BPB |
|------|-------------|-------------|
| 1000 | 1.3216 | 1.3216 |
| 2000 | 1.2649 | 1.2655 |
| 3000 | 1.2403 | 1.2403 |
| 4000 | 1.2321 | 1.2320 |
| 5000 | 1.2064 | 1.2061 |
| 6000 | 1.1808 | 1.1808 |
| final | 1.1523 | 1.1523 |

Note: Pre-quantization BPB is essentially identical between Phase 2 and 3. The difference comes purely from the quantization roundtrip (int5 MLP has more noise than int6).

## Log Files
- Training log: `logs/int5_seed1337_20260323_023157.log`
- SLURM output: `logs/slurm/10821205-int5.out`
- SLURM error: `logs/slurm/10821205-int5.err` (empty)

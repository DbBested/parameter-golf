# Plan 05-04 Summary: Depth Recurrence Experiment

## Status: COMPLETE — NO-GO

## Results (SLURM Job 10824972)

| Metric | Base (10 unique) | Recurrent (5×2) | Delta |
|--------|-----------------|-----------------|-------|
| BPB | 1.1421 | 1.2141 | +0.072 (MUCH WORSE) |
| Artifact | 15,824,167 | 9,046,203 | -6.78MB (tiny!) |
| Headroom | 176KB | 6.95MB | Massive room |
| Params | 25.5M | 14.2M | -44% |

## Go/No-Go Decision: **NO-GO**

5 unique layers × 2 loops converges to 5-layer quality, not 10-layer quality. BPB 1.2141 is almost back to baseline level (1.2254). The model lacks representational diversity — shared weights can only learn one set of transformations applied twice, not the specialized per-layer representations that 10 unique layers provide.

The artifact savings are dramatic (6.78MB, 43% reduction) but the quality regression (+0.072 BPB) is catastrophic. At 16MB scale, unique layers are far more valuable than effective depth through repetition.

## Key Insight

Depth recurrence works at billion-parameter scale (Huginn-3.5B) because individual blocks are large enough to be expressive. At 16MB / 25M parameter scale, each layer needs to specialize differently. Sharing defeats this specialization.

## key-files

### created
- repo/experiments/moonshot_recurrence.py
- scripts/slurm/train_moonshot_recurrence.sbatch

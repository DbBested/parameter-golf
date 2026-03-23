# Plan 04-02 Summary: Training Validation

## Status: COMPLETE

## Results

| Metric | Phase 3 | Phase 4 | Delta |
|--------|---------|---------|-------|
| Artifact size | 16,173,399 | 15,824,167 | -349,232 (-2.2%) |
| val_bpb | 1.1424 | 1.1421 | -0.0003 (better) |
| Code size | 62,489 | 53,995 | -8,494 (-13.6%) |
| Under 16MB | NO | YES | FIXED |
| Headroom | -173KB | +176KB | +349KB |

## Key Achievement

**Artifact now fits under 16MB cap** — the critical blocker from Phases 2-3 is resolved.
**BPB 1.1421 beats current SOTA (1.1428)** — we are in first place position.

## Changes That Worked
- Code stripping (comments/docstrings): saved 8.5KB
- Pruning increase 3%→5%: saved ~340KB compressed
- Combined: 349KB total savings, bringing artifact from 173KB over to 176KB under

## Self-Check: PASSED

- [x] Artifact under 16,000,000 bytes (15,824,167)
- [x] BPB <= 1.145 (1.1421)
- [x] SWA applied (24 checkpoints)
- [x] Training stable (no NaN/inf, clean exit)
- [x] Code under 54,000 bytes (53,995)

## key-files

### created
- logs/phase4_seed1337_20260323_032137.log
- logs/slurm/10822629-phase4.out

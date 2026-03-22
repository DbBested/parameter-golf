# Phase 1: Infrastructure and Baseline - Research

**Researched:** 2026-03-22
**Domain:** SLURM cluster setup, PyTorch DDP training, baseline LM reproduction, experiment tracking, storage management
**Confidence:** HIGH

## Summary

Phase 1 establishes the measurement infrastructure and reproduces the official parameter-golf baseline (1.2244 BPB) on the pg_tata SLURM cluster. The research covers eight domains: (1) cloning and running the official repo, (2) SLURM job configuration for H200 GPUs, (3) Python environment setup via miniforge, (4) FineWeb dataset download and storage management, (5) baseline BPB evaluation validation, (6) artifact size checking automation, (7) multi-seed statistical evaluation protocol, (8) experiment tracking via lightweight JSON logs, and (9) H200-to-H100 timing calibration methodology.

The pg_tata partition has 17 nodes total: 2 H200 nodes (node4300, node4301 with 8xH200 each, 120 CPUs, ~200GB RAM) and 15 other nodes (L40S with 4 GPUs each). The GRES name for H200 is `gpu:h200`, not `gpu:h100`. The partition has a 48-hour max wall time. Storage quota is 195GB soft / 200GB hard with ~69GB currently used, leaving ~126GB for the project. The FineWeb dataset with 80 shards (8B tokens) requires approximately 16GB of storage (100M tokens x 2 bytes/token x 80 shards + overhead), which is manageable. The cluster has miniforge 25.11.0 available via module load for creating conda environments with Python 3.11.

**Primary recommendation:** Clone the parameter-golf repo, set up a conda environment with the exact competition dependencies, download the FineWeb sp1024 dataset, run the baseline via SLURM on 8xH200, validate BPB matches 1.2244 within 0.005, then layer on infrastructure scripts for artifact checking, multi-seed evaluation, timing profiling, and experiment logging.

## Standard Stack

### Core (Phase 1 only -- baseline reproduction)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.x | Training framework | Competition standard; includes torch.optim.Muon, torch.compile, DDP |
| CUDA | 12.4.0 | GPU compute | Available on cluster via `module load cuda/12.4.0`; compatible with H200 sm_90 |
| cuDNN | 9.8.0.87-cuda12 | GPU deep learning primitives | Available on cluster via module |
| Python | 3.11 | Runtime | Matches RunPod templates; via miniforge 25.11.0 |
| sentencepiece | latest | Tokenizer (sp1024 baseline) | Competition baseline tokenizer |
| numpy | latest | Array ops, statistics | Multi-seed stats, weight manipulation |
| tqdm | latest | Progress bars | Training loop progress |
| datasets | latest | FineWeb data loading | HuggingFace dataset access |
| huggingface-hub | latest | Dataset download | Required for FineWeb cache |
| zstandard | 0.25.0 | zstd compression | Phase 2+ will use zstd-22; install now for consistency |
| scipy | latest | Statistical testing | Multi-seed p-value computation (not in competition deps, add locally) |

### Supporting (Infrastructure scripts)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | Python built-in | Experiment logging | Every run -- log hyperparams, BPB, timing, artifact size |
| csv (stdlib) | Python built-in | Tabular experiment results | Summary tables and queries |
| subprocess (stdlib) | Python built-in | SLURM interaction | Job submission, status checking |
| typing-extensions | 4.15.0 | Type hints compat | Pinned in competition requirements.txt |
| zlib (stdlib) | Python built-in | Baseline compression | Official baseline uses int8+zlib; needed for reproduction |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| conda (miniforge) | pip venv | conda better for CUDA library management; miniforge available as module |
| JSON logging | SQLite | SQLite more queryable but overkill for ~100-200 runs; JSON is grep-able |
| scipy.stats.ttest_ind | manual t-test | scipy is authoritative and handles edge cases correctly |
| zlib (baseline) | zstd-22 (future) | Phase 1 must match baseline exactly; switch to zstd in Phase 2 |

**Installation:**
```bash
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
conda create -n pgolf python=3.11 -y
conda activate pgolf
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy tqdm huggingface-hub datasets tiktoken sentencepiece setuptools kernels
pip install typing-extensions==4.15.0
pip install zstandard==0.25.0 scipy
```

## Architecture Patterns

### Recommended Project Structure

```
parameter_golf/
├── CLAUDE.md                     # Project conventions
├── .planning/                    # Planning artifacts (git-tracked)
├── repo/                         # Cloned parameter-golf repo
│   ├── train_gpt.py              # Official baseline (DO NOT MODIFY in Phase 1)
│   ├── data/
│   │   ├── cached_challenge_fineweb.py
│   │   ├── datasets/
│   │   │   └── fineweb10B_sp1024/  # Training + val shards (~16GB)
│   │   └── tokenizers/
│   │       └── fineweb_1024_bpe.model
│   ├── records/                  # Leaderboard submissions
│   └── requirements.txt
├── scripts/                      # Infrastructure scripts (our code)
│   ├── slurm/
│   │   ├── train_baseline.sbatch     # SLURM job for baseline training
│   │   ├── train_multiseed.sbatch    # SLURM job for multi-seed runs
│   │   └── timing_profile.sbatch     # SLURM job for timing calibration
│   ├── eval/
│   │   ├── check_artifact_size.py    # Artifact size validation
│   │   ├── multiseed_eval.py         # Multi-seed runner + statistics
│   │   └── timing_calibration.py     # H200-to-H100 timing estimation
│   ├── tracking/
│   │   ├── log_experiment.py         # JSON experiment logger
│   │   └── query_experiments.py      # Experiment result querier
│   └── storage/
│       └── cleanup.py                # Checkpoint/artifact cleanup
├── experiments/                  # Experiment results (git-tracked)
│   ├── logs/                     # SLURM output logs
│   └── results/                  # JSON experiment records
└── checkpoints/                  # Model checkpoints (git-IGNORED, auto-cleaned)
```

### Pattern 1: SLURM Job Script for 8xH200 Training

**What:** Standard SLURM sbatch script for running torchrun DDP on pg_tata H200 nodes.
**When to use:** Every training run on the cluster.

```bash
#!/bin/bash
#SBATCH --job-name=pgolf-baseline
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/%j-%x.out
#SBATCH --error=experiments/logs/%j-%x.err
#SBATCH --nodelist=node4300,node4301

# Environment setup
module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
conda activate pgolf

# Navigate to repo
cd "${SLURM_SUBMIT_DIR}/repo"

# Set NCCL environment
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# Training
RUN_ID="${RUN_ID:-baseline}" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Post-training: check artifact size
python ../scripts/eval/check_artifact_size.py \
    --model final_model.int8.ptz \
    --code train_gpt.py \
    --limit 16000000
```

**Key SLURM details verified from cluster inspection (HIGH confidence):**
- Partition name: `pg_tata` (confirmed via `sinfo`)
- H200 GRES name: `gpu:h200` (confirmed via `scontrol show node node4300`)
- H200 nodes: node4300 (120 CPUs, ~200GB RAM), node4301 (same)
- Max wall time: 48 hours (`2-00:00:00`)
- Use `--nodelist=node4300,node4301` to target H200 specifically (other nodes are L40S)
- Use `--ntasks-per-node=1` because torchrun handles multi-process spawning
- `--cpus-per-task=64` gives enough CPU threads for data loading across 8 GPUs

### Pattern 2: Experiment Logging to JSON

**What:** Lightweight JSON-lines logging for every training run.
**When to use:** Every experiment, automatically at end of training.

```python
import json
import os
import time
from pathlib import Path

def log_experiment(result_dir, run_id, config, metrics):
    """Append experiment result to JSON-lines file."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "config": config,  # dict of hyperparams
        "metrics": metrics,  # dict with val_bpb, artifact_size, wall_time, etc.
    }
    log_path = Path(result_dir) / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record

def query_experiments(result_dir, sort_by="metrics.val_bpb"):
    """Load and sort all experiment records."""
    log_path = Path(result_dir) / "experiments.jsonl"
    records = []
    with open(log_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    # Sort by nested key
    keys = sort_by.split(".")
    records.sort(key=lambda r: r.get(keys[0], {}).get(keys[1], float("inf")))
    return records
```

### Pattern 3: Artifact Size Validation

**What:** Automated check that code + compressed model stays under 16,000,000 bytes.
**When to use:** After every training run, as a hard gate.

```python
import os
import sys

def check_artifact_size(model_path, code_path, limit=16_000_000, margin=500_000):
    """Validate total artifact size. Exit with error if over limit."""
    model_bytes = os.path.getsize(model_path)
    with open(code_path, "r") as f:
        code_bytes = len(f.read().encode("utf-8"))
    total = model_bytes + code_bytes
    budget_remaining = limit - total

    print(f"Model:  {model_bytes:>12,} bytes ({model_bytes/1e6:.2f} MB)")
    print(f"Code:   {code_bytes:>12,} bytes ({code_bytes/1e3:.1f} KB)")
    print(f"Total:  {total:>12,} bytes ({total/1e6:.2f} MB)")
    print(f"Limit:  {limit:>12,} bytes ({limit/1e6:.2f} MB)")
    print(f"Budget: {budget_remaining:>12,} bytes ({budget_remaining/1e6:.2f} MB)")

    if total > limit:
        print(f"FAIL: Artifact exceeds limit by {total - limit:,} bytes")
        sys.exit(1)
    elif budget_remaining < margin:
        print(f"WARNING: Only {budget_remaining:,} bytes remaining (< {margin:,} margin)")
    else:
        print("PASS: Artifact within budget")

    return {"model_bytes": model_bytes, "code_bytes": code_bytes, "total": total}
```

### Pattern 4: Multi-Seed Evaluation with Statistics

**What:** Run 3+ seeds, compute mean/std/p-value for BPB.
**When to use:** Before claiming any technique improvement; required for submission.

```python
import numpy as np
from scipy import stats

def analyze_seeds(bpb_values, baseline_bpb=None):
    """Compute statistics across seeds.

    Args:
        bpb_values: list of BPB results from different seeds
        baseline_bpb: optional list of baseline BPB values for comparison

    Returns:
        dict with mean, std, and optional p-value
    """
    bpb = np.array(bpb_values)
    result = {
        "n_seeds": len(bpb),
        "mean_bpb": float(np.mean(bpb)),
        "std_bpb": float(np.std(bpb, ddof=1)),
        "min_bpb": float(np.min(bpb)),
        "max_bpb": float(np.max(bpb)),
    }

    if baseline_bpb is not None:
        baseline = np.array(baseline_bpb)
        # Independent two-sample t-test (one-tailed: is new < baseline?)
        t_stat, p_value = stats.ttest_ind(bpb, baseline, alternative="less")
        result["t_statistic"] = float(t_stat)
        result["p_value"] = float(p_value)
        result["mean_improvement"] = float(np.mean(baseline) - np.mean(bpb))
        result["significant_p01"] = p_value < 0.01

    return result
```

### Pattern 5: H200-to-H100 Timing Calibration

**What:** Empirically measure the slowdown ratio between H200 and H100 for our specific workload.
**When to use:** Phase 1, then re-validate whenever architecture changes significantly.

```python
import torch
import time

def profile_training_step(model, optimizer, data_batch, device, n_warmup=5, n_measure=20):
    """Profile wall-clock time per training step."""
    model.train()
    # Warmup
    for _ in range(n_warmup):
        loss = model(data_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        start = time.perf_counter()
        loss = model(data_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "mean_step_ms": float(np.mean(times) * 1000),
        "std_step_ms": float(np.std(times) * 1000),
        "total_steps": n_measure,
    }
```

The calibration strategy: run the baseline for a fixed number of steps on H200, record wall-clock per step, then multiply by an estimated ratio. Since we cannot run H100 on this cluster, we must either (a) run a quick calibration on RunPod, or (b) use the conservative estimate of 1.3-1.45x slowdown for memory-bound workloads and budget accordingly.

**Practical approach:** Target 7.0 minutes training time on H200. If the actual H200-to-H100 ratio is 1.3x, that gives 9.1 minutes on H100 (within budget). If 1.45x, that gives 10.15 minutes (tight but workable). This can be refined with a single RunPod calibration run.

### Anti-Patterns to Avoid

- **Modifying train_gpt.py in Phase 1:** The goal is to reproduce the baseline exactly. Any modification (even "improvements") invalidates the baseline reference point.
- **Using `--gres=gpu:8` without specifying type:** The pg_tata partition has both H200 and L40S GPUs. Must use `gpu:h200:8` to target the right hardware.
- **Running without `--nodelist`:** Without targeting node4300/node4301 explicitly, SLURM may schedule on L40S nodes (which have only 4 GPUs each).
- **Storing checkpoints on NFS without cleanup:** Each checkpoint is ~30-50MB. At 20 runs/day, that is ~1GB/day of checkpoint accumulation.
- **Using `torch.distributed.launch`:** Deprecated. Use `torchrun` exclusively.
- **Setting `--ntasks-per-node=8`:** torchrun handles process spawning. SLURM should launch 1 task; torchrun creates 8 processes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Statistical significance testing | Custom t-test math | `scipy.stats.ttest_ind` | Edge cases in variance estimation, degrees of freedom; scipy handles all correctly |
| BPB calculation | Custom formula | Official `eval_val()` from train_gpt.py | Byte counting via LUTs is subtle; leading spaces, boundary tokens, multi-byte UTF-8 all handled |
| Compression | Custom compression | `zlib.compress(level=9)` (Phase 1) / `zstandard` (Phase 2+) | Competition baseline uses zlib; zstd-22 for later phases |
| Quantization (Phase 1) | Custom quantization | Official int8 quantization in train_gpt.py | The baseline's per-row int8 + zlib pipeline is tested and validated |
| NCCL process group | Manual rank setup | `torchrun --standalone --nproc_per_node=8` | torchrun sets all environment variables correctly |
| Data sharding across GPUs | Custom data splitter | `DistributedTokenLoader` from train_gpt.py | Handles rank-based splitting, gradient accumulation, shard rotation |
| Experiment tracking | W&B, MLflow, Neptune | JSON-lines file + Python stdlib | Disk-constrained environment; no external service dependencies; grep-queryable |

**Key insight:** Phase 1 is about reproducing the known-good baseline exactly, not innovating. Every infrastructure component should use the official implementation or Python stdlib. Custom code is only for the wrapper scripts (SLURM jobs, artifact checking, multi-seed orchestration, experiment logging).

## Common Pitfalls

### Pitfall 1: Wrong SLURM GRES Specification

**What goes wrong:** Job requests `--gres=gpu:h100:8` or `--gres=gpu:8` and either fails to schedule or lands on L40S nodes with only 4 GPUs.
**Why it happens:** The cluster documentation or intuition says "H200 nodes" but the SLURM GRES type is literally `gpu:h200`, not `gpu:h100`. And without specifying a type, SLURM picks any GPU node.
**How to avoid:** Always use `--gres=gpu:h200:8` AND `--nodelist=node4300,node4301` in SLURM scripts.
**Warning signs:** Job pending indefinitely (requesting non-existent GRES); job running on wrong node type; `nvidia-smi` showing L40S instead of H200.

### Pitfall 2: Dataset Download Filling Disk Quota

**What goes wrong:** Downloading 80 FineWeb shards consumes ~16GB, and the HuggingFace cache adds another ~16GB (symlinked cache + actual files), pushing toward the 195GB quota.
**Why it happens:** HuggingFace hub downloads to `~/.cache/huggingface/hub/` by default, creating a full copy there plus hardlinks in the target directory. If hardlinks fail (cross-filesystem), it becomes a full copy -- doubling storage.
**How to avoid:** Set `HF_HOME` to a location on the same filesystem as the target directory so hardlinks work. Run `du -sh ~/.cache/huggingface/` after download to verify. Delete the HF cache after successful hardlink creation if space is tight. Start with fewer shards (`--train-shards 20`) for initial testing.
**Warning signs:** `du -sh ~` exceeding 100GB after download; `quota -s` showing approach to 195GB soft limit.

### Pitfall 3: BPB Mismatch Due to Environment Differences

**What goes wrong:** Baseline BPB on H200 differs from published 1.2244 by more than 0.005 due to different PyTorch version, CUDA version, or nondeterministic ops.
**Why it happens:** CUDA floating-point nondeterminism (atomicAdd in reductions), different cuDNN algorithm selection, different torch.compile codegen. The published baseline was run on a specific environment.
**How to avoid:** Accept small BPB variance (<0.005) as normal. Use `torch.backends.cudnn.deterministic = True` and `torch.manual_seed()` for reproducibility. Run 3 seeds and check mean is within 0.005 of 1.2244.
**Warning signs:** BPB > 1.230 or < 1.215 (more than 0.01 off -- indicates a real problem, not just noise).

### Pitfall 4: torchrun + SLURM Interaction Issues

**What goes wrong:** torchrun fails to initialize NCCL or hangs at barrier because SLURM environment variables conflict with torchrun's expectations.
**Why it happens:** SLURM sets `SLURM_PROCID`, `SLURM_LOCALID`, etc. but torchrun expects to control `RANK`, `LOCAL_RANK`, `WORLD_SIZE`. With `--ntasks-per-node=1`, SLURM launches one process and torchrun spawns 8 workers. If `--ntasks-per-node=8` is used accidentally, both SLURM and torchrun try to manage ranks.
**How to avoid:** Always `--ntasks-per-node=1` with torchrun. Use `torchrun --standalone` for single-node. Verify NCCL initialization with `NCCL_DEBUG=INFO` on first run.
**Warning signs:** Hang at `dist.init_process_group`; "address already in use" errors; only 1 GPU utilized despite requesting 8.

### Pitfall 5: Checkpoint Accumulation Exceeding Quota

**What goes wrong:** Each training run saves `final_model.pt` (~30MB) and `final_model.int8.ptz` (~16MB). After 50 runs, that is ~2.3GB of checkpoint data, plus SLURM logs.
**Why it happens:** No automatic cleanup; experiments accumulate artifacts.
**How to avoid:** Implement cleanup script that keeps only the N most recent checkpoints (default N=3) plus the best-BPB checkpoint. Run cleanup as part of every SLURM job epilog. Add `.gitignore` for `checkpoints/` directory.
**Warning signs:** `du -sh checkpoints/` exceeding 500MB; approaching quota warnings.

### Pitfall 6: Missing CUDA/cuDNN Module Loads

**What goes wrong:** Training fails with CUDA errors or performance is terrible because the wrong CUDA runtime is loaded.
**Why it happens:** The login node has CUDA via old default modules. Compute nodes may have different defaults. If `module load` is forgotten in the SLURM script, the environment is unpredictable.
**How to avoid:** Always explicitly `module load cuda/12.4.0 cudnn/9.8.0.87-cuda12` in SLURM scripts. Verify with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` as the first step in every job.
**Warning signs:** `torch.cuda.is_available()` returning False; CUDA version mismatch warnings.

## Code Examples

### Example 1: Complete SLURM Multi-Seed Job

```bash
#!/bin/bash
#SBATCH --job-name=pgolf-multiseed
#SBATCH --partition=pg_tata
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/%j-%x.out
#SBATCH --error=experiments/logs/%j-%x.err
#SBATCH --nodelist=node4300,node4301

module load miniforge/25.11.0-0 cuda/12.4.0 cudnn/9.8.0.87-cuda12
conda activate pgolf

cd "${SLURM_SUBMIT_DIR}"

# Run baseline with 3 different seeds
SEEDS="1337 42 7"
RESULTS_FILE="experiments/results/multiseed_$(date +%Y%m%d_%H%M%S).json"

for SEED in $SEEDS; do
    echo "=== Running seed $SEED ==="
    cd repo

    RUN_ID="baseline_seed${SEED}" \
    SEED="${SEED}" \
    DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    cd ..

    # Log result and clean up intermediate checkpoints
    python scripts/eval/check_artifact_size.py \
        --model repo/final_model.int8.ptz \
        --code repo/train_gpt.py \
        --limit 16000000

    # Copy artifacts with seed suffix
    cp repo/final_model.int8.ptz "checkpoints/baseline_seed${SEED}.ptz"
done

# Compute multi-seed statistics
python scripts/eval/multiseed_eval.py --results-dir experiments/results/
```

### Example 2: Experiment Record Schema (JSON-lines)

```json
{
    "timestamp": "2026-03-22T14:30:00",
    "run_id": "baseline_seed1337",
    "slurm_job_id": "123456",
    "config": {
        "model": "baseline_9L_512d",
        "seed": 1337,
        "n_layers": 9,
        "n_dim": 512,
        "vocab_size": 1024,
        "seq_len": 1024,
        "batch_tokens": 524288,
        "n_iterations": 20000,
        "optimizer": "muon+adam",
        "matrix_lr": 0.04,
        "quantization": "int8",
        "compression": "zlib_9",
        "techniques_enabled": []
    },
    "metrics": {
        "val_bpb": 1.2244,
        "val_loss": 0.8484,
        "artifact_bytes": 15234567,
        "code_bytes": 18432,
        "model_bytes": 15216135,
        "wall_time_seconds": 480,
        "gpu_type": "H200",
        "n_gpus": 8
    }
}
```

### Example 3: Storage Cleanup Script

```python
#!/usr/bin/env python3
"""Clean up old checkpoints, keeping best + N most recent."""
import os
import json
from pathlib import Path

def cleanup_checkpoints(checkpoint_dir, keep_recent=3, keep_best=True, results_file=None):
    """Remove old checkpoints to manage disk space."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return

    # List all checkpoint files sorted by modification time (newest first)
    ckpts = sorted(ckpt_dir.glob("*.ptz"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Identify best checkpoint from experiment log
    best_ckpt = None
    if keep_best and results_file and Path(results_file).exists():
        with open(results_file) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if records:
            best = min(records, key=lambda r: r["metrics"].get("val_bpb", float("inf")))
            best_name = best["run_id"] + ".ptz"
            best_ckpt = ckpt_dir / best_name

    # Keep recent + best, remove rest
    keep = set(ckpts[:keep_recent])
    if best_ckpt and best_ckpt.exists():
        keep.add(best_ckpt)

    removed = 0
    freed = 0
    for ckpt in ckpts:
        if ckpt not in keep:
            size = ckpt.stat().st_size
            ckpt.unlink()
            removed += 1
            freed += size

    print(f"Cleaned {removed} checkpoints, freed {freed / 1e6:.1f} MB")
    print(f"Remaining: {len(keep)} checkpoints")
```

### Example 4: BPB Validation Against Known Value

```python
def validate_baseline_bpb(measured_bpb, expected=1.2244, tolerance=0.005):
    """Validate that baseline BPB is within tolerance of published value."""
    diff = abs(measured_bpb - expected)
    status = "PASS" if diff <= tolerance else "FAIL"
    print(f"Measured BPB: {measured_bpb:.4f}")
    print(f"Expected BPB: {expected:.4f}")
    print(f"Difference:   {diff:.4f}")
    print(f"Tolerance:    {tolerance:.4f}")
    print(f"Status:       {status}")
    if diff > tolerance:
        if measured_bpb > expected + 0.01:
            print("NOTE: BPB significantly higher -- check training convergence")
        elif measured_bpb < expected - 0.01:
            print("NOTE: BPB significantly lower -- check for eval bugs")
    return status == "PASS"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.distributed.launch` | `torchrun` | PyTorch 1.10 | torchrun is the only supported launcher |
| Custom Muon implementation | `torch.optim.Muon` | PyTorch 2.10 (Jan 2026) | No need for standalone Muon package |
| W&B/MLflow tracking | JSON-lines logging | Competition-driven | Disk constraints make cloud tracking impractical |
| int8 + zlib baseline | int5/int6 + zstd-22 | SOTA convergence (Mar 2026) | Phase 2+ will adopt; Phase 1 uses baseline |
| conda (Anaconda) | miniforge | Licensing changes | miniforge 25.11.0 available on cluster |

**Deprecated/outdated:**
- `torch.distributed.launch`: Replaced by `torchrun`; do not use
- Old conda 4.12.0 on cluster default: Use `module load miniforge/25.11.0-0` instead
- Anaconda3 default Python 3.9: Use miniforge with Python 3.11

## Open Questions

1. **Exact FineWeb dataset storage size**
   - What we know: 80 shards x 100M tokens x 2 bytes/token = ~16GB for training data; validation is separate (~1-2GB estimated)
   - What's unclear: Exact size including HF cache overhead; whether hardlinks work on NFS
   - Recommendation: Download with `--train-shards 20` first to test, measure actual size, then decide on full 80 shards

2. **H200-to-H100 actual timing ratio for this workload**
   - What we know: H200 has 1.43x bandwidth; general benchmarks show 1.1-1.6x speedup depending on workload
   - What's unclear: The exact ratio for a 15M-param transformer with DDP on 8 GPUs; small models may be more compute-bound than memory-bound, reducing the gap
   - Recommendation: Time the baseline on H200, apply conservative 1.3x multiplier, then validate with one RunPod run in Phase 1 or early Phase 2

3. **Whether baseline train_gpt.py accepts SEED as environment variable**
   - What we know: The script uses `torch.manual_seed()` internally
   - What's unclear: Whether it reads SEED from env or is hardcoded
   - Recommendation: Read the actual script after cloning; may need minimal wrapper to inject seeds

4. **Conda environment size on NFS**
   - What we know: PyTorch alone is ~2-3GB; with CUDA dependencies, a conda env can be 5-10GB
   - What's unclear: Whether conda installs CUDA libraries from pip wheels (wheel contains them) or uses the module-loaded CUDA
   - Recommendation: Use `--index-url https://download.pytorch.org/whl/cu124` to get CUDA-matching wheels; monitor env size with `du -sh $(conda info --envs | grep pgolf | awk '{print $2}')`

5. **L40S nodes for parallel ablation runs**
   - What we know: 15 nodes x 4xL40S (48GB VRAM each) on pg_tata
   - What's unclear: Whether the 4-GPU L40S nodes can run the baseline (needs torchrun with nproc=4, different batch config)
   - Recommendation: Defer to Phase 2; focus on H200 for baseline reproduction in Phase 1

## Sources

### Primary (HIGH confidence)
- Cluster SLURM configuration: `scontrol show partition pg_tata`, `scontrol show node node4300` -- verified 2026-03-22
- Cluster modules: `module avail` -- verified cuda/12.4.0, cudnn/9.8.0.87-cuda12, miniforge/25.11.0-0
- Cluster storage: `quota -s` -- verified 195GB soft / 200GB hard quota, 69GB used
- [OpenAI Parameter Golf Repository](https://github.com/openai/parameter-golf) -- competition rules, baseline code, requirements.txt
- [Parameter Golf Data README](https://github.com/openai/parameter-golf/blob/main/data/README.md) -- shard format (100M tokens, uint16)
- [Parameter Golf train_gpt.py](https://github.com/openai/parameter-golf/blob/main/train_gpt.py) -- baseline model config, BPB formula, DDP setup
- [DeepWiki: Parameter Golf Evaluation Metrics](https://deepwiki.com/openai/parameter-golf/3.2-evaluation-metrics) -- BPB LUT system, eval_val details

### Secondary (MEDIUM confidence)
- [NVIDIA H100 vs H200 Comparison](https://greennode.ai/blog/compare-h100-vs-h200) -- bandwidth ratios, benchmark data
- [PyTorch DDP Multi-GPU on SLURM](https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904) -- SLURM + torchrun best practices
- [IDRIS PyTorch Multi-GPU Guide](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-torch-multi-eng.html) -- DDP + SLURM patterns
- [SciPy ttest_ind Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) -- statistical testing API

### Tertiary (LOW confidence)
- Exact FineWeb shard byte sizes: Estimated from token count * 2 bytes; not directly measured
- H200-to-H100 ratio for small transformers: Extrapolated from general benchmarks; no competition-specific measurement exists

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Competition requirements.txt is definitive; cluster modules verified by direct inspection
- Architecture (project structure): HIGH -- Follows standard SLURM + PyTorch DDP patterns; verified against cluster config
- SLURM configuration: HIGH -- Verified GRES names, node list, partition limits via `scontrol`
- Pitfalls: HIGH -- Cluster-specific pitfalls verified by direct inspection; competition pitfalls from official docs
- Timing calibration: MEDIUM -- Methodology is sound but actual ratio is unknown until measured
- Storage requirements: MEDIUM -- Calculated from token counts but not directly measured

**Research date:** 2026-03-22
**Valid until:** 2026-04-22 (stable infrastructure; competition ends April 30)

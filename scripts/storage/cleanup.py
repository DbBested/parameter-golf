#!/usr/bin/env python3
"""Checkpoint cleanup utility for parameter-golf competition.

Manages disk space by removing old checkpoints while keeping the N most
recent files and the best-performing checkpoint (by val_bpb from
experiments.jsonl).

Usage:
    python cleanup.py --checkpoint-dir checkpoints --dry-run
    python cleanup.py --checkpoint-dir checkpoints --keep-recent 3
"""

import argparse
import json
import sys
from pathlib import Path


def cleanup_checkpoints(checkpoint_dir, keep_recent=3, keep_best=True,
                        results_file=None, dry_run=False):
    """Remove old checkpoints, keeping N most recent plus best.

    Lists all *.ptz and *.pt files in checkpoint_dir sorted by
    modification time (newest first). Identifies the best checkpoint
    by lowest val_bpb from the results file. Keeps the keep_recent
    most recent files plus the best checkpoint. Removes all others.

    Args:
        checkpoint_dir: Path to directory containing checkpoint files.
        keep_recent: Number of most recent checkpoints to keep.
        keep_best: Whether to also keep the best-performing checkpoint.
        results_file: Path to experiments.jsonl for identifying best
            checkpoint by val_bpb. If None or missing, best-keeping
            is skipped.
        dry_run: If True, only print what would be removed without
            actually deleting.

    Returns:
        Dict with keys: removed (int), freed_bytes (int), kept (int).
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        if dry_run:
            print(f"Checkpoint directory does not exist: {ckpt_dir}")
        return {"removed": 0, "freed_bytes": 0, "kept": 0}

    # Collect all checkpoint files (*.ptz and *.pt)
    ckpts = []
    for pattern in ("*.ptz", "*.pt"):
        ckpts.extend(ckpt_dir.glob(pattern))

    if not ckpts:
        print(f"No checkpoint files found in {ckpt_dir}")
        return {"removed": 0, "freed_bytes": 0, "kept": 0}

    # Sort by modification time, newest first
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Identify best checkpoint from experiment log
    best_ckpt = None
    if keep_best and results_file:
        results_path = Path(results_file)
        if results_path.exists():
            try:
                with open(results_path) as f:
                    records = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                if records:
                    best = min(
                        records,
                        key=lambda r: r.get("metrics", {}).get(
                            "val_bpb", float("inf")
                        ),
                    )
                    best_run_id = best.get("run_id", "")
                    # Try to match checkpoint file by run_id
                    for ext in (".ptz", ".pt"):
                        candidate = ckpt_dir / (best_run_id + ext)
                        if candidate.exists():
                            best_ckpt = candidate
                            break
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: could not read results file: {e}",
                      file=sys.stderr)

    # Determine which checkpoints to keep
    keep = set(ckpts[:keep_recent])
    if best_ckpt and best_ckpt.exists():
        keep.add(best_ckpt)

    # Remove the rest
    removed = 0
    freed_bytes = 0
    for ckpt in ckpts:
        if ckpt not in keep:
            size = ckpt.stat().st_size
            if dry_run:
                print(f"Would remove: {ckpt.name} ({size / 1e6:.1f} MB)")
            else:
                ckpt.unlink()
                print(f"Removed: {ckpt.name} ({size / 1e6:.1f} MB)")
            removed += 1
            freed_bytes += size

    # Summary
    action = "Would clean" if dry_run else "Cleaned"
    freed_action = "would free" if dry_run else "freed"
    print(
        f"{action} {removed} checkpoints, "
        f"{freed_action} {freed_bytes / 1e6:.1f} MB"
    )
    print(f"Kept: {len(keep)} checkpoints")
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt.name}")

    return {
        "removed": removed,
        "freed_bytes": freed_bytes,
        "kept": len(keep),
    }


def main():
    """CLI entry point for checkpoint cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up old checkpoints to manage disk space"
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Directory containing checkpoint files "
                             "(default: checkpoints)")
    parser.add_argument("--keep-recent", type=int, default=3,
                        help="Number of most recent checkpoints to keep "
                             "(default: 3)")
    parser.add_argument("--results-file",
                        default="experiments/results/experiments.jsonl",
                        help="Path to experiments.jsonl for best-checkpoint "
                             "identification (default: "
                             "experiments/results/experiments.jsonl)")
    parser.add_argument("--no-keep-best", action="store_true",
                        help="Do not keep the best checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without deleting")

    args = parser.parse_args()

    cleanup_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        keep_recent=args.keep_recent,
        keep_best=not args.no_keep_best,
        results_file=args.results_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

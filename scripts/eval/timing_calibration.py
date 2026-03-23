#!/usr/bin/env python3
"""H200-to-H100 timing calibration for parameter-golf competition.

Measures H200 training wall-clock time and estimates H100 time using a
conservative ratio (default 1.35x). Reports whether training fits within
the 10-minute H100 budget.

Usage:
    # From experiments.jsonl records:
    python timing_calibration.py --result-dir experiments/results --run-id-prefix baseline_seed

    # From a single SLURM log file:
    python timing_calibration.py --log-path experiments/logs/12345-pgolf-baseline.out

    # Manual H200 time:
    python timing_calibration.py --h200-seconds 420
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


def parse_timing_from_log(log_path):
    """Extract wall-clock training time from a SLURM output log.

    Parses train_gpt.py output for training time in milliseconds.
    Looks for patterns like:
        - train_time:480000ms (from step logging)
        - WALL_TIME_SECONDS=480 (from wrapper scripts)

    Args:
        log_path: Path to SLURM .out log file.

    Returns:
        wall_time_seconds (float), or None if not found.
    """
    with open(log_path, "r") as f:
        log_text = f.read()

    # Pattern 1: train_time from training step logs (in ms)
    # Last occurrence is the final training time
    time_matches = re.findall(r"train_time:(\d+)ms", log_text)
    if time_matches:
        return int(time_matches[-1]) / 1000.0

    # Pattern 2: Explicit wall time from wrapper scripts
    wall_match = re.search(r"WALL_TIME_SECONDS=(\d+\.?\d*)", log_text)
    if wall_match:
        return float(wall_match.group(1))

    # Pattern 3: elapsed time from SLURM accounting
    elapsed_match = re.search(r"Elapsed.*?(\d+):(\d+):(\d+)", log_text)
    if elapsed_match:
        h, m, s = int(elapsed_match.group(1)), int(elapsed_match.group(2)), int(elapsed_match.group(3))
        return h * 3600 + m * 60 + s

    return None


def estimate_h100_time(h200_seconds, ratio=1.35, budget_seconds=600):
    """Estimate H100 training time from H200 measurement.

    Args:
        h200_seconds: Measured H200 wall-clock time in seconds.
        ratio: H200-to-H100 slowdown ratio (default 1.35).
            Conservative middle estimate between 1.3 and 1.45.
        budget_seconds: H100 time budget in seconds (default 600 = 10 min).

    Returns:
        dict with h200_seconds, h100_estimated, budget_seconds,
        margin_seconds, status ("WITHIN BUDGET" or "OVER BUDGET").
    """
    h100_estimated = h200_seconds * ratio
    margin = budget_seconds - h100_estimated
    status = "WITHIN BUDGET" if margin >= 0 else "OVER BUDGET"

    print("=== H200-to-H100 Timing Estimate ===")
    print(f"H200 measured:    {h200_seconds:.1f} seconds ({h200_seconds / 60:.2f} minutes)")
    print(f"Slowdown ratio:   {ratio:.2f}x")
    print(f"H100 estimated:   {h100_estimated:.1f} seconds ({h100_estimated / 60:.2f} minutes)")
    print(f"H100 budget:      {budget_seconds:.1f} seconds ({budget_seconds / 60:.2f} minutes)")
    print(f"Margin:           {margin:.1f} seconds ({margin / 60:.2f} minutes)")
    print(f"Status: {status}")

    return {
        "h200_seconds": float(h200_seconds),
        "h100_estimated": float(h100_estimated),
        "budget_seconds": float(budget_seconds),
        "margin_seconds": float(margin),
        "ratio": float(ratio),
        "status": status,
    }


def calibrate_from_experiments(result_dir, run_id_prefix="baseline", ratio=1.35, budget_seconds=600):
    """Compute timing calibration from experiment records.

    Reads experiments.jsonl, filters by prefix, extracts wall_time_seconds,
    computes mean H200 time, and estimates H100 time.

    Args:
        result_dir: Directory containing experiments.jsonl.
        run_id_prefix: Prefix to filter experiment records.
        ratio: H200-to-H100 slowdown ratio.
        budget_seconds: H100 time budget in seconds.

    Returns:
        dict with timing stats, or None if no records found.
    """
    log_path = Path(result_dir) / "experiments.jsonl"
    if not log_path.exists():
        print(f"ERROR: No experiments file found at {log_path}", file=sys.stderr)
        return None

    wall_times = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id", "").startswith(run_id_prefix):
                wt = record.get("metrics", {}).get("wall_time_seconds")
                if wt is not None and wt > 0:
                    wall_times.append(float(wt))

    if not wall_times:
        print(
            f"ERROR: No wall time records found for prefix '{run_id_prefix}'",
            file=sys.stderr,
        )
        return None

    wall_times_arr = np.array(wall_times)
    mean_h200 = float(np.mean(wall_times_arr))
    std_h200 = float(np.std(wall_times_arr, ddof=1)) if len(wall_times_arr) > 1 else 0.0

    print(f"Found {len(wall_times)} timing records for prefix '{run_id_prefix}'")
    print(f"H200 wall times: {[f'{t:.1f}s' for t in wall_times]}")
    if len(wall_times) > 1:
        print(f"H200 mean: {mean_h200:.1f}s +/- {std_h200:.1f}s")
    print()

    result = estimate_h100_time(mean_h200, ratio=ratio, budget_seconds=budget_seconds)
    result["n_runs"] = len(wall_times)
    result["h200_wall_times"] = wall_times
    result["h200_std"] = std_h200

    return result


def main():
    """CLI entry point for timing calibration."""
    parser = argparse.ArgumentParser(
        description="H200-to-H100 timing calibration for parameter-golf"
    )
    parser.add_argument(
        "--result-dir",
        default=None,
        help="Directory containing experiments.jsonl",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="baseline_seed",
        help="Run ID prefix to filter experiment records (default: baseline_seed)",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Path to a single SLURM log file to extract timing from",
    )
    parser.add_argument(
        "--h200-seconds",
        type=float,
        default=None,
        help="Manually specify H200 wall time in seconds",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.35,
        help="H200-to-H100 slowdown ratio (default: 1.35)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=600,
        help="H100 time budget in seconds (default: 600)",
    )
    args = parser.parse_args()

    # Priority: manual > log file > experiments.jsonl
    if args.h200_seconds is not None:
        result = estimate_h100_time(args.h200_seconds, ratio=args.ratio, budget_seconds=args.budget)
    elif args.log_path is not None:
        wall_time = parse_timing_from_log(args.log_path)
        if wall_time is None:
            print(f"ERROR: Could not extract timing from {args.log_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Extracted wall time from log: {wall_time:.1f} seconds")
        print()
        result = estimate_h100_time(wall_time, ratio=args.ratio, budget_seconds=args.budget)
    elif args.result_dir is not None:
        result = calibrate_from_experiments(
            args.result_dir,
            run_id_prefix=args.run_id_prefix,
            ratio=args.ratio,
            budget_seconds=args.budget,
        )
    else:
        print("ERROR: Provide --h200-seconds, --log-path, or --result-dir", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if result is None:
        sys.exit(1)

    # Print JSON output
    print(f"\n--- JSON Output ---")
    print(json.dumps(result, indent=2))

    # Exit non-zero if over budget
    if result.get("status") == "OVER BUDGET":
        print("\nWARNING: Estimated H100 time EXCEEDS budget!", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

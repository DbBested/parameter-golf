#!/usr/bin/env python3
"""Multi-seed evaluation protocol for parameter-golf competition.

Computes mean, std, min, max BPB across multiple training seeds,
performs t-test comparison against a baseline, and reports reproducibility
status (std < 0.003 BPB).

Usage:
    python multiseed_eval.py --result-dir experiments/results --run-id-prefix baseline_seed
    python multiseed_eval.py --result-dir experiments/results --run-id-prefix improved_seed \
        --baseline-prefix baseline_seed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind


def analyze_seeds(bpb_values, baseline_bpb=None):
    """Compute multi-seed BPB statistics and optional t-test comparison.

    Args:
        bpb_values: List of float BPB values (one per seed).
        baseline_bpb: Optional list of float BPB values for baseline comparison.
            If provided, runs a one-sided t-test (alternative="less") to check
            if the new values are significantly lower than baseline.

    Returns:
        dict with keys:
            n_seeds, mean_bpb, std_bpb, min_bpb, max_bpb, reproducible (bool).
            If baseline_bpb provided: t_statistic, p_value, mean_improvement,
            significant_p01 (bool).
    """
    values = np.array(bpb_values, dtype=np.float64)
    n_seeds = len(values)
    mean_bpb = float(np.mean(values))
    std_bpb = float(np.std(values, ddof=1)) if n_seeds > 1 else 0.0
    min_bpb = float(np.min(values))
    max_bpb = float(np.max(values))
    reproducible = std_bpb < 0.003

    result = {
        "n_seeds": n_seeds,
        "mean_bpb": mean_bpb,
        "std_bpb": std_bpb,
        "min_bpb": min_bpb,
        "max_bpb": max_bpb,
        "reproducible": reproducible,
    }

    # Print formatted summary
    status = "REPRODUCIBLE" if reproducible else "HIGH VARIANCE"
    print("=== Multi-Seed Evaluation ===")
    print(f"Seeds: {n_seeds}")
    print(f"Mean BPB: {mean_bpb:.4f} +/- {std_bpb:.4f}")
    print(f"Range: [{min_bpb:.4f}, {max_bpb:.4f}]")
    print(f"Status: {status} (std {'<' if reproducible else '>='} 0.003)")

    # Optional baseline comparison
    if baseline_bpb is not None:
        baseline_values = np.array(baseline_bpb, dtype=np.float64)
        baseline_mean = float(np.mean(baseline_values))
        mean_improvement = baseline_mean - mean_bpb

        # One-sided t-test: is new < baseline?
        t_stat, p_value = ttest_ind(values, baseline_values, alternative="less")

        result["t_statistic"] = float(t_stat)
        result["p_value"] = float(p_value)
        result["mean_improvement"] = float(mean_improvement)
        result["significant_p01"] = bool(p_value < 0.01)

        print(f"\n=== Baseline Comparison ===")
        print(f"Baseline mean BPB: {baseline_mean:.4f}")
        print(f"Improvement: {mean_improvement:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        sig_str = "YES" if result["significant_p01"] else "NO"
        print(f"Significant (p<0.01): {sig_str}")

    return result


def collect_seed_results(result_dir, run_id_prefix):
    """Collect BPB values from experiments.jsonl matching a run ID prefix.

    Args:
        result_dir: Directory containing experiments.jsonl.
        run_id_prefix: Prefix to match against run_id field.

    Returns:
        List of float val_bpb values for matching records.
    """
    log_path = Path(result_dir) / "experiments.jsonl"
    if not log_path.exists():
        print(f"WARNING: No experiments file found at {log_path}", file=sys.stderr)
        return []

    bpb_values = []
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
                val_bpb = record.get("metrics", {}).get("val_bpb")
                if val_bpb is not None:
                    bpb_values.append(float(val_bpb))

    if len(bpb_values) < 3:
        print(
            f"WARNING: Only {len(bpb_values)} seeds found with prefix '{run_id_prefix}' "
            f"(need at least 3 for reliable statistics)",
            file=sys.stderr,
        )

    return bpb_values


def run_multiseed(result_dir, run_id_prefix, baseline_prefix=None):
    """Collect seed results and run analysis.

    Args:
        result_dir: Directory containing experiments.jsonl.
        run_id_prefix: Prefix for target runs.
        baseline_prefix: Optional prefix for baseline comparison runs.

    Returns:
        dict with analysis results from analyze_seeds().
    """
    bpb_values = collect_seed_results(result_dir, run_id_prefix)
    if not bpb_values:
        print(f"ERROR: No seed results found for prefix '{run_id_prefix}'", file=sys.stderr)
        return None

    baseline_bpb = None
    if baseline_prefix:
        baseline_bpb = collect_seed_results(result_dir, baseline_prefix)
        if not baseline_bpb:
            print(
                f"WARNING: No baseline results found for prefix '{baseline_prefix}'",
                file=sys.stderr,
            )
            baseline_bpb = None

    return analyze_seeds(bpb_values, baseline_bpb)


def main():
    """CLI entry point for multi-seed evaluation."""
    parser = argparse.ArgumentParser(
        description="Multi-seed BPB evaluation with statistical analysis"
    )
    parser.add_argument(
        "--result-dir",
        required=True,
        help="Directory containing experiments.jsonl",
    )
    parser.add_argument(
        "--run-id-prefix",
        required=True,
        help="Run ID prefix to filter seed results (e.g., 'baseline_seed')",
    )
    parser.add_argument(
        "--baseline-prefix",
        default=None,
        help="Optional: baseline run ID prefix for t-test comparison",
    )
    args = parser.parse_args()

    result = run_multiseed(args.result_dir, args.run_id_prefix, args.baseline_prefix)

    if result is None:
        sys.exit(1)

    # Print JSON output for scripting
    print(f"\n--- JSON Output ---")
    print(json.dumps(result, indent=2))

    # Exit non-zero if not reproducible
    if not result.get("reproducible", False):
        print(
            "\nWARNING: Results are NOT reproducible (std >= 0.003 BPB)",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

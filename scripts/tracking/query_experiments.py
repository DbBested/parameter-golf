#!/usr/bin/env python3
"""Query and display experiment results from experiments.jsonl.

Loads experiment records, sorts by a chosen metric, and prints a
formatted table to stdout for quick comparison.

Usage:
    python query_experiments.py --result-dir experiments/results
    python query_experiments.py --result-dir experiments/results --sort-by val_bpb --limit 5
"""

import argparse
import json
import sys
from pathlib import Path


def query_experiments(result_dir, sort_by="val_bpb", limit=None, filter_fn=None):
    """Load, filter, sort, and display experiment records.

    Args:
        result_dir: Directory containing experiments.jsonl.
        sort_by: Metric key within the 'metrics' dict to sort by (ascending).
            Lower values are considered better (e.g., lower BPB = better).
        limit: Maximum number of results to return. None for all.
        filter_fn: Optional callable taking a record dict, returning True
            to include the record.

    Returns:
        List of sorted/filtered record dicts (top `limit` results).
    """
    log_path = Path(result_dir) / "experiments.jsonl"
    if not log_path.exists():
        print(f"No experiments found at {log_path}", file=sys.stderr)
        return []

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print("No experiment records found.", file=sys.stderr)
        return []

    # Apply filter
    if filter_fn is not None:
        records = [r for r in records if filter_fn(r)]

    # Sort by metrics[sort_by] ascending (lower = better)
    records.sort(
        key=lambda r: r.get("metrics", {}).get(sort_by, float("inf"))
    )

    # Apply limit
    if limit is not None:
        records = records[:limit]

    # Print formatted table
    _print_table(records, sort_by)

    return records


def _print_table(records, sort_by):
    """Print a formatted table of experiment results.

    Args:
        records: List of experiment record dicts.
        sort_by: The metric used for sorting (highlighted in header).
    """
    if not records:
        print("No results to display.")
        return

    # Header
    header = f"{'#':>3}  {'run_id':<30}  {'val_bpb':>8}  {'artifact_bytes':>14}  {'wall_time':>10}  {'seed':>6}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    # Rows
    for i, record in enumerate(records, 1):
        metrics = record.get("metrics", {})
        config = record.get("config", {})
        run_id = record.get("run_id", "unknown")
        val_bpb = metrics.get("val_bpb", float("nan"))
        artifact_bytes = metrics.get("artifact_bytes", 0)
        wall_time = metrics.get("wall_time_seconds", 0.0)
        seed = config.get("seed", "?")

        print(
            f"{i:>3}  {run_id:<30}  {val_bpb:>8.4f}  {artifact_bytes:>14,}  {wall_time:>10.1f}  {seed:>6}"
        )

    print(separator)
    print(f"Total: {len(records)} experiments (sorted by {sort_by} ascending)")


def main():
    """CLI entry point for querying experiments."""
    parser = argparse.ArgumentParser(
        description="Query and display experiment results"
    )
    parser.add_argument("--result-dir", required=True,
                        help="Directory containing experiments.jsonl")
    parser.add_argument("--sort-by", default="val_bpb",
                        help="Metric to sort by (default: val_bpb)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of results to show")
    parser.add_argument("--min-bpb", type=float, default=None,
                        help="Filter: minimum val_bpb")
    parser.add_argument("--max-bpb", type=float, default=None,
                        help="Filter: maximum val_bpb")

    args = parser.parse_args()

    # Build filter function from CLI args
    filter_fn = None
    if args.min_bpb is not None or args.max_bpb is not None:
        def filter_fn(record):
            bpb = record.get("metrics", {}).get("val_bpb", float("inf"))
            if args.min_bpb is not None and bpb < args.min_bpb:
                return False
            if args.max_bpb is not None and bpb > args.max_bpb:
                return False
            return True

    query_experiments(
        result_dir=args.result_dir,
        sort_by=args.sort_by,
        limit=args.limit,
        filter_fn=filter_fn,
    )


if __name__ == "__main__":
    main()

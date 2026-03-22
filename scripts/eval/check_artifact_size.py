#!/usr/bin/env python3
"""Artifact size validation for parameter-golf competition.

Checks that the total submission artifact (compressed model + code files)
stays within the 16,000,000 byte (16MB) limit. Reports model bytes,
code bytes, total, budget remaining, and a PASS/FAIL/WARNING status.

Usage:
    python check_artifact_size.py --model repo/final_model.int8.ptz --code repo/train_gpt.py
    python check_artifact_size.py --model model.ptz --code train.py utils.py --limit 16000000
"""

import argparse
import os
import sys


def check_artifact_size(model_path, code_paths, limit=16_000_000, margin=500_000):
    """Validate total artifact size against the competition limit.

    Args:
        model_path: Path to the compressed model file (e.g., final_model.int8.ptz).
        code_paths: List of code file paths whose UTF-8 byte sizes are summed.
        limit: Maximum allowed total bytes (default: 16,000,000).
        margin: Warning threshold -- if budget_remaining < margin, status is WARNING.

    Returns:
        dict with keys: model_bytes, code_bytes, total, limit, budget_remaining, status.
        status is "PASS", "FAIL", or "WARNING".

    Side effects:
        Prints a formatted size report to stdout.
        Calls sys.exit(1) if status is "FAIL".
    """
    # Measure model file size
    model_bytes = os.path.getsize(model_path)

    # Measure code files -- read and encode to UTF-8, sum byte lengths
    code_bytes = 0
    for code_path in code_paths:
        with open(code_path, "r", encoding="utf-8") as f:
            content = f.read()
        code_bytes += len(content.encode("utf-8"))

    total = model_bytes + code_bytes
    budget_remaining = limit - total

    # Determine status
    if total > limit:
        status = "FAIL"
    elif budget_remaining < margin:
        status = "WARNING"
    else:
        status = "PASS"

    # Print formatted report
    print("=== Artifact Size Report ===")
    print(f"Model:  {model_bytes:>12,} bytes ({model_bytes / 1e6:.2f} MB)")
    print(f"Code:   {code_bytes:>12,} bytes ({code_bytes / 1e3:.1f} KB)")
    print(f"Total:  {total:>12,} bytes ({total / 1e6:.2f} MB)")
    print(f"Limit:  {limit:>12,} bytes ({limit / 1e6:.2f} MB)")
    print(f"Budget: {budget_remaining:>12,} bytes ({budget_remaining / 1e6:.2f} MB)")
    print(f"Status: {status}")

    result = {
        "model_bytes": model_bytes,
        "code_bytes": code_bytes,
        "total": total,
        "limit": limit,
        "budget_remaining": budget_remaining,
        "status": status,
    }

    if status == "FAIL":
        print(f"\nFAIL: Artifact exceeds limit by {total - limit:,} bytes")
        sys.exit(1)
    elif status == "WARNING":
        print(f"\nWARNING: Only {budget_remaining:,} bytes remaining (< {margin:,} margin)")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check artifact size against competition limit"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to compressed model file (e.g., final_model.int8.ptz)"
    )
    parser.add_argument(
        "--code", required=True, nargs="+",
        help="Path(s) to code file(s) (e.g., train_gpt.py)"
    )
    parser.add_argument(
        "--limit", type=int, default=16_000_000,
        help="Maximum allowed total bytes (default: 16000000)"
    )
    parser.add_argument(
        "--margin", type=int, default=500_000,
        help="Warning threshold for budget remaining (default: 500000)"
    )
    args = parser.parse_args()

    check_artifact_size(args.model, args.code, limit=args.limit, margin=args.margin)


if __name__ == "__main__":
    main()

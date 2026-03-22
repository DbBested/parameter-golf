#!/usr/bin/env python3
"""BPB validation for parameter-golf competition.

Compares a measured bits-per-byte (BPB) value against the expected baseline
value of 1.2244, reporting PASS/FAIL based on a configurable tolerance.

Usage:
    python validate_bpb.py --measured 1.2240
    python validate_bpb.py --measured 1.3000 --expected 1.2244 --tolerance 0.005
"""

import argparse
import sys


def validate_baseline_bpb(measured_bpb, expected=1.2244, tolerance=0.005):
    """Validate that measured BPB is within tolerance of expected value.

    Args:
        measured_bpb: The BPB value from training evaluation.
        expected: The expected baseline BPB (default: 1.2244).
        tolerance: Maximum acceptable absolute difference (default: 0.005).

    Returns:
        True if abs(measured_bpb - expected) <= tolerance, False otherwise.

    Side effects:
        Prints a formatted validation report to stdout.
        If diff > 0.01, prints a diagnostic note about potential issues.
    """
    diff = abs(measured_bpb - expected)
    status = "PASS" if diff <= tolerance else "FAIL"

    print("=== BPB Validation ===")
    print(f"Measured:  {measured_bpb:.4f}")
    print(f"Expected:  {expected:.4f}")
    print(f"Diff:      {diff:.4f}")
    print(f"Tolerance: {tolerance:.4f}")
    print(f"Status:    {status}")

    if diff > 0.01:
        if measured_bpb > expected:
            print(
                "\nNOTE: BPB significantly higher than expected. "
                "Check training convergence, dataset integrity, and "
                "environment differences (CUDA version, cuDNN algorithm)."
            )
        else:
            print(
                "\nNOTE: BPB significantly lower than expected. "
                "Check for evaluation bugs or data leakage."
            )

    return status == "PASS"


def main():
    parser = argparse.ArgumentParser(
        description="Validate measured BPB against expected baseline"
    )
    parser.add_argument(
        "--measured", type=float, required=True,
        help="Measured BPB value from training"
    )
    parser.add_argument(
        "--expected", type=float, default=1.2244,
        help="Expected baseline BPB (default: 1.2244)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.005,
        help="Maximum acceptable difference (default: 0.005)"
    )
    args = parser.parse_args()

    passed = validate_baseline_bpb(args.measured, args.expected, args.tolerance)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

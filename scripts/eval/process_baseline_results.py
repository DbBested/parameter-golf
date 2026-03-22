#!/usr/bin/env python3
"""Process baseline training results from SLURM log output.

Extracts BPB and artifact size from the training log, runs validation
checks, and logs the experiment to experiments.jsonl.

Usage:
    # After training completes, find the SLURM log and run:
    python scripts/eval/process_baseline_results.py \
        --log experiments/logs/<JOBID>-pgolf-baseline.out \
        --result-dir experiments/results \
        --run-id baseline_seed1337

    # Or specify BPB and artifact info manually:
    python scripts/eval/process_baseline_results.py \
        --val-bpb 1.2244 \
        --model-file repo/final_model.int8.ptz \
        --result-dir experiments/results \
        --run-id baseline_seed1337
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def extract_from_log(log_path):
    """Extract BPB, artifact size, and wall time from a training log.

    Parses the SLURM output log for the final_int8_zlib_roundtrip line
    and serialization size lines produced by train_gpt.py.

    Args:
        log_path: Path to SLURM .out log file.

    Returns:
        dict with val_bpb, val_loss, model_bytes, code_bytes, wall_time_ms.
        Values are None if not found.
    """
    result = {
        "val_bpb": None,
        "val_loss": None,
        "model_bytes": None,
        "code_bytes": None,
        "wall_time_ms": None,
        "seed": None,
    }

    with open(log_path, "r") as f:
        log_text = f.read()

    # Extract final roundtrip BPB (the definitive metric)
    # Format: final_int8_zlib_roundtrip val_loss:0.8484 val_bpb:1.2244
    roundtrip_match = re.search(
        r"final_int8_zlib_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)",
        log_text
    )
    if roundtrip_match:
        result["val_loss"] = float(roundtrip_match.group(1))
        result["val_bpb"] = float(roundtrip_match.group(2))

    # Extract exact values if available
    # Format: final_int8_zlib_roundtrip_exact val_loss:0.84840000 val_bpb:1.22440000
    exact_match = re.search(
        r"final_int8_zlib_roundtrip_exact val_loss:([\d.]+) val_bpb:([\d.]+)",
        log_text
    )
    if exact_match:
        result["val_loss"] = float(exact_match.group(1))
        result["val_bpb"] = float(exact_match.group(2))

    # Extract int8+zlib model size
    # Format: Serialized model int8+zlib: 15234567 bytes
    size_match = re.search(
        r"Serialized model int8\+zlib: (\d+) bytes",
        log_text
    )
    if size_match:
        result["model_bytes"] = int(size_match.group(1))

    # Extract code size
    # Format: Code size: 18432 bytes
    code_match = re.search(r"Code size: (\d+) bytes", log_text)
    if code_match:
        result["code_bytes"] = int(code_match.group(1))

    # Extract total submission size
    # Format: Total submission size int8+zlib: 15253000 bytes
    total_match = re.search(
        r"Total submission size int8\+zlib: (\d+) bytes",
        log_text
    )
    if total_match:
        result["total_bytes"] = int(total_match.group(1))

    # Extract wall time from last training step
    # Format: step:20000/20000 ... train_time:480000ms
    time_match = re.findall(r"train_time:(\d+)ms", log_text)
    if time_match:
        result["wall_time_ms"] = int(time_match[-1])

    # Extract seed
    seed_match = re.search(r"seed:(\d+)", log_text)
    if seed_match:
        result["seed"] = int(seed_match.group(1))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Process baseline training results"
    )
    parser.add_argument(
        "--log", default=None,
        help="Path to SLURM output log file"
    )
    parser.add_argument(
        "--val-bpb", type=float, default=None,
        help="Manually specify val BPB (overrides log extraction)"
    )
    parser.add_argument(
        "--model-file", default="repo/final_model.int8.ptz",
        help="Path to model file for size measurement"
    )
    parser.add_argument(
        "--code-file", default="repo/train_gpt.py",
        help="Path to code file for size measurement"
    )
    parser.add_argument(
        "--result-dir", default="experiments/results",
        help="Directory for experiments.jsonl"
    )
    parser.add_argument(
        "--run-id", default="baseline_seed1337",
        help="Experiment run identifier"
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Random seed used for training"
    )
    parser.add_argument(
        "--wall-time", type=float, default=None,
        help="Training wall time in seconds (overrides log extraction)"
    )
    args = parser.parse_args()

    # Extract from log or use manual values
    if args.log:
        print(f"Extracting results from: {args.log}")
        extracted = extract_from_log(args.log)
        print(f"Extracted: {json.dumps(extracted, indent=2)}")
    else:
        extracted = {}

    val_bpb = args.val_bpb or extracted.get("val_bpb")
    if val_bpb is None:
        print("ERROR: No val_bpb found. Provide --log or --val-bpb.")
        sys.exit(1)

    seed = extracted.get("seed") or args.seed

    # Measure artifact size from files
    model_bytes = 0
    code_bytes = 0
    if os.path.exists(args.model_file):
        model_bytes = os.path.getsize(args.model_file)
    elif extracted.get("model_bytes"):
        model_bytes = extracted["model_bytes"]
    else:
        print(f"WARNING: Model file not found: {args.model_file}")

    if os.path.exists(args.code_file):
        with open(args.code_file, "r", encoding="utf-8") as f:
            code_bytes = len(f.read().encode("utf-8"))
    elif extracted.get("code_bytes"):
        code_bytes = extracted["code_bytes"]

    total_bytes = model_bytes + code_bytes

    # Wall time
    wall_time_seconds = args.wall_time
    if wall_time_seconds is None and extracted.get("wall_time_ms"):
        wall_time_seconds = extracted["wall_time_ms"] / 1000.0
    if wall_time_seconds is None:
        wall_time_seconds = 0.0
        print("WARNING: No wall time found; set to 0.")

    # Run BPB validation
    print("\n--- BPB Validation ---")
    from validate_bpb import validate_baseline_bpb
    bpb_passed = validate_baseline_bpb(val_bpb)

    # Run artifact size check (print only, no exit)
    print("\n--- Artifact Size ---")
    print(f"Model:  {model_bytes:>12,} bytes ({model_bytes / 1e6:.2f} MB)")
    print(f"Code:   {code_bytes:>12,} bytes ({code_bytes / 1e3:.1f} KB)")
    print(f"Total:  {total_bytes:>12,} bytes ({total_bytes / 1e6:.2f} MB)")
    print(f"Limit:  {16_000_000:>12,} bytes (16.00 MB)")
    size_passed = total_bytes <= 16_000_000
    print(f"Status: {'PASS' if size_passed else 'FAIL'}")

    # Log experiment
    print("\n--- Logging experiment ---")
    # Add scripts dir to path for import
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tracking"))
    from log_experiment import log_experiment

    config = {
        "seed": seed,
        "n_layers": 9,
        "n_dim": 512,
        "vocab_size": 1024,
        "optimizer": "muon+adam",
        "quantization": "int8",
        "compression": "zlib_9",
        "techniques_enabled": [],
    }

    metrics = {
        "val_bpb": val_bpb,
        "artifact_bytes": total_bytes,
        "code_bytes": code_bytes,
        "model_bytes": model_bytes,
        "wall_time_seconds": wall_time_seconds,
        "gpu_type": "H200",
        "n_gpus": 8,
    }

    record = log_experiment(args.result_dir, args.run_id, config, metrics)
    print(f"Logged: {json.dumps(record, indent=2)}")

    # Summary
    print("\n" + "=" * 50)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 50)
    print(f"BPB:           {val_bpb:.4f} ({'PASS' if bpb_passed else 'FAIL'})")
    print(f"Artifact:      {total_bytes:,} bytes ({'PASS' if size_passed else 'FAIL'})")
    print(f"Wall time:     {wall_time_seconds:.1f}s")
    print(f"Run ID:        {args.run_id}")
    print(f"Logged to:     {args.result_dir}/experiments.jsonl")

    if not bpb_passed or not size_passed:
        print("\nWARNING: One or more checks FAILED")
        sys.exit(1)
    else:
        print("\nAll checks PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Experiment logging utility for parameter-golf competition.

Appends JSON-lines records to experiments.jsonl with run configuration
and metrics. Each record captures timestamp, run ID, SLURM job context,
hyperparameters, and evaluation results.

Usage:
    python log_experiment.py --result-dir experiments/results \\
        --run-id baseline_seed1337 --val-bpb 1.2244 \\
        --artifact-bytes 15000000 --wall-time 480
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log_experiment(result_dir, run_id, config, metrics):
    """Append experiment result to JSON-lines file.

    Args:
        result_dir: Directory containing experiments.jsonl.
        run_id: Unique identifier for this run.
        config: Dict of hyperparameters. Must include at minimum:
            seed, n_layers, n_dim, vocab_size, optimizer, quantization,
            compression, techniques_enabled (list).
        metrics: Dict of evaluation results. Must include at minimum:
            val_bpb (float), artifact_bytes (int), code_bytes (int),
            model_bytes (int), wall_time_seconds (float), gpu_type (str),
            n_gpus (int).

    Returns:
        The record dict that was written.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "config": config,
        "metrics": metrics,
    }
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    log_path = result_path / "experiments.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_experiments(result_dir):
    """Read all experiment records from experiments.jsonl.

    Args:
        result_dir: Directory containing experiments.jsonl.

    Returns:
        List of parsed JSON dicts. Empty list if file doesn't exist.
    """
    log_path = Path(result_dir) / "experiments.jsonl"
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed line: {line[:80]}...",
                          file=sys.stderr)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log an experiment result to experiments.jsonl"
    )
    parser.add_argument("--result-dir", required=True,
                        help="Directory for experiments.jsonl")
    parser.add_argument("--run-id", required=True,
                        help="Unique run identifier")
    parser.add_argument("--val-bpb", type=float, required=True,
                        help="Validation bits-per-byte")
    parser.add_argument("--artifact-bytes", type=int, required=True,
                        help="Total artifact size in bytes")
    parser.add_argument("--wall-time", type=float, required=True,
                        help="Training wall time in seconds")
    parser.add_argument("--code-bytes", type=int, default=0,
                        help="Code size in bytes")
    parser.add_argument("--model-bytes", type=int, default=0,
                        help="Model size in bytes")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed used")
    parser.add_argument("--n-layers", type=int, default=9,
                        help="Number of transformer layers")
    parser.add_argument("--n-dim", type=int, default=512,
                        help="Model hidden dimension")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="Vocabulary size")
    parser.add_argument("--optimizer", default="muon+adam",
                        help="Optimizer name")
    parser.add_argument("--quantization", default="int8",
                        help="Quantization method")
    parser.add_argument("--compression", default="zlib_9",
                        help="Compression method")
    parser.add_argument("--gpu-type", default="H200",
                        help="GPU type used")
    parser.add_argument("--n-gpus", type=int, default=8,
                        help="Number of GPUs used")
    parser.add_argument("--techniques", nargs="*", default=[],
                        help="List of techniques enabled")

    args = parser.parse_args()

    config = {
        "seed": args.seed,
        "n_layers": args.n_layers,
        "n_dim": args.n_dim,
        "vocab_size": args.vocab_size,
        "optimizer": args.optimizer,
        "quantization": args.quantization,
        "compression": args.compression,
        "techniques_enabled": args.techniques,
    }

    model_bytes = args.model_bytes if args.model_bytes else (
        args.artifact_bytes - args.code_bytes
    )
    metrics = {
        "val_bpb": args.val_bpb,
        "artifact_bytes": args.artifact_bytes,
        "code_bytes": args.code_bytes,
        "model_bytes": model_bytes,
        "wall_time_seconds": args.wall_time,
        "gpu_type": args.gpu_type,
        "n_gpus": args.n_gpus,
    }

    record = log_experiment(args.result_dir, args.run_id, config, metrics)
    print(json.dumps(record, indent=2))

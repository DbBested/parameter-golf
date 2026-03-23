"""Precompute zlib compression ratio per training shard as difficulty metric."""
import json
import glob
import zlib
import numpy as np
from pathlib import Path

DATA_PATH = "/orcd/home/002/tomli/parameter_golf/repo/data/datasets/fineweb10B_sp1024"
PATTERN = f"{DATA_PATH}/fineweb_train_*.bin"
SAMPLE_SIZE = 100_000  # tokens to sample per shard

def load_data_shard(file: Path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens_np

def score_shard(path: Path) -> float:
    tokens = load_data_shard(path)
    sample = tokens[:min(SAMPLE_SIZE, len(tokens))].tobytes()
    compressed = zlib.compress(sample, 1)
    return len(compressed) / len(sample)

if __name__ == "__main__":
    files = sorted(glob.glob(PATTERN))
    print(f"Scoring {len(files)} shards...")
    scores = {}
    for f in files:
        p = Path(f)
        ratio = score_shard(p)
        scores[p.name] = ratio
        print(f"  {p.name}: {ratio:.4f}")
    out = Path(__file__).parent / "difficulty.json"
    with open(out, "w") as fp:
        json.dump(scores, fp, indent=2)
    print(f"Saved to {out}")
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print(f"Easiest: {ranked[0][0]} ({ranked[0][1]:.4f})")
    print(f"Hardest: {ranked[-1][0]} ({ranked[-1][1]:.4f})")

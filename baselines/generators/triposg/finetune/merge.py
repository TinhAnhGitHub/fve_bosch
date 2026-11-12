"""Merge sharded predict.py outputs into one npz plus the challenge zip (geometry/{id}.ply, 8192 points each)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shards", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect", type=int, help="Required number of objects, e.g. 511 for the public test")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; use another path")
    parts = [dict(np.load(s, allow_pickle=False)) for s in args.shards]
    fixed = ("seed", "steps", "dense_depth", "hierarchical_depth", "cache_hash", "shape_checkpoint")
    for key in fixed:
        if len({str(part[key]) for part in parts}) != 1:
            raise ValueError(f"Shards disagree on {key}")
    merged = {k: np.concatenate([part[k] for part in parts]) for k in ("ids", "pts", "unit", "radius_mm")}
    order = np.argsort(merged["ids"])
    merged = {k: v[order] for k, v in merged.items()}
    ids = [str(i) for i in merged["ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs across shards")
    if args.expect is not None and len(ids) != args.expect:
        raise ValueError(f"{len(ids)} objects, expected {args.expect}")
    if merged["pts"].shape[1:] != (8192, 3) or not np.isfinite(merged["pts"]).all():
        raise ValueError("Points must be finite with shape (n, 8192, 3)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **merged, **{k: parts[0][k] for k in fixed})
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from geom import pack
    pack(args.output.with_suffix(".zip"), dict(zip(ids, merged["pts"])))
    print(f"merged {len(ids)} objects from {len(parts)} shards -> {args.output.with_suffix('.zip')}")


if __name__ == "__main__":
    main()

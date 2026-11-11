"""Rotation/translation-only diagnostic: metric clouds and oracle-size clouds.

This is a LOCAL diagnostic, not a verified replacement for the official grader.
Use identical alignment, target samples, seeds and IDs for every candidate.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from common import metric_points, read_json


def pca(points):
    p = points - points.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    rotation = vt.T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return p @ rotation


def aligned_cd(pred, truth, iterations=20):
    """24 proper rotation starts + rigid ICP. Never reflect or fit scale."""
    a, b = pca(pred), pca(truth)
    tree = cKDTree(b)
    best = float("inf")
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rot = np.diag(signs)[:, list(perm)]
            if np.linalg.det(rot) < 0:
                continue
            q = a @ rot
            for _ in range(iterations):
                _, index = tree.query(q)
                match = b[index]
                ca, cb = q.mean(0), match.mean(0)
                u, _, vt = np.linalg.svd((q-ca).T @ (match-cb))
                correction = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
                q = (q-ca) @ (u @ correction @ vt) + cb
            score = np.square(tree.query(q)[0]).mean() + np.square(cKDTree(q).query(b)[0]).mean()
            best = min(best, float(score))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    meta = read_json(args.cache / "metadata.json")
    rows = {r["id"]: r for r in meta["rows"] if r["split"] == "val"}
    with np.load(args.predictions, allow_pickle=False) as z:
        ids, pts = z["ids"].tolist(), z["pts"].copy()
    if len(set(ids)) != len(ids) or set(ids) != set(rows) or pts.shape != (len(ids), 8192, 3):
        raise ValueError("Score ALL validation IDs exactly once, with 8192 points each")
    metric, oracle, radius_errors = [], [], []
    for sid, pred in zip(ids, pts):
        if not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction {sid}")
        mesh = trimesh.load(args.data / rows[sid]["mesh"], force="mesh", process=False)
        truth, _ = trimesh.sample.sample_surface(mesh, 8192, seed=42)
        truth *= meta["mesh_unit_mm"]
        truth -= truth.mean(0)
        radius = np.sqrt(np.square(truth).sum(1).mean())
        own = np.sqrt(np.square(pred-pred.mean(0)).sum(1).mean())
        metric.append(aligned_cd(pred, truth))
        oracle.append(aligned_cd(metric_points(pred, radius), truth))
        radius_errors.append((own-radius)**2)
        print(sid, "metric", metric[-1], "oracle-size", oracle[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=np.array(ids), metric=np.array(metric), oracle=np.array(oracle), radius_mse=np.array(radius_errors))
    print({"n": len(ids), "metric_mean": float(np.mean(metric)), "metric_median": float(np.median(metric)),
           "oracle_size_mean": float(np.mean(oracle)), "radius_mse_mm2": float(np.mean(radius_errors))})


if __name__ == "__main__":
    main()

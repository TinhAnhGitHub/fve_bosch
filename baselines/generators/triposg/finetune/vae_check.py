"""Measure the VAE reconstruction ceiling before any training.

Decodes cached posterior means of a few validation objects through the frozen VAE and scores them
against the true meshes at the TRUE size. No transformer, no generation, no training. If this error
is already close to the Atlas model's, shape adaptation cannot help and the run should stop.

Run on the CUDA machine after `prepare.py encode` finished:

    python vae_check.py --upstream /content/TripoSG --cache CACHE --data DATA --count 10 --output out.npz

Reports two numbers per object, squared Chamfer in mm^2 (sum of both directions, like `evaluate.py`):
`plain` compares in the shared canonical frame, `aligned` adds the 24 rotation starts plus ICP.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from common import normalized_surface, read_json, upstream

ENCODE_SEED: int = 42  # prepare.py encode default; reusing it regenerates the exact surface the VAE saw


def plain_cd(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Chamfer distance (both directions summed) between two clouds in the same frame."""
    return float(np.square(cKDTree(b).query(a)[0]).mean() + np.square(cKDTree(a).query(b)[0]).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--dense-depth", type=int, default=8)
    p.add_argument("--hierarchical-depth", type=int, default=9)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; use another path")
    if not torch.cuda.is_available():
        raise RuntimeError("Run on the CUDA machine")

    upstream(args.upstream)
    from triposg.inference_utils import hierarchical_extract_geometry
    from triposg.models.autoencoders import TripoSGVAEModel

    import evaluate  # same directory; reuses aligned_cd so the numbers match evaluate.py

    meta = read_json(args.cache / "metadata.json")
    val = [r for r in meta["rows"] if r["split"] == "val"]
    picked = val[::max(1, len(val) // args.count)][:args.count]
    vae = TripoSGVAEModel.from_pretrained(meta["model"], subfolder="vae", revision=meta["revision"],
                                         torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    bounds = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)  # pipeline default

    ids: list[str] = []
    plain: list[float] = []
    aligned: list[float] = []
    radius_ratio: list[float] = []
    for row in picked:
        with np.load(args.cache / f"{row['id']}.npz") as z:
            mean = torch.from_numpy(z["mean"].astype(np.float32))[None].cuda()
        mesh = trimesh.load(args.data / row["mesh"], force="mesh", process=False)
        surface, radius_mm = normalized_surface(mesh, meta["surface_points"], ENCODE_SEED,
                                                meta["mesh_unit_mm"], meta["pose"])
        truth_unit = surface[:8192, :3].astype(np.float64)  # random surface draw, so a prefix is unbiased
        scale = radius_mm / float(np.sqrt(np.square(truth_unit - truth_unit.mean(0)).sum(1).mean()))

        with torch.no_grad():
            verts, faces = hierarchical_extract_geometry(
                lambda x: vae.decode(mean, sampled_points=x).sample, "cuda", bounds=bounds,
                dense_octree_depth=args.dense_depth, hierarchical_octree_depth=args.hierarchical_depth)[0]
        recon = trimesh.Trimesh(np.asarray(verts, np.float32), np.ascontiguousarray(faces), process=False)
        recon_unit = np.asarray(trimesh.sample.sample_surface(recon, 8192, seed=ENCODE_SEED)[0], np.float64)

        truth_mm, recon_mm = truth_unit * scale, recon_unit * scale  # recon deliberately NOT re-normalised
        ids.append(row["id"])
        plain.append(plain_cd(recon_mm, truth_mm))
        aligned.append(evaluate.aligned_cd(recon_mm - recon_mm.mean(0), truth_mm - truth_mm.mean(0)))
        radius_ratio.append(float(np.sqrt(np.square(recon_mm - recon_mm.mean(0)).sum(1).mean()) / radius_mm))
        print(row["id"], f"plain={plain[-1]:.2f} aligned={aligned[-1]:.2f} radius_ratio={radius_ratio[-1]:.3f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=np.array(ids), plain=np.array(plain), aligned=np.array(aligned),
             radius_ratio=np.array(radius_ratio))
    print({"n": len(ids), "plain_mean": float(np.mean(plain)), "plain_median": float(np.median(plain)),
           "aligned_mean": float(np.mean(aligned)), "aligned_median": float(np.median(aligned)),
           "radius_ratio_mean": float(np.mean(radius_ratio))})
    print("Reference (all 204 holdout, different objects): Atlas 38.9, TripoSG at true size 29.6.")


if __name__ == "__main__":
    main()

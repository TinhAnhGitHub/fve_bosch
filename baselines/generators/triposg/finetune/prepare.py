"""Build a manifest locally; encode image and mesh caches on an authorized GPU."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from common import (MODEL_REVISION, SCHEMA, UPSTREAM_COMMIT, digest,
                    normalized_surface, prepared_image, read_json, upstream, write_json)


def manifest(args):
    root = args.data.resolve()
    images = sorted((root / "train").glob("*/*/*/rgb_industrial.png"))
    if not images:
        raise ValueError(f"No training images under {root}")
    groups = read_json(args.groups) if args.groups else {}
    rows, seen = [], set()
    for image in images:
        mesh = image.with_name("mesh.ply")
        if not mesh.exists():
            raise ValueError(f"Missing mesh: {mesh}")
        sid = image.parent.name
        if sid in seen:
            raise ValueError(f"Duplicate sample id: {sid}")
        seen.add(sid)
        mh = digest(mesh)
        rows.append(dict(id=sid, image=str(image.relative_to(root)), mesh=str(mesh.relative_to(root)),
                         mesh_sha256=mh, image_sha256=digest(image),
                         group=groups.get(sid, mh), category=image.parent.parent.name))
    # Union user groups and exact-duplicate meshes: neither may cross splits.
    parent = {r["group"]: r["group"] for r in rows}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    hashes = {}
    for row in rows:
        g = row["group"]
        if row["mesh_sha256"] in hashes:
            parent[find(g)] = find(hashes[row["mesh_sha256"]])
        hashes[row["mesh_sha256"]] = g
    for row in rows:
        row["group"] = find(row["group"])
    if getattr(args, "val_ids", None):
        # Fixed comparison holdout (one id per line). Training rows that share a group or an exact
        # mesh with a holdout row are excluded, so nothing near-identical can leak into training.
        wanted = {line.strip() for line in args.val_ids.read_text().splitlines() if line.strip()}
        missing = wanted - seen
        if missing:
            raise ValueError(f"--val-ids not in the data: {sorted(missing)[:5]} ({len(missing)} total)")
        validation = {r["group"] for r in rows if r["id"] in wanted}
        for row in rows:
            if row["id"] in wanted:
                row["split"] = "val"
            else:
                row["split"] = "excluded" if row["group"] in validation else "train"
    else:
        unique = sorted({r["group"] for r in rows})
        np.random.default_rng(args.seed).shuffle(unique)
        validation = set(unique[:max(1, round(len(unique) * args.val_fraction))])
        for row in rows:
            row["split"] = "val" if row["group"] in validation else "train"
    if not any(r["split"] == "train" for r in rows):
        raise ValueError("Need at least two distinct object groups")
    write_json(args.output, dict(schema=SCHEMA, seed=args.seed, rows=rows))
    print(Counter(r["split"] for r in rows))
    print("Manifest only. Nothing uploaded, encoded, or trained.")


def encode(args):
    import trimesh
    from transformers import AutoImageProcessor, Dinov2Model

    upstream(args.upstream)
    from triposg.models.autoencoders import TripoSGVAEModel
    import triposg.models.autoencoders.autoencoder_kl_triposg as vae_module
    from torch_cluster import fps

    # Upstream comments this import out in the inference release.
    vae_module.fps = fps
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Encode on an authorized CUDA machine")
    doc = read_json(args.manifest)
    if doc["schema"] != SCHEMA:
        raise ValueError("Unsupported manifest schema")
    encoded = [r for r in doc["rows"] if r["split"] != "excluded"]  # excluded rows are never trained or scored
    root, out = args.data.resolve(), args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory; never mix cache versions")
    out.mkdir(parents=True, exist_ok=True)
    image_model = Dinov2Model.from_pretrained(args.model, subfolder="image_encoder_dinov2",
                                             revision=MODEL_REVISION).float().cuda().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(args.model, subfolder="feature_extractor_dinov2",
                                                   revision=MODEL_REVISION)
    # Stage 1: image features. No VAE on GPU yet.
    with torch.no_grad():
        for row in encoded:
            path = root / row["image"]
            if digest(path) != row["image_sha256"]:
                raise ValueError(f"Image changed since manifest: {path}")
            pixels = processor(images=prepared_image(path), return_tensors="pt").pixel_values.cuda()
            feat = image_model(pixels).last_hidden_state[0].float().cpu().numpy()
            np.savez(out / f"{row['id']}.npz", features=feat.astype(np.float16))
    del image_model
    torch.cuda.empty_cache()
    vae = TripoSGVAEModel.from_pretrained(args.model, subfolder="vae", revision=MODEL_REVISION,
                                         torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
    # Stage 2: VAE input is (B,N,6), XYZ + surface normal.
    with torch.no_grad():
        for row in encoded:
            path = root / row["mesh"]
            if digest(path) != row["mesh_sha256"]:
                raise ValueError(f"Mesh changed since manifest: {path}")
            mesh = trimesh.load(path, force="mesh", process=False)
            if len(mesh.faces) == 0:
                raise ValueError(f"Not a triangle mesh: {path}")
            surface, radius = normalized_surface(mesh, args.surface_points, args.seed, args.mesh_unit_mm, args.pose)
            dist = vae.encode(torch.from_numpy(surface)[None].cuda(), num_tokens=args.tokens,
                              seed=args.seed).latent_dist
            mean, logvar = dist.mean[0].float().cpu().numpy(), dist.logvar[0].float().cpu().numpy()
            cache = out / f"{row['id']}.npz"
            with np.load(cache) as z:
                features = z["features"].copy()
            np.savez(cache, features=features, mean=mean, logvar=logvar, radius_mm=np.float32(radius))
            print(row["id"], mean.shape, f"radius={radius:.3f}mm", flush=True)
    # Completion marker written only after both stages finish.
    write_json(out / "metadata.json", dict(schema=SCHEMA, model=args.model,
               revision=MODEL_REVISION, upstream=UPSTREAM_COMMIT,
               manifest_sha256=digest(args.manifest), tokens=args.tokens,
               surface_points=args.surface_points, mesh_unit_mm=args.mesh_unit_mm, pose=args.pose,
               vae_normalization="surface-center,max-abs-to-0.95",
               image_preprocess="rgba-crop-or-rgb-full,white-square-pad-10percent,v1", rows=doc["rows"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--data", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--groups", type=Path)
    m.add_argument("--val-ids", type=Path, help="Fixed holdout ids, one per line; overrides --val-fraction")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--val-fraction", type=float, default=0.1)
    e = sub.add_parser("encode")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--upstream", type=Path, required=True)
    e.add_argument("--model", default="VAST-AI/TripoSG")
    e.add_argument("--surface-points", type=int, default=32768)
    e.add_argument("--tokens", type=int, default=2048)
    e.add_argument("--mesh-unit-mm", type=float, required=True)
    e.add_argument("--pose", choices=["mesh", "pca"], default="pca")
    e.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.cmd == "manifest":
        if not 0 < args.val_fraction < 1:
            p.error("val-fraction must be between 0 and 1")
        manifest(args)
    else:
        if args.tokens < 1 or args.surface_points < 4 * args.tokens or args.mesh_unit_mm <= 0:
            p.error("Need positive units/tokens and surface-points >= 4*tokens")
        encode(args)


if __name__ == "__main__":
    main()

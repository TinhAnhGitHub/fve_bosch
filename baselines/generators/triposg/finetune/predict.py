"""Generate metric food clouds with frozen, LoRA, or full-tuned TripoSG."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download

from common import (SCHEMA, SizeHead, configure_transformer, metric_points,
                    prepared_image, restore_model, upstream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--size", type=Path, required=True)
    p.add_argument("--shape", type=Path, help="Omit to use the unchanged pretrained generator")
    p.add_argument("--split", choices=["train", "val", "eval"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    # Mesh-extraction grid. Upstream default 8/9 (512^3) takes ~50 s/object on an A100;
    # 7/8 (256^3) was used for the zero-shot TripoSG holdout run and is ample for 8192 surface points.
    p.add_argument("--dense-depth", type=int, default=8)
    p.add_argument("--hierarchical-depth", type=int, default=9)
    # One process per GPU: shard K of N takes rows K, K+N, ...; sharded runs write only the npz (merge.py packs the zip).
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--allow-size-cache-mismatch", action="store_true",
                   help="With --split eval only: pair a shape adapter and a size head from different caches "
                        "(the public photos are in neither cache, so nothing can leak)")
    args = p.parse_args()
    if args.allow_size_cache_mismatch and args.split != "eval":
        p.error("--allow-size-cache-mismatch is only valid with --split eval")
    if not 0 <= args.shard < args.num_shards:
        p.error("need 0 <= shard < num-shards")
    if args.steps <= 0:
        p.error("steps must be positive")
    if args.output.exists():
        raise ValueError("Output already exists; use another path")
    upstream(args.upstream)
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline

    size = torch.load(args.size, map_location="cpu", weights_only=False)
    if size["schema"] != SCHEMA or size["phase"] != "size":
        raise ValueError("Expected size-phase checkpoint")
    meta = size["metadata"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
    # A repo id fails in diffusers' download check (custom scheduler in model_index.json); a local snapshot loads like upstream's script.
    weights = snapshot_download(meta["model"], revision=meta["revision"])
    pipe = TripoSGPipeline.from_pretrained(weights, torch_dtype=dtype).to("cuda")
    shape_cache_hash = ""
    if args.shape:
        shape = torch.load(args.shape, map_location="cpu", weights_only=False)
        if shape["schema"] != SCHEMA or shape["phase"] != "shape":
            raise ValueError("Expected shape-phase checkpoint")
        if shape["cache_hash"] != size["cache_hash"] and not args.allow_size_cache_mismatch:
            raise ValueError("Shape and size artifacts must use the same cache/split")
        shape_cache_hash = shape["cache_hash"]
        configure_transformer(pipe.transformer, shape["config"])
        restore_model(pipe.transformer, shape["state"], shape["config"]["mode"])
        del shape
    pipe.transformer.eval().requires_grad_(False)
    head = SizeHead(hidden=size["config"]["hidden"]).cuda().eval()
    head.load_state_dict(size["state"])
    if args.split == "eval":
        images = sorted((args.data / "eval").glob("*/*/*/rgb_industrial.png"))
        rows = [dict(id=image.parent.name, image=str(image.relative_to(args.data))) for image in images]
    else:
        rows = [r for r in meta["rows"] if r["split"] == args.split]
    rows = rows[args.shard::args.num_shards]
    if not rows:
        raise ValueError("No input objects")
    ids, clouds, units, radii = [], [], [], []
    with torch.no_grad():
        for row in rows:
            image = prepared_image(args.data / row["image"])
            features, _ = pipe.encode_image(image, "cuda", 1)
            radius = float(head(features.float()).exp().item())
            verts, faces = pipe(image=image, num_tokens=meta["tokens"],
                    num_inference_steps=args.steps, guidance_scale=args.guidance,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    dense_octree_depth=args.dense_depth, hierarchical_octree_depth=args.hierarchical_depth,
                    use_flash_decoder=False).samples[0]
            mesh = trimesh.Trimesh(verts, faces, process=False)
            if len(mesh.faces) == 0:
                raise ValueError(f"Empty generated mesh: {row['id']}")
            points, _ = trimesh.sample.sample_surface(mesh, 8192, seed=args.seed)
            unit = metric_points(points, 1.0)
            ids.append(row["id"])
            units.append(unit)
            clouds.append(metric_points(points, radius))
            radii.append(radius)
            print(row["id"], f"radius={radius:.3f}mm", flush=True)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ids=np.array(ids), pts=np.stack(clouds), unit=np.stack(units),
             radius_mm=np.array(radii), seed=args.seed, steps=args.steps,
             dense_depth=args.dense_depth, hierarchical_depth=args.hierarchical_depth,
             cache_hash=size["cache_hash"], shape_cache_hash=shape_cache_hash if args.shape else "",
             shape_checkpoint=str(args.shape or "pretrained"))
    if args.num_shards > 1:
        return
    # Challenge archive uses existing repository point-count/path validation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from geom import pack
    pack(args.output.with_suffix(".zip"), dict(zip(ids, clouds)))


if __name__ == "__main__":
    main()

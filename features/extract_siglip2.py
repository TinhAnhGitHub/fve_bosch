# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "pillow", "numpy", "sentencepiece", "protobuf"]
# ///
"""Extract frozen SigLIP2 vision features for every train/val image (Apache-2.0 weights).

SigLIP has no CLS token; it summarises an image with an attention-pooling head. To keep the same
6-block layout as the DINO files, each of three layers gives:
  [pooled = head(post_layernorm(h_layer)),  mean of patch tokens of h_layer]
so blocks (0, 2, 4) are the "CLS-like" pooled vectors used by the B3.x pipeline.
Averaged over the image and its horizontal flip.

Run:  uv run features/extract_siglip2.py [--model google/siglip2-so400m-patch16-512 --size 512]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

PROJECT: Path = Path(__file__).resolve().parent.parent
DATA: Path = PROJECT / "data" / "fve"


class Images(Dataset):
    """Resize (no crop, backdrop kept) and SigLIP normalisation (mean 0.5, std 0.5)."""

    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255 - 0.5) / 0.5
        return torch.from_numpy(x).permute(2, 0, 1)


@torch.inference_mode()
def embed(vision: torch.nn.Module, x: torch.Tensor, layers: list[int]) -> torch.Tensor:
    out = vision(pixel_values=x, output_hidden_states=True)
    parts = []
    for layer in layers:
        h = out.hidden_states[layer]
        parts += [vision.head(vision.post_layernorm(h)), h.mean(dim=1)]
    return torch.cat(parts, dim=1).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-so400m-patch16-512")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--layers", type=int, nargs=3, default=[14, 20, 27])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="siglip2_so400m_512")
    args = ap.parse_args()

    keys, paths = [], []
    for split in ("train", "val"):
        for p in sorted((DATA / split).rglob("rgb_industrial.png")):
            keys.append(f"{split}/{p.parent.name}")
            paths.append(p)

    vision = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).vision_model.cuda().eval()
    n_layers = vision.config.num_hidden_layers
    assert max(args.layers) <= n_layers, f"model has {n_layers} layers"
    print(f"{args.model} vision: {sum(p.numel() for p in vision.parameters()) / 1e6:.0f}M params, "
          f"{n_layers} layers, size {args.size}, layers {args.layers}; {len(paths)} images")

    loader = DataLoader(Images(paths, args.size), batch_size=args.batch, num_workers=8, pin_memory=True)
    feats, t0 = [], time.time()
    for i, x in enumerate(loader):
        x = x.cuda(non_blocking=True).to(torch.bfloat16)
        feats.append(((embed(vision, x, args.layers) + embed(vision, torch.flip(x, dims=[3]), args.layers)) / 2)
                     .cpu().numpy())
        if i % 40 == 0:
            print(f"  batch {i}/{len(loader)}  {time.time() - t0:.0f}s  peak VRAM "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite features")
    out = PROJECT / "features" / f"{args.out}.npz"
    np.savez_compressed(out, X=X, keys=np.array(keys), model=args.model, size=args.size, layers=np.array(args.layers))
    print(f"saved {X.shape} -> {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

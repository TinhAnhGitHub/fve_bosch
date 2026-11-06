# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "pillow", "numpy"]
# ///
"""Extract frozen ViT features (DINOv2 / DINOv3) for every train/val image and cache them.

Per image: CLS token + mean of patch tokens (registers excluded) from three layers, averaged over
the image and its horizontal flip -> 6 blocks x hidden size. The last layer uses the final norm.

Run:
  uv run features/extract_dinov2.py                                   # DINOv2-L-reg @518 (B3 default)
  uv run features/extract_dinov2.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --size 512 \
      --layers 12 18 24 --out dinov3_vitl16_512

The Hugging Face token is read from HF_TOKEN, or from HUGGINGFACE_API_TOKEN in .env.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

PROJECT: Path = Path(__file__).resolve().parent.parent
DATA: Path = PROJECT / "data" / "fve"
ENV_FILE: Path = PROJECT / ".env"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_token() -> None:
    """Export HF_TOKEN from .env if it is not already set (value never printed)."""
    if os.environ.get("HF_TOKEN") or not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("HUGGINGFACE_API_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")


class Images(Dataset):
    """Loads, resizes (no crop, backdrop kept) and normalises one image."""

    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1)


@torch.inference_mode()
def embed(model: torch.nn.Module, x: torch.Tensor, layers: list[int], n_reg: int) -> torch.Tensor:
    """CLS + mean-patch from each requested layer -> (B, 2 * len(layers) * hidden)."""
    out = model(pixel_values=x, output_hidden_states=True)
    hidden = out.hidden_states
    parts = []
    for layer in layers:
        h = out.last_hidden_state if layer == len(hidden) - 1 else hidden[layer]
        parts += [h[:, 0], h[:, 1 + n_reg:].mean(dim=1)]
    return torch.cat(parts, dim=1).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov2-with-registers-large")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--layers", type=int, nargs=3, default=[12, 18, 24])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="dinov2l_reg_518")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                    help="bfloat16 for DINOv3-L (overflows to NaN in float16)")
    args = ap.parse_args()
    load_token()

    keys, paths = [], []
    for split in ("train", "val"):
        for p in sorted((DATA / split).rglob("rgb_industrial.png")):
            keys.append(f"{split}/{p.parent.name}")
            paths.append(p)

    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).cuda().eval()
    n_reg = getattr(model.config, "num_register_tokens", 0)
    n_layers = model.config.num_hidden_layers
    assert max(args.layers) <= n_layers, f"model has {n_layers} layers"
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {n_layers} layers, "
          f"{n_reg} registers, size {args.size}, layers {args.layers}; {len(paths)} images")

    loader = DataLoader(Images(paths, args.size), batch_size=args.batch, num_workers=8, pin_memory=True)
    feats, t0 = [], time.time()
    for i, x in enumerate(loader):
        x = x.cuda(non_blocking=True).to(dtype)
        f = (embed(model, x, args.layers, n_reg) + embed(model, torch.flip(x, dims=[3]), args.layers, n_reg)) / 2
        feats.append(f.cpu().numpy())
        if i % 40 == 0:
            print(f"  batch {i}/{len(loader)}  {time.time() - t0:.0f}s  peak VRAM "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    if not np.isfinite(X).all():
        raise SystemExit(f"non-finite features in {(~np.isfinite(X)).any(1).sum()} images; try --dtype bfloat16")
    out = PROJECT / "features" / f"{args.out}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, keys=np.array(keys), model=args.model, size=args.size, layers=np.array(args.layers))
    print(f"saved {X.shape} -> {out} in {time.time() - t0:.0f}s; finite={np.isfinite(X).all()}")


if __name__ == "__main__":
    main()

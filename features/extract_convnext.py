# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "timm", "transformers", "pillow", "numpy"]
# ///
"""Frozen ConvNeXt (timm, IN-22k pretrained) features for every train/val image, flip-averaged.

6-block layout of the B3.x harness (d = last-stage width); only blocks 0-2 are used (select (0, 1, 2)):
  b0 = stage-4 average pool, b1 = stage-4 max pool, b2 = [stage-3 avg, stage-3 max]; b3-b5 zeros (unused).

Run:  uv run features/extract_convnext.py [--model convnext_large.fb_in22k_ft_in1k --size 512 --out convnext_l22k_512]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import os
import numpy as np, timm, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "fve"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32); STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Images(Dataset):
    def __init__(self, paths, size): self.paths, self.size = paths, size
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy((np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)


@torch.inference_mode()
def embed(m, x):
    if HF:
        hs = m(pixel_values=x, output_hidden_states=True).hidden_states
        f3, f4 = hs[-2], hs[-1]
    else:
        f3, f4 = m(x)[-2:]
    avg, mx = (lambda t: t.mean((2, 3))), (lambda t: t.amax((2, 3)))
    d = f4.shape[1]
    b = [avg(f4), mx(f4), torch.cat([avg(f3), mx(f3)], 1)]
    assert b[2].shape[1] == d
    return torch.cat(b + [torch.zeros_like(b[0])] * 3, 1).float()


HF = False


def main():
    global HF
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convnext_large.fb_in22k_ft_in1k")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="convnext_l22k_512")
    a = ap.parse_args()
    keys, paths = [], []
    for split in ("train", "val"):
        for p in sorted((DATA / split).rglob("rgb_industrial.png")):
            keys.append(f"{split}/{p.parent.name}"); paths.append(p)
    HF = a.model.startswith("facebook/")
    if HF:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(a.model, token=os.environ.get("HF_TOKEN"), dtype=torch.bfloat16).cuda().eval()
    else:
        m = timm.create_model(a.model, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3)).cuda().eval().to(torch.bfloat16)
    print(f"{a.model}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M params, {len(paths)} images", flush=True)
    feats, t0 = [], time.time()
    for i, x in enumerate(DataLoader(Images(paths, a.size), batch_size=a.batch, num_workers=0)):
        x = x.cuda().to(torch.bfloat16)
        feats.append(((embed(m, x) + embed(m, torch.flip(x, dims=[3]))) / 2).cpu().numpy())
        if i % 40 == 0: print(f"  batch {i} {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
    X = np.concatenate(feats).astype(np.float32)
    assert np.isfinite(X).all()
    np.savez_compressed(PROJECT / "features" / f"{a.out}.npz", X=X, keys=np.array(keys), model=a.model, size=a.size)
    print(f"saved {X.shape} in {time.time()-t0:.0f}s")

main()

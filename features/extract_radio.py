# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "transformers", "timm", "einops", "pillow", "numpy"]
# ///
"""Frozen C-RADIOv4-H features (local weights dir), flip-averaged, fp16, 512 px.

6-block layout (d = 1280): b0, b1 = the two 1280-d halves of the RADIO summary vector; b2..b5 = mean of patch tokens after
blocks 32, 24, 16, 8. Use blocks (0, 1, 2) for the B3.4 screen.
Run: uv run features/extract_radio.py [--model ~/kaggle_stage/cradio]
"""
import argparse, os, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModel

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "fve"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.expanduser("~/kaggle_stage/cradio"))
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--out", default="cradio_v4h_512")
a = ap.parse_args()
keys, paths = [], []
for split in ("train", "val"):
    for p in sorted((DATA / split).rglob("rgb_industrial.png")):
        keys.append(f"{split}/{p.parent.name}"); paths.append(p)
m = AutoModel.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float16).cuda().eval()
vit = m.radio_model.model
cap = []
for i in (31, 23, 15, 7):
    vit.blocks[i].register_forward_hook(lambda mod, inp, out: cap.append(out[0] if isinstance(out, tuple) else out))
n_patch = (a.size // 16) ** 2


@torch.inference_mode()
def embed(x):
    cap.clear()
    s, _ = m(x)
    d = s.shape[1] // 2
    return torch.cat([s[:, :d], s[:, d:]] + [c[:, -n_patch:].mean(1) for c in cap], 1).float()


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((a.size, a.size), Image.BICUBIC)).copy()).permute(2, 0, 1)


feats, t0 = [], time.time()
for i in range(0, len(paths), a.batch):
    x = torch.stack([load(p) for p in paths[i:i + a.batch]]).cuda().half() / 255
    feats.append(((embed(x) + embed(x.flip(-1))) / 2).cpu().numpy())
    if (i // a.batch) % 40 == 0:
        print(f"  {i}/{len(paths)} {time.time()-t0:.0f}s peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
X = np.concatenate(feats).astype(np.float32)
assert np.isfinite(X).all(), "non-finite features"
np.savez_compressed(PROJECT / "features" / f"{a.out}.npz", X=X, keys=np.array(keys), model=a.model, size=a.size)
print("saved", X.shape, f"{time.time()-t0:.0f}s")

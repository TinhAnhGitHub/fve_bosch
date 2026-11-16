# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers", "pillow", "numpy"]
# ///
"""Frozen DINOv3 features for the size model, same recipe as Track 2's features/extract_dinov2.py.

Per photo: CLS + mean patch token (registers excluded) from three layers (last one after the final norm), averaged over
the photo and its horizontal flip. Photos are resized to SIZE x SIZE, no crop. Keys train/<id> and val/<id> (val = the
511 public eval photos), as in Track 2's feature files, so tools/size_knn.py can read them.

usage: extract_feats.py TRAIN_DIR EVAL_DIR OUT_DIR MODEL:SIZE:L1,L2,L3:BATCH:NAME [...]
  TRAIN_DIR / EVAL_DIR hold flat <id>.png photos. HF_TOKEN from the environment (gated models).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Images(Dataset):
    def __init__(self, paths: list[Path], size: int) -> None:
        self.paths, self.size = paths, size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy((np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD).permute(2, 0, 1)


@torch.inference_mode()
def embed(model: torch.nn.Module, x: torch.Tensor, layers: list[int], n_reg: int) -> torch.Tensor:
    out = model(pixel_values=x, output_hidden_states=True)
    hidden = out.hidden_states
    parts = []
    for layer in layers:
        h = out.last_hidden_state if layer == len(hidden) - 1 else hidden[layer]
        parts += [h[:, 0], h[:, 1 + n_reg:].mean(dim=1)]
    return torch.cat(parts, dim=1).float()


def main() -> None:
    train_dir, eval_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    keys, paths = [], []
    for split, d in (("train", train_dir), ("val", eval_dir)):
        for p in sorted(d.glob("*.png")):
            keys.append(f"{split}/{p.stem}")
            paths.append(p)
    print(f"{len(paths)} photos", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in sys.argv[4:]:
        name_model, size, layers, batch, name = spec.split(":")
        size, batch, layers = int(size), int(batch), [int(v) for v in layers.split(",")]
        t0 = time.time()
        model = AutoModel.from_pretrained(name_model, dtype=torch.bfloat16).cuda().eval()
        n_reg = getattr(model.config, "num_register_tokens", 0)
        print(f"{name}: {name_model} {model.config.num_hidden_layers} layers, {n_reg} registers, size {size}, "
              f"layers {layers}, load {time.time() - t0:.0f} s", flush=True)
        loader = DataLoader(Images(paths, size), batch_size=batch, num_workers=4, pin_memory=True)
        feats = []
        for i, x in enumerate(loader):
            x = x.cuda(non_blocking=True).to(torch.bfloat16)
            f = (embed(model, x, layers, n_reg) + embed(model, torch.flip(x, dims=[3]), layers, n_reg)) / 2
            feats.append(f.cpu().numpy())
            if i % 50 == 0:
                print(f"  {name} batch {i}/{len(loader)} {time.time() - t0:.0f} s, peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        X = np.concatenate(feats)
        if not np.isfinite(X).all():
            raise SystemExit(f"{name}: non-finite features")
        np.savez(out_dir / f"{name}.npz", X=X.astype(np.float16), keys=np.array(keys), model=name_model, size=size,
                 layers=np.array(layers))
        print(f"{name}: saved {X.shape} in {time.time() - t0:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

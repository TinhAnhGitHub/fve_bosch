# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Submission zip for the 511 public photos from one checkpoint or an aligned point mix of several.

usage: mix_zip.py OUT_ZIP MEMBER [MEMBER ...]
  MEMBER is a checkpoint, or several joined by commas: a pooled member (e.g. the 5 fold models of a recipe), whose
  clouds are first mixed among themselves (aligned to the group's first checkpoint) and then count as one member.

One member: its clouds as they are. Several: each member is aligned per photo to the first one
(best of the 8 axis sign flips, then rigid ICP; mixlib.align_to, as in RH-Mix5) and the points are
pooled with an equal share per member (atlas.mix, index stride), exactly 8,192 per photo.
The first member must be right-handed (the reference). Fine-tuned checkpoints (train --finetune) read the
photos through their own encoder blocks.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "baselines" / "tools"))
from atlas import ENCODERS, encoder, image_tensor, image_tokens, load_net, mix, rows, tuned_encoder  # noqa: E402
from geom import pack  # noqa: E402
from mixlib import align_to  # noqa: E402


def aligned_mix(clouds: list[torch.Tensor], i: int, dev: torch.device) -> np.ndarray:
    """Photo i: every cloud aligned to the first one, then an equal index-stride share of each."""
    ref = clouds[0][i].to(dev)
    if len(clouds) == 1:
        return ref.cpu().numpy()
    return mix([ref.cpu().numpy()] + [align_to(c[i].to(dev), ref).cpu().numpy() for c in clouds[1:]])


def main() -> None:
    out = Path(sys.argv[1])
    groups = [[Path(p) for p in spec.split(",")] for spec in sys.argv[2:]]
    ckpts = [p for g in groups for p in g]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = rows("val")
    if len(paths) != 511:
        raise SystemExit(f"expected the 511 public photos, found {len(paths)}")
    size = ENCODERS["dinov3l"]
    pixels = torch.stack([image_tensor(p, size) for p in paths])
    cats = [p.parent.parent.name for p in paths]
    frozen_tok = None
    clouds = []
    for ck in ckpts:
        net, blob = load_net(ck, dev)
        if blob.get("encoder") != "dinov3l":
            raise SystemExit(f"{ck.name}: encoder {blob.get('encoder')}, expected dinov3l")
        tuner = tuned_encoder(blob, dev)
        if tuner is None and frozen_tok is None:
            enc = encoder(dev, "dinov3l")
            with torch.inference_mode():
                frozen_tok = torch.cat([image_tokens(enc, pixels[i : i + 8].to(dev)).cpu() for i in range(0, len(paths), 8)])
            del enc
        vocab = blob["vocab"]
        cat = torch.tensor([vocab.get(c, vocab.get("<unk>", 0)) for c in cats], dtype=torch.long)
        with torch.inference_mode():
            parts = []
            for i in range(0, len(paths), 8):
                tok = frozen_tok[i : i + 8].to(dev) if tuner is None else tuner(pixels[i : i + 8].to(dev))
                parts.append(net(tok, cat[i : i + 8].to(dev)).float().cpu())
        clouds.append(torch.cat(parts))
        print(f"{ck.name}: predicted {len(paths)}", flush=True)
        del net, tuner
        torch.cuda.empty_cache()
    sizes = [len(g) for g in groups]
    starts = np.cumsum([0] + sizes)
    result = {}
    for i, p in enumerate(paths):
        members = [torch.from_numpy(aligned_mix(clouds[a:b], i, dev))[None] for a, b in zip(starts[:-1], starts[1:])]
        result[p.parent.name] = aligned_mix(members, 0, dev)
    pack(out, result)
    print(f"{len(result)} clouds from {len(groups)} member(s), {len(ckpts)} checkpoint(s) {sizes} -> {out}")


if __name__ == "__main__":
    main()

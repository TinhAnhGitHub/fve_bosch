# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Out-of-fold evaluation of a k-fold recipe.

usage:
  oof.py curve LOGDIR NAME                 per-epoch EMA check of NAME_f0..f4 (train logs), mean curve, E* = its argmin
  oof.py predict CKDIR NAME EPOCH OUT.npz [SUFFIX]  OOF clouds: fold k's objects from NAME_fk_ema_e<EPOCH><SUFFIX>.pt
                                           (SUFFIX _h = encoder blocks stored in fp16), strict scores

E* is one epoch per recipe, chosen on the mean curve of all folds, never a per-fold best, so the OOF score is not
picked on the objects it reports. Fine-tuned checkpoints (train --finetune) compute tokens with their own encoder
from the photo pack. Truth is the right-handed cloud (atlas.right_handed), scored strictly (centroid, then ICP).
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "baselines" / "tools"))
from atlas import PHOTOS, _load_cache, load_net, plain_pixels, right_handed, split, tuned_encoder  # noqa: E402
from mixlib import strict  # noqa: E402

FOLDS = ROOT / "baselines" / "folds" / "f3dr_k5_v1.csv"
K = 5


def curve(logdir: Path, name: str) -> int:
    curves = []
    for k in range(K):
        text = (logdir / f"{name}_f{k}.train.log").read_text()
        ema = {int(e): float(v) for e, v in re.findall(r"^epoch (\d+) .* ema ([\d.]+)", text, re.M)}
        curves.append(ema)
    common = sorted(set.intersection(*(set(c) for c in curves)))
    mean = {e: float(np.mean([c[e] for c in curves])) for e in common}
    best = min(mean, key=mean.get)
    for e in common:
        print(f"epoch {e:2d}  mean {mean[e]:6.2f}  " + " ".join(f"{c[e]:6.1f}" for c in curves) + ("  <- E*" if e == best else ""))
    print(f"{name}: E* = {best}, mean fold check {mean[best]:.2f}; per-fold best epochs {[min(c, key=c.get) for c in curves]}")
    return best


def predict(ckdir: Path, name: str, epoch: int, out: Path, suffix: str = "") -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z, _ = _load_cache("dinov3l")
    ids = z["ids"].tolist()
    feats = z["feats"]
    photos = None
    clouds = np.zeros((len(ids), *z["pts"].shape[1:]), np.float16)
    scores = np.full(len(ids), np.nan)
    fold_of = np.full(len(ids), -1)
    for k in range(K):
        val, _, _ = split(ids, FOLDS, str(k))
        fold_of[val] = k
        net, blob = load_net(ckdir / f"{name}_f{k}_ema_e{epoch}{suffix}.pt", dev)
        if blob.get("fold") != str(k):
            raise SystemExit(f"{name}_f{k} was trained with fold {blob.get('fold')!r}")
        tuner = tuned_encoder(blob, dev)
        if tuner is not None and photos is None:
            with np.load(PHOTOS, allow_pickle=False) as pz:
                photos = pz["imgs"]
        vocab = blob["vocab"]
        cat = torch.tensor([vocab.get(c, vocab.get("<unk>", 0)) for c in z["cats"].tolist()], dtype=torch.long)
        gt = right_handed(z["pts"][val], [ids[i] for i in val])
        with torch.inference_mode():
            for s in range(0, len(val), 8):
                b = val[s : s + 8]
                if tuner is None:
                    tok = torch.tensor(feats[b], dtype=torch.float32, device=dev)
                else:
                    tok = tuner(torch.stack([plain_pixels(photos[i]) for i in b]).to(dev))
                pred = net(tok, cat[b].to(dev)).float()
                for j, i in enumerate(b):
                    clouds[i] = pred[j].cpu().numpy()
                    scores[i] = strict(pred[j], torch.tensor(gt[s + j], device=dev))
        print(f"fold {k}: {len(val)} objects, strict mean {np.nanmean(scores[val]):.2f} median {np.nanmedian(scores[val]):.1f}", flush=True)
        del net, tuner
        torch.cuda.empty_cache()
    np.savez(out, ids=np.array(ids), fold=fold_of, pts=clouds, strict=scores, epoch=epoch, name=name)
    report(out)


def report(path: Path) -> None:
    """Mean and median OOF strict score, overall and by slice."""
    with np.load(path, allow_pickle=False) as o:
        ids, s, fold = o["ids"].tolist(), o["strict"], o["fold"]
    with open(FOLDS, newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    radius = np.array([float(rows[i]["radius"]) for i in ids])
    cats = [rows[i]["category"] for i in ids]
    count = {c: cats.count(c) for c in set(cats)}
    old = np.array([rows[i]["old_holdout"] == "1" for i in ids])
    big = radius >= np.quantile(radius, 0.6)
    single = np.array([count[c] == 1 for c in cats])

    def line(label: str, m: np.ndarray) -> str:
        return f"  {label:24s} n={int(m.sum()):4d}  mean {s[m].mean():6.2f}  median {np.median(s[m]):5.1f}"

    print(f"OOF {path.name}")
    print(line("all", np.ones(len(ids), bool)))
    for k in range(K):
        print(line(f"fold {k}", fold == k))
    print(line("largest 40%", big))
    print(line("other 60%", ~big))
    print(line("unseen category", single))
    print(line("old 204 holdout objects", old))
    print(f"  largest 40% share of total error: {s[big].sum() / s.sum():.1%}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "curve":
        curve(Path(sys.argv[2]), sys.argv[3])
    elif cmd == "predict":
        predict(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]), sys.argv[6] if len(sys.argv) > 6 else "")
    elif cmd == "report":
        report(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)

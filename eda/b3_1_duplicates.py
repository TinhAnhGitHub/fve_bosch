# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "matplotlib", "pillow"]
# ///
"""B3.1: near-duplicate check on frozen DINOv2 features.

Finds the most similar image pairs (train-train and val-train), compares train-pair volumes,
and saves a side-by-side sheet of the top pairs.

Run:  uv run eda/b3_1_duplicates.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT: Path = Path(__file__).resolve().parent.parent
DATA: Path = PROJECT / "data" / "fve"
OUT: Path = PROJECT / "eda" / "out"
FEATURES: Path = PROJECT / "features" / "dinov2l_reg_518.npz"
LAST_CLS = slice(4 * 1024, 5 * 1024)  # layer-24 CLS block


def meta() -> pd.DataFrame:
    """key -> split, category, sample_id, volume, image path."""
    rows = []
    for split in ("train", "val"):
        for img in sorted((DATA / split).rglob("rgb_industrial.png")):
            gt = img.parent / "gt.json"
            rows.append({"key": f"{split}/{img.parent.name}", "split": split, "category": img.parent.parent.name,
                         "sample_id": img.parent.name, "img": str(img),
                         "volume": json.loads(gt.read_text())["volume"]["primary_cm3"] if gt.exists() else np.nan})
    return pd.DataFrame(rows).set_index("key")


def main() -> None:
    npz = np.load(FEATURES)
    m = meta().loc[npz["keys"]]
    F = npz["X"][:, LAST_CLS]
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    S = F @ F.T
    np.fill_diagonal(S, -1)
    is_tr = (m["split"] == "train").to_numpy()

    tt = S[np.ix_(is_tr, is_tr)][np.triu_indices(is_tr.sum(), 1)]
    vt = S[np.ix_(~is_tr, is_tr)].ravel()
    q = np.quantile(tt, [0.5, 0.99, 0.999, 0.9999])
    print("train-train cosine quantiles 50/99/99.9/99.99%:", q.round(4), " max", tt.max().round(4))
    print("val-train   cosine max per val image: median", np.median(S[np.ix_(~is_tr, is_tr)].max(1)).round(4),
          " max", vt.max().round(4))

    # nearest train neighbour for every image
    tr_idx = np.flatnonzero(is_tr)
    rows = []
    for i in range(len(m)):
        s = S[i, tr_idx]
        j = tr_idx[np.argmax(s)]
        rows.append({"key": m.index[i], "split": m["split"].iloc[i], "nn_key": m.index[j], "cos": s.max(),
                     "same_category": m["category"].iloc[i] == m["category"].iloc[j],
                     "vol": m["volume"].iloc[i], "nn_vol": m["volume"].iloc[j]})
    nn = pd.DataFrame(rows).sort_values("cos", ascending=False)
    nn["vol_ratio"] = nn["vol"] / nn["nn_vol"]
    OUT.mkdir(parents=True, exist_ok=True)
    nn.to_csv(OUT / "b3_1_nearest_neighbours.csv", index=False)

    thr = q[2]  # 99.9th percentile of train-train similarity
    for split in ("train", "val"):
        d = nn[nn.split == split]
        print(f"\n{split}: images whose nearest train neighbour has cos > {thr:.4f}: "
              f"{(d.cos > thr).sum()} / {len(d)}  (same category: {(d[d.cos > thr].same_category).mean():.0%})")
    top = nn[nn.split == "train"].drop_duplicates(subset=["cos"]).head(30)
    print("\ntop train-train pairs (vol_ratio ~1 => likely same object; ~1000 => same object, unit error):")
    print(top[["key", "nn_key", "cos", "vol", "nn_vol", "vol_ratio"]].round(3).to_string(index=False))
    print("\ntop val-train pairs:")
    print(nn[nn.split == "val"].head(15)[["key", "nn_key", "cos", "nn_vol"]].round(3).to_string(index=False))

    # contact sheet: top 12 train-train and top 12 val-train pairs
    pairs = list(top.head(12).itertuples()) + list(nn[nn.split == "val"].head(12).itertuples())
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)))
    for r, p in enumerate(pairs):
        for c, (k, v) in enumerate([(p.key, p.vol), (p.nn_key, p.nn_vol)]):
            axes[r, c].imshow(Image.open(m.loc[k, "img"]).resize((256, 256)))
            axes[r, c].set_title(f"{k.split('/')[1]}\n{v:.1f} cm³  cos={p.cos:.3f}", fontsize=7)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "b3_1_top_pairs.png", dpi=90)
    print(f"\nsaved {OUT / 'b3_1_top_pairs.png'} and {OUT / 'b3_1_nearest_neighbours.csv'}")


if __name__ == "__main__":
    main()

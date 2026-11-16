# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Does a better size estimate improve the real score? Out-of-fold test on all 2,042 training objects.

Inputs: out-of-fold clouds of one recipe (oof.py predict) and out-of-fold sizes from size_knn.py.
  1. combined size: per fold, least squares on the other 4 folds:  ln true r ~ a ln own r + b ln knn r + c
     (own r = RMS radius of our centred cloud). Size errors of own (x best constant), knn and combined are printed.
  2. rescale each cloud about its centre: new r = own r * (k * combined / own r) ** gamma. gamma = 0 is the cloud as
     predicted; k < 1 keeps the deliberate shrink our clouds use to hedge against size errors.
  3. strict score (mixlib.strict: centroid, then ICP) of every (k, gamma) on every object; (k, gamma) chosen on 4 folds
     and reported on the 5th, so the final number is out of sample. Paired bootstrap of the gain over gamma = 0.

usage: size_rescale_oof.py OOF_CLOUDS.npz SIZE_KNN.npz STEP OUT.npz [GT.npz]
  GT.npz: truth clouds; written (then exit) if missing, read if present.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "baselines" / "tools"))
from atlas import _load_cache, right_handed  # noqa: E402
from mixlib import strict  # noqa: E402

import os  # noqa: E402

# Grid override for follow-up runs: RESCALE_KS / RESCALE_GAMMAS = comma lists.
KS = tuple(float(v) for v in os.environ.get("RESCALE_KS", "0.70,0.73,0.76,0.79,0.82,0.85").split(","))
GAMMAS = tuple(float(v) for v in os.environ.get("RESCALE_GAMMAS", "0.25,0.5,0.75,1.0").split(","))


def main() -> None:
    clouds_path, size_path, step, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o = np.load(clouds_path)
    ids = o["ids"].tolist()
    fold = o["fold"]
    pts = o["pts"].astype(np.float32)
    pts -= pts.mean(1, keepdims=True)
    own = np.sqrt((pts ** 2).sum(-1).mean(-1))
    sz = np.load(size_path)
    pos = {i: n for n, i in enumerate(sz["ids"].tolist())}
    order = [pos[i] for i in ids]
    true_r = sz["radius"][order]
    knn = sz["oof_ln_r"][list(sz["steps"]).index(step)][order]
    t, lo = np.log(true_r), np.log(own)

    comb = np.zeros(len(ids))
    own_c = np.zeros(len(ids))
    for k in range(5):
        tr, va = fold != k, fold == k
        M = np.stack([lo, knn, np.ones(len(ids))], 1)
        coef = np.linalg.lstsq(M[tr], t[tr], rcond=None)[0]
        comb[va] = M[va] @ coef
        own_c[va] = lo[va] + np.median(t[tr] - lo[tr])
        print(f"fold {k}: ln r = {coef[0]:.3f} own + {coef[1]:.3f} knn + {coef[2]:.3f}")
    for name, e in (("own x constant", own_c - t), ("size_knn", knn - t), ("combined", comb - t)):
        print(f"size error {name:15s} median |log| {np.median(np.abs(e)):.4f}  mean {np.mean(np.abs(e)):.4f}")

    variants = [(1.0, 0.0)] + [(k, g) for k in KS for g in GAMMAS]
    factor = np.stack([(k * np.exp(comb) / own) ** g for k, g in variants])  # (V, n)
    gt_path = Path(sys.argv[5]) if len(sys.argv) > 5 else None  # truth clouds file: lets a GPU VM skip train.npz
    if gt_path and gt_path.exists():
        g = np.load(gt_path)
        assert g["ids"].tolist() == ids
        gt = g["gt"]
    else:
        z, _ = _load_cache("dinov3l")
        zid = z["ids"].tolist()
        zpos = {i: n for n, i in enumerate(zid)}
        rows = np.array([zpos[i] for i in ids])
        gt = right_handed(z["pts"][rows], ids)
        if gt_path:
            np.savez(gt_path, ids=np.array(ids), gt=gt.astype(np.float32))
            raise SystemExit(f"saved {gt_path}")
    S = np.zeros((len(variants), len(ids)))
    with torch.inference_mode():
        for n in range(len(ids)):
            p = torch.tensor(pts[n], device=dev)
            g = torch.tensor(gt[n], dtype=torch.float32, device=dev)
            for v in range(len(variants)):
                S[v, n] = strict(p * float(factor[v, n]), g)
            if n % 200 == 0:
                print(f"scored {n}/{len(ids)}", flush=True)
    print(f"as predicted (gamma 0): mean {S[0].mean():.2f} median {np.median(S[0]):.2f}  (oof file says {np.nanmean(o['strict']):.2f})")
    for v, (k, g) in enumerate(variants[1:], 1):
        print(f"k {k:.2f} gamma {g:.2f}: mean {S[v].mean():6.2f} median {np.median(S[v]):6.2f}")
    cv = np.zeros(len(ids))
    for k in range(5):
        tr, va = fold != k, fold == k
        best = int(np.argmin(S[:, tr].mean(1)))
        cv[va] = S[best, va]
        print(f"fold {k}: picked k {variants[best][0]:.2f} gamma {variants[best][1]:.2f}")
    gain = S[0] - cv
    rng = np.random.default_rng(0)
    boot = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(2000)])
    print(f"cross-validated rescale: mean {cv.mean():.2f} median {np.median(cv):.2f} | as predicted {S[0].mean():.2f} "
          f"{np.median(S[0]):.2f} | gain {gain.mean():.2f} (95% CI {np.quantile(boot, 0.025):.2f} .. {np.quantile(boot, 0.975):.2f})")
    np.savez(out, ids=np.array(ids), fold=fold, variants=np.array(variants), scores=S, cv=cv, comb_ln_r=comb, own_r=own)


if __name__ == "__main__":
    main()

"""Side quest on all 2,042 training objects: SAM 3D's shape at our predicted length / width / height vs our own cloud.

SAM 3D never saw these photos' meshes, and every size comes from out-of-fold size models, so all 2,042 are fair.
Frame: SAM 3D's mesh points are put in the canonical frame our submissions use (geom.canonicalize: principal axes,
skewness signs), made right-handed like the truth (flip the least-skewed axis when det < 0), then each principal axis
is scaled to k x the predicted spread (size_knn --axes, longest / middle / shortest). Scored with mixlib.strict
(centroid, then ICP), the scorer whose out-of-fold numbers matched the leaderboard.

Variants per object (S rows):
  ours            our v27 fold cloud resized with the v43 rule (k 0.85, gamma 0.25 toward size_knn)
  sam_k<k>        SAM 3D at k x predicted spreads, k in KS
  sam_true        SAM 3D at the true spreads (ceiling)
  sam_flip        best of the 8 axis sign flips of sam_k0.90 (oracle: what the frame ambiguity costs)
  mix_k<k>        half the points from ours, half from sam_k<k> (index stride)
Then per-object pick rules (SAM 3D when the size model's neighbour spread is low), thresholds chosen on 4 folds,
scored on the 5th, and a paired bootstrap of each against ours.

usage: sam3d_oof.py OURS.npz GT.npz SIZE_KNN.npz SIZE_AXES.npz OUT.npz SAM3D.npz [SAM3D.npz ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "baselines" / "tools"))
from geom import canonicalize  # noqa: E402
from mixlib import SIGNS, cd, strict  # noqa: E402

STEP = 35
KS = (0.80, 0.85, 0.90, 0.95, 1.00)


def right_frame(p: np.ndarray) -> np.ndarray:
    """Canonical frame (longest axis first), right-handed: flip the least-skewed axis if the frame is a mirror."""
    y = canonicalize(p)
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    signs = np.where(np.mean(((p - p.mean(0)) @ vt.T) ** 3, axis=0) < 0, -1.0, 1.0)
    if np.linalg.det(vt.T * signs) < 0:
        y[:, int(np.argmin(np.abs((y ** 3).mean(0))))] *= -1
    return y


def main() -> None:
    ours_p, gt_p, size_p, axes_p, out = (Path(a) for a in sys.argv[1:6])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o, g, sz, ax = np.load(ours_p), np.load(gt_p), np.load(size_p), np.load(axes_p)
    ids = o["ids"].tolist()
    assert g["ids"].tolist() == ids and sz["ids"].tolist() == ids and ax["ids"].tolist() == ids
    fold = o["fold"]
    s_r, s_a = list(sz["steps"]).index(STEP), list(ax["steps"]).index(STEP)
    ln_sd, spread = ax["oof_ln_sd"][s_a], sz["oof_spread"][s_r]  # (3, n), (n,)
    sam = {}
    for f in sys.argv[6:]:
        with np.load(f) as z:
            for i, p in zip(z["ids"].tolist(), z["pts_mesh"]):
                if np.isfinite(p).all():
                    sam[str(i)] = p.astype(np.float64)
    keep = np.array([i in sam for i in ids])
    print(f"SAM 3D clouds for {keep.sum()} of {len(ids)} objects", flush=True)

    # ours with the v43 rule (combined size = size_knn, as rescale_zip uses it)
    pts = o["pts"].astype(np.float64)
    pts -= pts.mean(1, keepdims=True)
    own = np.sqrt((pts ** 2).sum(-1).mean(-1))
    knn_r = np.exp(sz["oof_ln_r"][s_r])
    ours = pts * ((0.85 * knn_r / own) ** 0.25)[:, None, None]

    names = ["ours"] + [f"sam_k{k:.2f}" for k in KS] + ["sam_true", "sam_flip"] + [f"mix_k{k:.2f}" for k in KS]
    S = np.full((len(names), len(ids)), np.nan)
    t32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    with torch.inference_mode():
        for n, i in enumerate(ids):
            t = t32(g["gt"][n])
            S[0, n] = strict(t32(ours[n]), t)
            if keep[n]:
                q = right_frame(sam[i])
                sd_q = q.std(0)
                pred = q * (np.exp(ln_sd[:, n]) / sd_q)
                for a, k in enumerate(KS):
                    S[1 + a, n] = strict(t32(k * pred), t)
                    mixed = np.concatenate([ours[n][0::2], (k * pred)[1::2]])
                    S[3 + len(KS) + a, n] = strict(t32(mixed), t)
                tq = right_frame(g["gt"][n]).std(0)
                S[1 + len(KS), n] = strict(t32(q * (tq / sd_q)), t)
                p9 = t32(0.90 * pred)
                best = min(SIGNS.to(dev), key=lambda s: cd((p9 * s)[::8] - (p9 * s).mean(0) + t.mean(0), t[::8]))
                S[2 + len(KS), n] = strict(p9 * best, t)
            if n % 200 == 0:
                print(f"scored {n}/{len(ids)}", flush=True)

    m = keep
    rng = np.random.default_rng(0)

    def report(name: str, v: np.ndarray) -> None:
        d = S[0, m] - v
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        print(f"{name:14s} mean {v.mean():7.2f} median {np.median(v):7.2f} | gain vs ours {d.mean():6.2f} "
              f"(95% CI {np.quantile(boot, 0.025):.2f} .. {np.quantile(boot, 0.975):.2f}), better on {np.mean(d > 0) * 100:.0f}%")

    print(f"\non the {m.sum()} objects with a SAM 3D cloud (CD-L2 mm^2, strict):")
    for r, name in enumerate(names):
        report(name, S[r, m])
    # pick rule: SAM 3D (or the mix) when the neighbour spread is below a threshold chosen on the other folds
    qs = np.quantile(spread[m], np.linspace(0, 1, 11))
    for r, name in enumerate(names[1:], 1):
        if name in ("sam_true", "sam_flip"):
            continue
        pick = np.zeros(len(ids))
        for k in range(5):
            tr, va = m & (fold != k), m & (fold == k)
            cost = [np.where(spread[tr] <= th, S[r, tr], S[0, tr]).mean() for th in qs]
            th = qs[int(np.argmin(cost))]
            pick[va] = np.where(spread[va] <= th, S[r, va], S[0, va])
        report(f"pick:{name}", pick[m])
    np.savez(out, ids=np.array(ids), fold=fold, names=np.array(names), scores=S, spread=spread, keep=keep)


if __name__ == "__main__":
    main()

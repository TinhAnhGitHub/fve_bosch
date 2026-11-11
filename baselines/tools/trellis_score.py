# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""Score TRELLIS shapes on the 204 holdout objects. usage: trellis_score.py root trellis_clouds.npz [ref_ckpt]

trellis_clouds.npz: ids (204,), pts (204, N, 3) from TRELLIS, any frame and scale.
Arms (all scored like our models: centre + 10 ICP steps, vs real-handed holdout shapes):
  ours      reference model alone (default cloud/s_posrh_s1_ema.pt)
  oracle    TRELLIS shape scaled to the TRUE size, best of 24 rotations + ICP onto the truth (shape quality only)
  predsize  TRELLIS shape scaled to OUR predicted size, rotation found against OUR predicted cloud (a real submission)
  pred_icp  as predsize, then ICP with scale onto our cloud, scale clamped to +-15%
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

root, npz = Path(sys.argv[1]), Path(sys.argv[2])
ref = Path(sys.argv[3]) if len(sys.argv) > 3 else root / "baselines/out/atlas/cloud/s_posrh_s1_ema.pt"
sys.path.insert(0, str(root / "baselines"))
from atlas import _load_cache, holdout, kabsch, load_net, nearest, right_handed

dev = torch.device("cuda")
z, _ = _load_cache("dinov3l")
val, _, _ = holdout(len(z["ids"]))
ids = z["ids"][val].tolist()
gt = torch.tensor(right_handed(z["pts"][val], ids), dtype=torch.float32).to(dev)
tz = np.load(npz, allow_pickle=False)
tr = {i: torch.tensor(p, dtype=torch.float32) for i, p in zip(tz["ids"].tolist(), tz["pts"])}

net, blob = load_net(ref, dev)
feats = torch.tensor(z["feats"][val], dtype=torch.float32)
cat = torch.tensor([blob["vocab"][c] for c in z["cats"][val].tolist()])
with torch.inference_mode():
    ours = torch.cat([net(feats[i : i + 4].to(dev), cat[i : i + 4].to(dev)).float() for i in range(0, len(val), 4)])

# the 24 proper rotations of the cube (axis permutations x sign flips with det +1)
ROT = torch.stack([torch.tensor(np.diag(s)[:, list(p)], dtype=torch.float32) for p in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3) if np.linalg.det(np.diag(s)[:, list(p)]) > 0.5]).to(dev)


@torch.no_grad()
def cd(q, t):
    return float(nearest(q, t)[0].mean() + nearest(t, q)[0].mean())


@torch.no_grad()
def icp(q, t, iters=10, scale=False):
    for _ in range(iters):
        _, j = nearest(q, t)
        R, tr_ = kabsch(q, t[j])
        q = q @ R.T + tr_
    return q


def pca(p):
    p = p - p.mean(0)
    _, _, vt = torch.linalg.svd(p, full_matrices=False)
    return p @ vt.T  # principal axes, sign and order ambiguous


def rms(p):
    return float((p - p.mean(0)).square().sum(-1).mean().sqrt())


@torch.no_grad()
def best_rot(p, target):
    """p (n,3) centred; the rotation from the 24 (after PCA) that gives the lowest Chamfer to target, on a subsample."""
    a = pca(p)
    best, bd = None, 1e18
    for R in ROT:
        q = a @ R.T
        for pca_t in (pca(target),):
            d = cd(q[::8] + target.mean(0), target[::8])
            if d < bd:
                best, bd = q, d
    return best + target.mean(0)


@torch.no_grad()
def strict(pred, t):
    return cd(icp(pred - pred.mean(0) + t.mean(0), t), t)


res = {k: [] for k in ("ours", "oracle", "predsize", "pred_icp")}
for k, i in enumerate(ids):
    t, o = gt[k], ours[k]
    p = tr[i].to(dev)
    res["ours"].append(strict(o, t))
    q = pca(p) * (rms(t) / rms(p))
    res["oracle"].append(strict(best_rot(q, t), t))
    q = pca(p) * (rms(o) / rms(p))
    q = best_rot(q, o)
    res["predsize"].append(strict(q, t))
    q2 = icp(q, o)
    s = float(np.clip(rms(o) / rms(q2), 0.85, 1.15))
    c = q2.mean(0)
    res["pred_icp"].append(strict(c + s * (q2 - c), t))
    if k % 20 == 0:
        print(k, {n: round(float(np.mean(v)), 1) for n, v in res.items()}, flush=True)
for n, v in res.items():
    v = np.array(v)
    print(f"{n:9s} mean {v.mean():6.2f}  median {np.median(v):5.1f}")
better = np.array(res["predsize"]) < np.array(res["ours"])
print(f"TRELLIS (predicted size) beats our model on {better.mean():.0%} of objects")
np.savez(npz.with_suffix(".scores.npz"), **{n: np.array(v) for n, v in res.items()}, ids=np.array(ids))

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "torch", "transformers", "trimesh"]
# ///
"""One candidate against the baseline: strict and 4-flip graders, normal and oversized objects, bootstrap."""
import sys, numpy as np, torch
from pathlib import Path
root, base, cand = Path(sys.argv[1]), sys.argv[2], sys.argv[3]; sys.path.insert(0, str(root / "baselines"))
from atlas import _load_cache, holdout, load_net, FLIPS, nearest, kabsch
dev = torch.device("cuda"); F = FLIPS.to(dev)
z, v = _load_cache("dinov3l"); val, _, _ = holdout(len(z["ids"]))
gt = torch.tensor(z["pts"][val], dtype=torch.float32).to(dev)
zs = np.load(root / "baselines/out/scale/scale.npz"); uu = dict(zip(zs["ids"], zs["u"])); u = np.array([uu[i] for i in z["ids"][val]])
f = torch.tensor(z["feats"][val], dtype=torch.float32); c = torch.tensor([v[x] for x in z["cats"][val]])
@torch.no_grad()
def score(pred, g, flips, iters=10):
    best = (float("inf"), 0.0, 0.0)
    for fl in flips:
        q = pred * fl; q = q - q.mean(0) + g.mean(0)
        for _ in range(iters):
            _, j = nearest(q, g); R, t = kabsch(q, g[j]); q = q @ R.T + t
        a, b = float(nearest(q, g)[0].mean()), float(nearest(g, q)[0].mean())
        if a + b < best[0]: best = (a + b, a, b)
    return best
res = {}
for name in (base, cand):
    net, _ = load_net(root / f"baselines/out/atlas/{name}.pt", dev)
    with torch.inference_mode():
        P = torch.cat([net(f[i:i+4].to(dev), c[i:i+4].to(dev)) for i in range(0, len(val), 4)])
    s = np.array([score(P[k], gt[k], F[:1]) for k in range(len(val))]); a = np.array([score(P[k], gt[k], F)[0] for k in range(len(val))])
    res[name] = s[:, 0]; n = u == 0
    print(f"{name:34s} strict normal {s[n,0].mean():5.1f} (stray {s[n,1].mean():5.1f}, missing {s[n,2].mean():5.1f})  4-flip normal {a[n].mean():5.1f}  strict over {s[u==1,0].mean():8.1f}", flush=True)
    del net; torch.cuda.empty_cache()
if base != cand:
    n = np.where(u == 0)[0]; rng = np.random.default_rng(0); ni = rng.integers(0, len(n), (2000, len(n)))
    d = res[cand] - res[base]; nd = d[n][ni].mean(1)
    print(f"RESULT {cand}: lower than {base} in {100*(nd<0).mean():.1f}% of resamples; mean change {d[n].mean():+.1f} mm^2 (5-95% {np.percentile(nd,5):+.1f} to {np.percentile(nd,95):+.1f})", flush=True)

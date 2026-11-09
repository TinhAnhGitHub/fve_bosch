"""How much did the fixed meshes change the training point clouds? usage: radii.py new.npz old.npz scale.npz"""
import sys
import numpy as np

new, old, sc = (np.load(p, allow_pickle=False) for p in sys.argv[1:4])
rad = lambda z: np.sqrt(((z["pts"] - z["pts"].mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
assert new["ids"].tolist() == old["ids"].tolist(), "ids differ"
a, b = rad(old), rad(new)
flag = dict(zip(sc["ids"].tolist(), sc["u"].tolist()))
u = np.array([flag[i] != 0 for i in new["ids"].tolist()])
print(f"objects {len(b)}; radius changed >1%: {int((np.abs(b / a - 1) > 0.01).sum())}; flagged oversized before: {int(u.sum())}")
print(f"old radius median {np.median(a):.1f} max {a.max():.1f}; new radius median {np.median(b):.1f} max {b.max():.1f}")
print(f"previously flagged, median radius old {np.median(a[u]):.1f} new {np.median(b[u]):.1f}")
print("new radius above 100/150/200 mm:", [int((b > t).sum()) for t in (100, 150, 200)])
for i in np.argsort(-b)[:5]:
    print(f"  largest {new['ids'][i]} {new['cats'][i]} old {a[i]:.1f} new {b[i]:.1f}")

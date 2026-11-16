"""Pick the resize rule (k, gamma) per size-confidence group, honestly, from size_rescale_oof.py's per-object scores.

Groups: thirds of size_knn's neighbour spread (oof_spread), cut points taken on the training folds. For each held-out
fold, the (k, gamma) with the lowest mean on the other 4 folds is chosen per group (and, for comparison, one rule for
all), then scored on the held-out fold. Everything is compared with v43's rule (k 0.85, gamma 0.25) by a paired
bootstrap over the 2,042 objects.

usage: rescale_conf.py SIZE_KNN.npz STEP SCORES.npz [SCORES.npz ...]   (shard files are merged)
"""
import csv
import sys
from pathlib import Path

import numpy as np

V43 = (0.85, 0.25)


def main() -> None:
    sz, step = np.load(sys.argv[1]), int(sys.argv[2])
    parts = [np.load(f) for f in sys.argv[3:]]
    ids, variants = parts[0]["ids"].tolist(), [tuple(v) for v in parts[0]["variants"].tolist()]
    S = np.full(parts[0]["scores"].shape, np.nan)
    for p in parts:
        assert p["ids"].tolist() == ids and [tuple(v) for v in p["variants"].tolist()] == variants
        S = np.where(np.isfinite(p["scores"]), p["scores"], S)
    assert np.isfinite(S).all(), "shards do not cover every object"
    assert sz["ids"].tolist() == ids
    spread = sz["oof_spread"][list(sz["steps"]).index(step)]
    with open(Path(__file__).resolve().parents[1] / "folds" / "f3dr_k5_v1.csv") as fh:
        fmap = {r["id"]: int(r["fold"]) for r in csv.DictReader(fh)}
    fold = np.array([fmap[i] for i in ids])
    ref = S[variants.index(V43)]
    print(f"v43 rule k {V43[0]} gamma {V43[1]}: mean {ref.mean():.2f} median {np.median(ref):.2f}")
    print("best fixed rules (in-sample, for reading only):")
    for v in np.argsort(S.mean(1))[:6]:
        print(f"  k {variants[v][0]:.2f} gamma {variants[v][1]:.2f}: mean {S[v].mean():.2f} median {np.median(S[v]):.2f}")

    rng = np.random.default_rng(0)

    def report(name: str, x: np.ndarray) -> None:
        d = ref - x
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        print(f"{name:30s} mean {x.mean():6.2f} median {np.median(x):6.2f} | gain {d.mean():5.2f} "
              f"(95% CI {np.quantile(boot, 0.025):.2f} .. {np.quantile(boot, 0.975):.2f}), better on {np.mean(d > 0) * 100:.0f}%")

    for groups in (1, 2, 3, 4):
        out = np.zeros(len(ids))
        picks = []
        for k in range(5):
            tr, va = fold != k, fold == k
            cuts = np.quantile(spread[tr], np.linspace(0, 1, groups + 1)[1:-1])
            g_tr, g_va = np.searchsorted(cuts, spread[tr]), np.searchsorted(cuts, spread[va])
            pick = [int(np.argmin(S[:, tr][:, g_tr == g].mean(1))) for g in range(groups)]
            picks.append([variants[p] for p in pick])
            out[va] = S[np.array(pick)[g_va], np.flatnonzero(va)]
        report(f"CV pick, {groups} spread group(s)", out)
        for g in range(groups):
            print(f"    group {g}: " + ", ".join(f"k{p[g][0]:.2f}/g{p[g][1]:.2f}" for p in picks))


if __name__ == "__main__":
    main()

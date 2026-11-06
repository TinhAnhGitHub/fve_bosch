# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3.3: swap DINOv2 for DINOv3 features (alone and combined), using the best B3.2 settings.

Run:  uv run experiments/b3_3_dinov3.py --n_pca 256 --C 0.05 --tau 0.5 --clip 1.5 --blocks 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from frozen import FEATURE_DIR, Config, Experiment  # noqa: E402

FILES = ["dinov2l_reg_518", "dinov3_vitl16_512", "dinov3_vith16plus_512", "dinov3_vitl16_768"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pca", type=lambda s: None if s == "None" else int(s), default=256)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.5)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    a = ap.parse_args()
    blocks = tuple(a.blocks)
    knobs = dict(n_pca=a.n_pca, C=a.C, tau=a.tau, r_clip=a.clip)
    present = [f for f in FILES if (FEATURE_DIR / f"{f}.npz").exists()]
    print("feature files:", present, "| knobs:", knobs, "blocks:", blocks)

    exp = Experiment()
    res = {}
    for f in present:
        cfg = Config(name=f"B3.3 {f}", features=((f, blocks),), **knobs)
        res[cfg] = exp.evaluate(cfg)
    v3 = [f for f in present if f.startswith("dinov3")]
    if v3 and "dinov2l_reg_518" in present:
        best_v3 = min((c for c in res if c.features[0][0].startswith("dinov3")), key=lambda c: res[c]["smape"])
        f3 = best_v3.features[0][0]
        for n in (a.n_pca, 512):
            cfg = Config(name=f"B3.3 v2 + {f3} pca={n}",
                         features=(("dinov2l_reg_518", blocks), (f3, blocks)), **{**knobs, "n_pca": n})
            res[cfg] = exp.evaluate(cfg)
    if len(v3) >= 2:
        cfg = Config(name="B3.3 all DINOv3 files", features=tuple((f, blocks) for f in v3), **{**knobs, "n_pca": 512})
        res[cfg] = exp.evaluate(cfg)

    best = min(res, key=lambda c: res[c]["smape"])
    print(f"\nbest: {best.name}  {res[best]['smape']:.2f}")
    print("fresh seeds 3-5:")
    exp.evaluate(Config(name=f"{best.name} [fresh seeds]", features=best.features, n_pca=best.n_pca, C=best.C,
                        tau=best.tau, r_clip=best.r_clip), seeds=(3, 4, 5))
    exp.permutation_test(best, n_perm=20)


if __name__ == "__main__":
    main()

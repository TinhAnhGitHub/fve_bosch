# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3.6: SigLIP2 in place of (or next to) DINOv3 in the best B3.4 kNN pipeline.

For each feature set: full B3.4 pipeline (ridge + classifier + kNN blend), blend weights and k
re-picked on seeds 0-2. The best set is then re-checked against B3.4 on fresh seeds 3-5.

Run:  uv run experiments/b3_6_siglip2.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
import b3_4_knn as knn  # noqa: E402
from b0_b1 import smape, write_submission  # noqa: E402
from frozen import Experiment  # noqa: E402

knn.KS = (2, 3)
CLS = (0, 2, 4)
V2, V3H, SG = ("dinov2l_reg_518", CLS), ("dinov3_vith16plus_512", CLS), ("siglip2_so400m_512", CLS)
SETS = {
    "B3.4  DINOv2 + DINOv3-H+": (V2, V3H),
    "SigLIP2 alone": (SG,),
    "DINOv2 + SigLIP2 (replace DINOv3)": (V2, SG),
    "DINOv3-H+ + SigLIP2": (V3H, SG),
    "DINOv2 + DINOv3-H+ + SigLIP2": (V2, V3H, SG),
}


def run(exp: Experiment, feats: tuple, seeds: tuple[int, ...]) -> list[dict[str, np.ndarray]]:
    knn.CFG = replace(knn.CFG, features=feats)
    return [knn.collect(exp, s) for s in seeds]


def main() -> None:
    exp = Experiment()
    results = {}
    for name, feats in SETS.items():
        outs = run(exp, feats, (0, 1, 2))
        g = knn.score_grid(exp, outs)
        best = g.iloc[0]
        alone = g[(g.w_ridge == 0) & (g.w_clf == 0)].smape.min()
        no_knn = g[(g.w_ridge == 1) & (g.w_clf == 1)].smape.min()
        results[name] = (feats, best)
        print(f"{name:36s} best {best.smape:6.2f} ±{best['std']:.2f} (k={int(best.k)}, w_ridge={best.w_ridge}, "
              f"w_clf={best.w_clf}) | no-kNN {no_knn:6.2f} | kNN-only {alone:6.2f}", flush=True)

    ref_name = "B3.4  DINOv2 + DINOv3-H+"
    win = min(results, key=lambda n: results[n][1].smape)
    print(f"\nbest feature set on seeds 0-2: {win}")
    if win == ref_name:
        print("B3.4 stays best: no new submission")
        return
    fresh = {}
    for name in (ref_name, win):
        feats, b = results[name]
        outs = run(exp, feats, (3, 4, 5))
        fresh[name] = np.mean([smape(exp.y, np.exp(knn.decode(o, int(b.k), b.w_ridge, b.w_clf))) for o in outs])
        print(f"fresh seeds 3-5: {name:36s} {fresh[name]:.2f}")
    gain = fresh[ref_name] - fresh[win]
    print(f"gain on fresh seeds: {gain:.2f}")
    if gain >= 0.5:
        feats, b = results[win]
        knn.CFG = replace(knn.CFG, features=feats)
        knn.submit(exp, int(b.k), b.w_ridge, b.w_clf, "b3_6_siglip2")


if __name__ == "__main__":
    main()

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""B3.2: tune the B3 pipeline on cached DINOv2 features (coordinate search, then fresh-seed check).

Stages: A) feature layers/blocks, B) PCA size, C) classifier strength C, D) mode threshold + clip.
Every run is appended to experiments/log.csv.

Run:  uv run experiments/b3_2_tune.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from frozen import Config, Experiment  # noqa: E402

V2 = "dinov2l_reg_518"
BLOCKS = {"all 3 layers": (0, 1, 2, 3, 4, 5), "last layer": (4, 5), "layers 18+24": (2, 3, 4, 5),
          "CLS only": (0, 2, 4), "mean-patch only": (1, 3, 5)}


def best(results: dict[Config, dict]) -> Config:
    return min(results, key=lambda c: results[c]["smape"])


def main() -> None:
    exp = Experiment()
    base = Config(name="B3.2 base (= B3)")
    res = {base: exp.evaluate(base)}

    print("\n-- A: which layers / tokens")
    for label, blocks in BLOCKS.items():
        cfg = replace(base, name=f"B3.2A blocks={label}", features=((V2, blocks),))
        if cfg.features != base.features:
            res[cfg] = exp.evaluate(cfg)
    cur = best(res)

    print("\n-- B: PCA size")
    for n in (128, 512, None):
        cfg = replace(cur, name=f"B3.2B {cur.name.split('blocks=')[-1]} pca={n}", n_pca=n)
        res[cfg] = exp.evaluate(cfg)
    cur = best(res)

    print("\n-- C: classifier strength")
    for C in (0.005, 0.01, 0.02, 0.1, 0.3, 1.0):
        cfg = replace(cur, name=f"B3.2C C={C}", C=C)
        res[cfg] = exp.evaluate(cfg)
    cur = best(res)

    print("\n-- D: mode threshold and correction clip")
    for tau in (0.3, 0.4, 0.6, 0.7):
        cfg = replace(cur, name=f"B3.2D tau={tau}", tau=tau)
        res[cfg] = exp.evaluate(cfg)
    cur = best(res)
    for clip in (0.75, 1.0, 2.5):
        cfg = replace(cur, name=f"B3.2D clip={clip}", r_clip=clip)
        res[cfg] = exp.evaluate(cfg)
    cur = best(res)

    print(f"\nbest config: {cur}")
    print("\n-- fresh seeds 3-5 (not used for any choice above)")
    exp.evaluate(replace(base, name="B3 base [fresh seeds]"), seeds=(3, 4, 5))
    exp.evaluate(replace(cur, name=f"{cur.name} [fresh seeds]"), seeds=(3, 4, 5))


if __name__ == "__main__":
    main()

"""Can the LoRA help RH-Mix5 anywhere? Object by object on the 204 holdout objects, both scored by the same scorer.

For every holdout object there is the CD-L2 of RH-Mix5 (cd_rhmix5.npz, from size_lookalike.py score_file on the clouds of
ens.py holdout_save) and of the LoRA at the look-alike size (holdout_cd.npz). This prints, per group of look-alike agreement:
how often the LoRA wins, and what a per-object choice would score: LoRA where the two look-alikes agree closely enough, RH-Mix5
elsewhere. The thresholds are tried on the same 204 objects that score them, so the best one is optimistic; all are shown.
A paired bootstrap over objects gives the noise of each difference.

usage: hybrid_rhmix.py SIZE_DIR      (needs numpy)
"""
import sys
from pathlib import Path

import numpy as np

CUTS = (0.056, 0.174)
NAMES = ("agree", "partly", "differ", "one only")


def main():
    out = Path(sys.argv[1])
    cd, sizes, rh = np.load(out / "holdout_cd.npz"), np.load(out / "holdout_sizes.npz"), np.load(out / "cd_rhmix5.npz")
    ids = cd["ids"].tolist()
    index = {i: n for n, i in enumerate(rh["ids"].tolist())}
    if set(index) != set(ids):
        raise SystemExit("RH-Mix5 and LoRA scores cover different objects")
    lora = cd["cd"][:, cd["arms"].tolist().index("blend")]
    mix5 = rh["cd"][[index[i] for i in ids]]
    gap = sizes["neighbour_gap"]
    group = np.where(np.isnan(gap), 3, np.digitize(np.nan_to_num(gap), CUTS))
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(ids), size=(2000, len(ids)))

    def interval(difference):
        means = difference[pick].mean(1)
        return np.percentile(means, [2.5, 97.5])

    print(f"{len(ids)} holdout objects, same scorer for both")
    print(f"RH-Mix5 alone                 mean {mix5.mean():6.2f}  median {np.median(mix5):6.2f}")
    print(f"LoRA at the look-alike size   mean {lora.mean():6.2f}  median {np.median(lora):6.2f}")
    print(f"per-object best of the two (an oracle, not usable)   mean {np.minimum(lora, mix5).mean():6.2f}")
    print("\nby how much the two look-alikes agree:")
    print("  group     objects  RH-Mix5 mean/median   LoRA mean/median    LoRA better on")
    for g, name in enumerate(NAMES):
        m = group == g
        if not m.any():
            continue
        print(f"  {name:9s} {m.sum():5d}    {mix5[m].mean():6.1f} / {np.median(mix5[m]):5.1f}      {lora[m].mean():6.1f} / {np.median(lora[m]):5.1f}"
              f"      {int((lora[m] < mix5[m]).sum())} of {m.sum()}")
    print("\nhybrid: LoRA where the look-alikes' log-radius gap is below the threshold, RH-Mix5 elsewhere")
    print("  threshold   LoRA objects   hybrid mean   change vs RH-Mix5 alone (95% interval)")
    for threshold in (0.0, 0.03, 0.056, 0.10, 0.174, 0.30, 9.0):
        use = np.nan_to_num(gap, nan=9.9) < threshold
        hybrid = np.where(use, lora, mix5)
        low, high = interval(hybrid - mix5)
        print(f"  {threshold:6.3f}      {int(use.sum()):5d}        {hybrid.mean():7.2f}      {hybrid.mean() - mix5.mean():+6.2f}  ({low:+.2f} to {high:+.2f})")
    print("\n(threshold 0 is RH-Mix5 alone; 9 is the LoRA alone except where only one look-alike exists)")


if __name__ == "__main__":
    main()

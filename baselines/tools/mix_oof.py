"""Does adding fold members to the point mix help? Out-of-fold, on all 2,042 objects.

Each OOF file (oof.py predict) holds one recipe's out-of-fold clouds. A mix aligns every member to the first one per
object (mixlib.align_to: best axis sign flip, then ICP), pools an equal share of points (atlas.mix, exactly 8,192), and
is scored with mixlib.strict against the truth (gt_oof.npz), the same way mix_zip.py builds submissions.
Reported: each member alone and every prefix mix (first 2, first 3, ...), with a paired bootstrap against the first
member alone, and against the previous prefix.

usage: mix_oof.py GT.npz OUT.npz OOF.npz OOF.npz [OOF.npz ...]   (first OOF = reference, right-handed)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "baselines" / "tools"))
from atlas import mix  # noqa: E402
from mixlib import align_to, strict  # noqa: E402


def main() -> None:
    gt_p, out = Path(sys.argv[1]), Path(sys.argv[2])
    files = [Path(a) for a in sys.argv[3:]]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = np.load(gt_p)
    ids = g["ids"].tolist()
    members = []
    for f in files:
        with np.load(f) as z:
            assert z["ids"].tolist() == ids, f"{f} ids differ"
            members.append(z["pts"].astype(np.float32))
    names = [f.stem for f in files]
    rows = [f"{n} alone" for n in names] + [" + ".join(names[: m + 1]) for m in range(1, len(names))]
    S = np.full((len(rows), len(ids)), np.nan)
    t32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    with torch.inference_mode():
        for n in range(len(ids)):
            t = t32(g["gt"][n])
            ref = t32(members[0][n])
            aligned = [members[0][n]] + [align_to(t32(m[n]), ref).cpu().numpy() for m in members[1:]]
            for a, m in enumerate(members):
                S[a, n] = strict(t32(m[n]), t)
            for m in range(1, len(members)):
                S[len(members) + m - 1, n] = strict(t32(mix(aligned[: m + 1])), t)
            if n % 200 == 0:
                print(f"scored {n}/{len(ids)}", flush=True)
    rng = np.random.default_rng(0)

    def gain(base: np.ndarray, x: np.ndarray) -> str:
        d = base - x
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        return f"{d.mean():5.2f} (95% CI {np.quantile(boot, 0.025):.2f} .. {np.quantile(boot, 0.975):.2f})"

    print("\nCD-L2 mm^2, strict, all objects:")
    for r, name in enumerate(rows):
        line = f"{name:50s} mean {S[r].mean():6.2f} median {np.median(S[r]):6.2f}"
        if r >= len(members):
            prev = 0 if r == len(members) else r - 1
            line += f" | gain vs {names[0]} alone {gain(S[0], S[r])}, vs previous {gain(S[prev], S[r])}"
        print(line)
    np.savez(out, ids=np.array(ids), rows=np.array(rows), scores=S)


if __name__ == "__main__":
    main()

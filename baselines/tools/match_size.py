# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""New shapes at a known-good size: scale every cloud of SRC about its centre to the RMS radius of the same photo's
cloud in SIZE_REF (e.g. the best leaderboard zip), so only the shape changes.
usage: match_size.py SRC_ZIP SIZE_REF_ZIP OUT_DIR   -> OUT_DIR/preds_geometry.zip
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import pack  # noqa: E402
from rescale_zip import read_zip, rms  # noqa: E402


def main() -> None:
    src, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    a, b = read_zip(src), read_zip(ref)
    assert a.keys() == b.keys(), "the zips hold different photos"
    result, ratio = {}, []
    for k, p in a.items():
        c = p.mean(0)
        s = rms(b[k]) / rms(p)
        ratio.append(s)
        result[k] = c + (p - c) * s
    pack(out / "preds_geometry.zip", result)
    print(f"{len(result)} clouds; scale median {np.median(ratio):.3f}, 5th-95th pct "
          f"{np.quantile(ratio, 0.05):.3f}-{np.quantile(ratio, 0.95):.3f}")


if __name__ == "__main__":
    main()

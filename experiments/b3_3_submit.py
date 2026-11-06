# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Write val submissions for the best B3.3 model and the DINOv2-only B3.2 fallback."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from frozen import Config, Experiment  # noqa: E402

KNOBS = dict(n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
CLS = (0, 2, 4)
exp = Experiment()
best = exp.submit(Config(name="B3.3 v2+v3H", features=(("dinov2l_reg_518", CLS), ("dinov3_vith16plus_512", CLS)), **KNOBS),
                  "b3_3_dinov2_dinov3h")
fallback = exp.submit(Config(name="B3.2 v2 only", features=(("dinov2l_reg_518", CLS),), **KNOBS), "b3_2_dinov2_tuned")
b3 = pd.read_csv("submissions/b3_dinov2_ridge/preds_volume.csv").volume_cm3.to_numpy()
d = np.abs(np.log(best / b3))
print(f"vs submitted B3: median change {np.median(d):.3f} (ln), cluster switches {(d > np.log(30)).sum()}")

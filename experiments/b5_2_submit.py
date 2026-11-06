# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Submission: B3.4 pipeline with DINOv3 ConvNeXt-L features added (the strongest CNN variant of B5.2)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines")); sys.path.insert(0, str(ROOT / "experiments"))
import b3_4_knn as knn
from frozen import Config, Experiment

knn.KS = (2,)
knn.CFG = Config(name="B5.2 +convnext-L", features=knn.CFG.features + (("dinov3_convnext_l_512", (0, 1, 2)),),
                 n_pca=None, C=0.01, tau=0.5, r_clip=2.5)
knn.submit(Experiment(), 2, 0.25, 0.25, "b5_2_convnext_l")

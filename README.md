# Food 3D Reconstruction baseline

One image to a point cloud in millimetres. Metric: CD-L2.

* `baselines/geom.py` — pose removed, scale kept, submission zip.
* `baselines/baseline.py` — PSGN point-set model.
* `baselines/METHODS.md` — why this baseline.

```
uv run baselines/geom.py
uv run baselines/baseline.py train
uv run baselines/baseline.py predict
```

Put the challenge files in `data/f3d/` (`data/fve/` is also accepted). Train needs `mesh.ply` next to each image. Eval images live in `val/` or `eval/`. Do not commit the data or the prediction zips.

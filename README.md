# Food 3D Reconstruction baseline

One image to a point cloud in millimetres. Metric: CD-L2.

* `baselines/geom.py` — pose removed, scale kept, submission zip.
* `baselines/baseline.py` — PSGN point-set model.
* `baselines/atlas.py` — DINOv2 tokens to an AtlasNet decoder that draws a unit-size shape.
* `baselines/scale.py` — size head: RMS radius in mm, multiplied onto the shape.
* `baselines/METHODS.md` — why this baseline, and the improvements on this branch.

```
uv run baselines/geom.py
uv run baselines/baseline.py train
uv run baselines/baseline.py predict

uv run baselines/atlas.py cache       # mesh samples + DINOv2 tokens
uv run baselines/atlas.py train
uv run baselines/scale.py             # needs features/*.npz from the FVE branch, else DINOv2-B
uv run baselines/atlas.py evaluate    # local grader on the holdout
uv run baselines/atlas.py predict
```

Put the challenge files in `data/f3d/` (`data/fve/` is also accepted). Train needs `mesh.ply` next to each image. Eval images live in `val/` or `eval/`. Do not commit the data or the prediction zips.

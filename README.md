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

## Experimental TripoSG adaptation

The [fine-tuning run guide](baselines/generators/triposg/finetune/README.md) covers
Lightning-based attention LoRA, full-transformer tuning, and an independent
physical-size MLP. Read the [architecture note](docs/TRIPOSG_ARCHITECTURE.md) or
open the [interactive diagrams](docs/triposg_architecture.html) for every block's
inputs, outputs, tensor dimensions, and training/inference paths. These recipes
have CPU contract tests, but no real-model training results or measured VRAM yet.

## Future directions

Keep flow matching as the main shape objective while comparing adapter capacity,
noise sampling, and geometry-aware supervision independently. The
[detailed experiment plan](baselines/generators/triposg/finetune/README.md#future-directions)
explains the flow equations, alternative adapters, noise-level validation,
decoded-geometry checkpoint selection, and optional decoder supervision. These
are proposed experiments rather than fixes for measured failures of the new setup.

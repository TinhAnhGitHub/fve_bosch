# Bosch Food Volume Estimation (Track 3) -- team "skibidi"

Predict `volume_cm3` from one auto-framed render. Metric: sMAPE. Best public score: **37.15** (B3.4 + 5-fold LoRA DINOv3-L blend).

* `docs/PIPELINE.md` -- data processing and every stage, step by step.
* `docs/EXPERIMENTS.md` -- every experiment, what it does, input/output, CV and board scores.
* `plan/experiment_plan.md` -- planned experiments, gates and the conv-head design.
* `reports/`, `research_notes/` -- literature reports behind the choices.

## Layout
```
baselines/    B0-B3 baselines and the shared frozen-feature harness (frozen.py)
features/     feature extractors (DINOv2/DINOv3/SigLIP2/ConvNeXt/C-RADIO); cached *.npz are NOT committed
experiments/  B3.2-B5.x experiment scripts, blends, submission builders, chain scripts
kaggle_pkg/   standalone fine-tuning trainer (b4_2_train.py), Kaggle launcher/kernel, prep data
eda/          label analytics and duplicate check
submissions/  built preds_volume.zip files (validated)
data/         NOT committed (challenge data must not be redistributed)
```

## Reproduce the best submission (outline)
1. Data in `data/fve/` (train with `mesh.ply`, val PNGs). `uv` manages dependencies per script (PEP 723 headers), `uv run <script>`.
2. Frozen features: `features/extract_dinov2.py` (DINOv2-L-reg@518 and DINOv3-H+/16@512, bf16).
3. `experiments/b3_4_knn.py` builds the B3.4 submission (`submissions/b3_4_knn`).
4. Fine-tune B4.0 per fold: `experiments/b4_0_chain.sh 0 1 2 3 4` (about 1 hour per fold on an RTX A3000), then
   `uv run experiments/b4_0_blend_submit.py 0.5 0,1,2,3,4 b4_0_blend5_w05`.

Scripts resolve paths from the repo root, so `data/fve/`, `features/*.npz` and `experiments/out_*` must sit inside this folder (links
are fine; all are git-ignored). Built zips and prediction CSVs in `submissions/` are git-ignored too.

## Secrets and licences
Tokens live in `.env` (never committed). DINOv3 and C-RADIOv4-H weights are downloaded from Hugging Face under their own licences
(DINOv3 is gated; keep the DINOv2 fallback for the code hand-over). Do not publish the challenge data.

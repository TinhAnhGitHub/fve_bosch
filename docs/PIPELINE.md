# Pipeline: data processing and stages

Task: predict `volume_cm3` of a food object from ONE auto-framed 1024x1024 Blender render (`rgb_industrial.png`). No camera
intrinsics and no scale cue; the category is known from the folder path. Metric: sMAPE per object,
`2 * tanh(|ln(pred / true)| / 2)` (symmetric in log space, bounded at 200%). Train 2,042 objects (164 categories), Public Eval 511
(local folder `data/fve/val`), Private Test 851.

## 0. Raw data (`data/fve/`, never committed)
```
train/<group>/<category>/<sample_id>/{rgb_industrial.png, gt.json, mesh.ply}
val/<group>/<category>/<sample_id>/rgb_industrial.png          # labels hidden
```
Label = `gt.json["volume"]["primary_cm3"]` (the enclosed mesh volume). `gt.json` also has `bbox_mm`. Sources: OmniObject3D
(`omni_*`) and MetaFood3D (`mf3d_*`).

Findings that shape everything downstream (see `eda/`):
* The camera auto-frames each object, so apparent size in the frame has ~0 correlation with real size. Scale must come from the
  category / shape priors, not from pixel area.
* About 10% of OmniObject3D labels are 200-1000x too large (mesh unit errors). They are official ground truth, so they are kept.
  Inside a category the volumes split into two clusters ("normal" and "oversized"). Picking the wrong cluster costs ~199% on that
  object, which makes the mode decision the dominant error source.
* No true duplicates between train and val (`eda/b3_1_duplicates.py`), but about a third of val has a near look-alike in train.

## 1. Targets and clusters (`baselines/b3.py`)
1. `ln` = ln(volume). `GAP` = ln 30.
2. `cluster_labels(T)`: `u=1` (oversized) for objects above a >=30x gap inside their category, plus whole categories >=30x above their
   level-1 group median.
3. `build_table(T, u)`: per (category, cluster) constant = the sMAPE-argmin of the ln volumes (`argmin_t`), plus a leave-one-out
   reference `ref_loo` for every training row. Fallback = global argmin for unseen categories.
4. Residual target `r = ln(true) - ref_loo`. Rows with `|r| >= ln 30` are excluded from regression (`ok` mask) and r is clipped to
   +-2.5 at prediction time.

Everything in 1-4 is rebuilt from the TRAINING part of each fold only, so held-out scores are clean.

## 2. Validation protocol
* 5-fold `StratifiedKFold` on category. Seeds 0-2 are for selection, seeds 3-5 are fresh checks. Fine-tuned models use seed 0 splits.
* Adopt a change only if it gains >= 0.5 CV on seeds 0-2, holds on fresh seeds 3-5 and (for submissions) beats the previous best.
* Single-fold gaps below ~2 points are noise. Board scores are on 511 objects: one wrong cluster call is about 0.4 points.
* The board is for confirming, never for tuning weights. 5 submissions per day.

## 3. Frozen-feature track (B3.x) -- `features/`, `baselines/frozen.py`
1. Extract frozen features once and cache them in `features/*.npz` (`extract_dinov2.py`, `extract_siglip2.py`,
   `extract_convnext.py`, `extract_radio.py`). Layout: 6 equal blocks per file (CLS-like / mean-patch for 3 layers), each image
   averaged with its horizontal flip. DINOv2 needs fp16/bf16 as noted per script; DINOv3 must run in bf16 (fp16 gives NaN).
2. `frozen.Experiment` loads the features in train/val order, builds splits, standardises (no PCA in the best config).
3. `fit_predict` (stages fitted on the fold's training rows):
   * Ridge (`RidgeCV`) on residual r (rows with `ok`).
   * Logistic regression `P(oversized)` for categories that have both clusters (C=0.01).
4. Within-category kNN (`experiments/b3_4_knn.py`): top-2 cosine neighbours on standardised features, restricted to the same
   category in the training fold. They give `p_knn` (similarity-weighted oversized rate) and `r_knn` (neighbour residual in the chosen
   cluster). Blend `r = 0.25*ridge + 0.75*kNN`, `p = 0.25*clf + 0.75*kNN`.
5. Decode: oversized cluster if `p > 0.5`, ln prediction = cluster constant + clipped r.
6. Submission (`b3_4_knn.submit`): refit on ALL train, predict val, write the zip.

Why the board beats CV for this track: at submission time the kNN bank is all 2,042 train objects (80% in CV).

## 4. Fine-tuning track (B4.x)
Data prep for fine-tuning (`experiments/b4_2_prep.py` -> `kaggle_pkg/b42_prep/`):
* `rows.csv`: train then val rows (group, category, sample_id) so a trainer without the repo can find the images.
* `fold{f}.npz`: `tr_idx`, `va_idx` (seed-0 split), per-training-row targets `r`, `ok`, oversized label `u`, `mixed` (category has
  both clusters), standardised `aux` (3 sorted ln bbox dims in cm, ln fill = volume / bbox volume, and 4 scale-free mesh descriptors
  from `experiments/mesh_targets.py`: ln convexity, ln compactness, two ln PCA eigenvalue ratios), plus decode tables
  `q_base0/q_base1` (cluster constants for held-out rows) and the same for val rows (all-train table).
* Mesh descriptors are computed from `mesh.ply` locally. The meshes never leave the machine and are never an input at test time;
  they only provide training-time auxiliary targets.

Models:
* **B4.0** (`b4_0_finetune.py`): DINOv3 ViT-L/16 @512, LoRA r16 on q/k/v (bf16 base, grad checkpointing), head on [CLS, mean-patch],
  outputs r, oversized logit, 4 aux. Loss = smooth-L1(r, beta .3, masked) + 0.5*BCE(mode, mixed categories) + 0.1*L1(aux). AdamW,
  LoRA lr 2e-4, head lr 1e-3, wd 0.05, cosine + 1 epoch warm-up, 10 epochs. Inference averages image and flip.
* **B4.1 / B4.2 / B4.3** (`kaggle_pkg/b4_2_train.py`, one trainer, `--backbone dinov3|radio`): MoGe-2-style conv head. Patch
  tokens from 4 depths -> per-layer 1x1 conv -> sum -> + UV coordinate channels -> residual conv blocks, stride-2 conv -> attention
  pooling (4 queries) -> concatenated with the backbone summary -> MLP -> [r, oversized logit, 8 aux]. EMA weights, head warm-up
  (LoRA lr 0 for 2 epochs), optional drop-path, mode label smoothing and scale jitter.
* Decode for fine-tuned models: cluster constant (+ clipped r), oversized if sigmoid(logit) > 0.5.

## 5. Blending (`experiments/b4_0_blend.py`, `b4_0_blend_submit.py`, `b4_2_blend.py`)
In ln space: `w*ln_B4 + (1-w)*ln_B34` where both models agree on the cluster (|diff| < ln 30); where they disagree the B3.4 call is
kept (never hedge between modes). Out-of-fold check pools all folds that have held-out outputs and uses a paired
category-clustered bootstrap. Weights come from OOF (flat between 0.4 and 0.6), not from the board.

## 6. Submission format
`preds_volume.zip` containing `volume/{sample_id}.json` = `{"volume_cm3": <number>}`; 511 files for Public Eval. Missing or invalid
predictions score 200%. Writer: `baselines/b0_b1.write_submission` (validates the zip).

## 7. Compute
* Local: WSL2, RTX A3000 laptop 6 GB, 17 GB RAM. Parallel trainings on one GPU are slower overall; run one at a time.
* Kaggle: 2xT4 (`--accelerator NvidiaTeslaT4`); P100 is retired. T4 has no bf16, so C-RADIO runs in fp16 with loss scaling
  (works), DINOv3 needs bf16 (run locally). One fold per GPU, launched by `kaggle_pkg/kernel/launcher.py`.

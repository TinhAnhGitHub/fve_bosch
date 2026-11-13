# Food 3D Reconstruction: full technical report

Bosch AI Challenge, Food 3D Reconstruction (F3DR). Status as of 2026-11-12, about 21:00. New since 2026-11-10: cloud training runs (section 9.7), handedness-aware mixes and the new best 43.85 (section 9.8), pretrained 3D generators (section 9.9), the new plan (section 17), and the fine-tuned TripoSG shape model with its size analysis (section 18).

This report describes the input and output format, data processing, model architecture, training, evaluation and every experiment so far. It is meant to be read by someone who has not seen the project and is asked to suggest improvements. It contains aggregate statistics only: no dataset images, meshes or point clouds.

## 1. Where things stand

| Item | Value |
|---|---|
| Best leaderboard score | **43.8453** (`v28_e5_v22plus_rh2`, RH-Mix5: `v22`'s three models + two right-handed models, each aligned to a right-handed reference before mixing; uploaded 2026-11-10 18:10) |
| Competitor | 46.88. RH-Mix5 is 3.0 below it |
| Best single model, leaderboard | **52.89** (`v27`, positional embedding + right-handed targets, seed 1); `v33` no category input 54.01 |
| Best single model, local | 33.5 (`s_l1_s0`, L1 Chamfer), but it scored 59.76 on the leaderboard. Local scores do not rank single models (section 8.2) |
| Most promising new direction | SAM 3D shapes at the true size: about 23 against about 49 for our model on 92 holdout objects (section 9.9). Size is the missing piece |
| New (2026-11-12): fine-tuned TripoSG | LoRA adapter on our 1,838 training objects: shape at the **true** size 24.5 (median 13.1) against 38.9 (17.4) for our best single model on the same 204 holdout objects. With our own size head it scores 79.3, because that size head is no better than the median size of the food category. With the size taken from the 2 most alike training photos of the same food type it scores 60.9 (section 18.10): better, but far from RH-Mix5, which scores 27.4 on this holdout (30.7 with the report's strict scorer; 43.85 on the leaderboard; section 18.11). The look-alike version was uploaded by the user on 2026-11-13 and scored **102.82** on the leaderboard (RH-Mix5: 43.85), 1.69 times its holdout mean, so a standalone LoRA is not a route to a better score (sections 18.7 and 18.10) |
| Upload next | `v35_e6_mix5_nocat` (RH-Mix5 + no-category model) and `v34_nocat_depth_s0_ema`, both built and checked, not uploaded |
| Running | IDOL generator on Kaggle (account 3): 204 holdout, then all 2,042 training photos |

Lower is better everywhere. Scores are squared millimetres (mm²).

The main diagnosis so far:

- **The model overfits heavily.** Final training loss is about 8.6 while holdout loss is about 35, a factor of four. The training set is only 1,838 objects.
- **Averaging helps more than regularization.** EMA and point mixes reliably reduced error. DropPath, attention dropout and grid jitter did not.
- **Architecture changes can still help.** Sheet-level attention is the first design change on the corrected data to clearly beat the plain sheet model.
- **Object size is a large share of the remaining error.** Picking the best size per object with knowledge of the answer would cut the best model from 34.3 to 19.1 (section 11.2). The average size is already right. The model just cannot tell each object's real size precisely, and the camera auto-framing hides it.
- **The decoder design looks close to its limit.** Overnight, none of seven single changes beat sheet attention alone (section 9.6). Most landed within seed noise. The gains came from mixing in deep-supervision models.
- **Weak mix members hurt on the leaderboard even when the local score improves.** `v23` is `v22` plus the two weakest models (41.4 and 42.1 alone). Locally it was 0.3 better; on the leaderboard it was 1.9 worse (48.28 against 46.35). The members were picked on the same 204 objects that scored them, so the local gain was selection noise. Mix only models that are strong on their own.
- **New (2026-11-11): every single-model change now lands within seed noise on the leaderboard (52.9–60.1).** Positional embedding, right-handed targets, L1 Chamfer, a size head, a coverage loss, token noise, 384 px, no category and depth tokens were all tried (section 9.7).
- **New: pretrained generators draw much better shapes than our decoder, but not the size.** SAM 3D at the true size is about half our model's error; at our predicted size it is twice our error (section 9.9). The plan is to use a generator for shape and train a small model for size (section 17).
- **New: our models hedge size by piling points into dense clumps.** The model that clumps most (v27, 43% of points in clumps) scored best on the leaderboard; L1 Chamfer removed the clumps and scored worst. Squared Chamfer rewards the hedge (section 9.9).

## 2. Competition task

- **Task:** one RGB image of one food object in, a 3D point cloud in millimetres out.
- **Metric:** CD-L2, bidirectional squared Chamfer distance, in mm².
- **Alignment:** the grader removes rotation and translation before scoring. Scale is **not** removed, so absolute size matters.
- **Ground truth:** the grader samples 10,000 points from the true mesh surface. The predicted cloud is used as submitted.
- **Failure penalty:** a missing sample scores twice the worst valid value.
- **Splits:** Train 2,042 objects, Public Eval 511 (live leaderboard), Private Test 851 (decides the final ranking, separate folder, released later, images only).
- **Timeline and rules:** ends 2027-01-01. Finalists hand over code and documents, and the organisers re-run the model, so the score has to reproduce. Hand-labelling test data is banned. Private code sharing outside the team is forbidden. Public sharing has to reach all participants through Bosch Connect. The dataset cannot be redistributed.
- **Upload limit:** 15 per day (stated by the user; the rules text still says 5).

## 3. Data

### 3.1 Folder layout

```
data/f3d/
  train/<group>/<category>/<sample_id>/
      rgb_industrial.png   1024 x 1024 RGB
      gt.json              metadata and labels
      mesh.ply             ground-truth mesh, binary PLY, in mm
  eval/<group>/<category>/<sample_id>/
      rgb_industrial.png   image only
```

The category and group are visible in the folder path for evaluation objects too. The model uses the category as an input (section 5.2). The rules do not explicitly address whether this is allowed.

### 3.2 `gt.json` fields

| Field | Meaning | Used by the model? |
|---|---|---|
| `sample_id`, `timepoint`, `source` | Identifier, capture time, origin dataset | No |
| `category`, `category_l1`, `group` | Food labels | `category` only, taken from the folder name |
| `reference_object` | Checkerboard card: square size in mm, inner corners, card size, whether it lies on the support plane, and an image `rgb_industrial_ref.png` | **No.** That image is not in our download |
| `volume` | Volume labels and protocol | No |
| `weight_g` | Weight, not available for every object | No |
| `freshness` | Freshness grade | No |
| `bbox_mm` | Mesh bounding box | No |
| `mesh` | Face and vertex counts, decimation information | No |

### 3.3 Dataset statistics (corrected data)

| Quantity | Train | Public Eval |
|---|---|---|
| Objects | 2,042 | 511 |
| Categories | 164 | 161, all also in train |
| OmniObject3D (`omni_`) | 1,634 | 409 |
| MetaFood3D (`mf3d_`) | 408 | 102 |

| Group | Train | Eval |
|---|---|---|
| fruits | 504 | 109 |
| vegetables | 484 | 103 |
| extra | 403 | 98 |
| bakery | 391 | 138 |
| packaged | 109 | 21 |
| seafood | 78 | 19 |
| meat | 73 | 23 |

- **Objects per training category:** minimum 1, median 8, maximum 65. Six categories have a single object.
- **RMS radius of the cleaned training clouds, in mm:** 1st percentile 7.7, 5th 10.9, 25th 21.8, **median 30.5**, 75th 41.6, 95th 60.6, 99th 82.3, maximum 108.5.
- **Mean squared radius:** 1,329 mm². Median radius is 28.7 mm for OmniObject3D and 38.4 mm for MetaFood3D.
- **Why size matters:** Chamfer error scales with the square of object size, so large objects dominate the mean score.

### 3.4 Properties that shape the problem

- **The camera auto-frames each object.** Real size has about zero correlation with how large the object appears in the frame, so scale has to come from what the object is and how it looks. No camera intrinsics or depth are provided.
- **Mesh pose carries no information.** Training meshes have no meaningful orientation relative to the image, so the model predicts in a canonical frame (section 4.3).

### 3.5 The label correction

The original release had about 10% of OmniObject3D meshes with wrong units.

The organisers released corrected data on 2026-11-09:

- 267 training meshes changed and all images stayed identical.
- 174 were objects we had flagged as oversized. Their median radius went from 143.7 mm to 14.4 mm.
- The other 93 had looked normal to us and shrank by exactly ×0.10.
- The evaluation answers appear to have been corrected too: our first model trained on fixed data moved the leaderboard from 173.09 to 68.54.

Scores before and after this date are on different scales and cannot be compared.

## 4. Input and output, and data processing

### 4.1 Tensor contract

| Stage | Shape | Notes |
|---|---|---|
| Original image | 1024 × 1024 × 3 | PNG |
| Encoder input | 256 × 256 × 3 | Bicubic resize, ImageNet normalization |
| Image tokens | 257 × 1024 | Class token + 256 patch tokens (16 × 16 grid), register tokens dropped |
| Category | 1 integer | Index into 164 categories from the folder name |
| Model output | 8192 × 3 | Multiplied by 40 to give millimetres |
| Target during training | 8192 × 3 | Sampled mesh surface, canonical frame, mm, divided by 40 |

### 4.2 Image processing

1. Load `rgb_industrial.png` as RGB.
2. Resize to 256 × 256 with bicubic filtering.
3. Scale to 0–1 and normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
4. Run frozen DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) from Hugging Face.
5. Keep the class token and the 256 patch tokens from the last layer. Drop the register tokens.
6. Store as float16 in `train_dinov3l.npz` with shape `(2042, 257, 1024)`.

**No image augmentation is used.** Each object has exactly one token set, computed once.

### 4.3 Target processing

1. Load `mesh.ply` with trimesh, without processing.
2. Sample 8,192 points uniformly on the surface once. These same points are used in every epoch.
3. Canonicalize: subtract the centroid and rotate onto the principal axes with SVD (PCA). **Do not rescale.**
4. Fix each axis sign deterministically so the third moment (skewness) along that axis is positive.
5. Divide by a constant 40 mm during training (about the typical radius) so the network output is near unit scale.

Known weakness: on a nearly symmetric axis the sign can flip between similar objects, because the skewness is close to zero. Under `--no-flip-loss` the loss does not forgive that flip.

### 4.4 Split

- A fixed 10% holdout from `numpy.random.default_rng(0).permutation`, giving **204 holdout and 1,838 training objects**.
- The holdout has 166 OmniObject3D and 38 MetaFood3D objects, median radius 29.1 mm.
- Different seeds change initialization and shuffling, not the split.
- The final zips are trained on the 1,838 objects only. The 204 holdout objects have not been used for training.

### 4.5 Submission format

```
preds_geometry.zip
  geometry/<sample_id>.ply
```

Each file is an ASCII PLY with vertex properties `float x, y, z`, five decimals, and **exactly 8,192 points**. An earlier zip with 8,356 points in some clouds failed all of its samples with "missing prediction". The zip builder now enforces the count.

## 5. Model architecture (current best: `talk`)

### 5.1 Overview

The approach is AtlasNet-style. Each output point starts as a position on one of several 2D sheets, and a decoder moves it to its place in 3D.

```
image 1024 x 1024
   | resize 256, DINOv3 ViT-L/16 (frozen, cached)
   v
tokens (B, 257, 1024) --LayerNorm, Linear--> (B, 257, 256) = t

for every point on every sheet (8 sheets x 1024 points):
   category code (32) + sheet code (16) + uv (2)  = 50 values
   | Linear 50 -> 256
   v
points x (B, 8192, 256)
   | Block 1 (reads t)
   | Block 2 (reads t)
   | Block 3 (reads t)
   | Block 4 (reads t)
   | LayerNorm, Linear 256 -> 3   (32-bit float)
   v
points (B, 8192, 3) x 40 = millimetres
```

### 5.2 Point queries

Each output point gets a query vector of 50 values:

| Part | Size | Source |
|---|---|---|
| Category code | 32 | `nn.Embedding(164, 32)`, same for all points of an object |
| Sheet code | 16 | `nn.Parameter(randn(8, 16) * 0.02)`, one learned vector per sheet |
| uv | 2 | Fixed grid, `linspace(-1, 1, 32)` in both directions |

- Eight sheets of 32 × 32 points give 8 × 1,024 = 8,192 points.
- A `Linear(50, 256)` projects each query to the working width.
- An unseen category falls back to index 0, which is a real category rather than a dedicated "unknown" token. This has not been a problem on Public Eval (no unseen categories). The private set may differ.

### 5.3 One decoder block (`TalkBlock`)

There are four blocks, each width 256, in this order:

```
x: (B, 8192, 256)    t: (B, 257, 256)

1. Cross-attention to the image, residual, 8 heads
      q = Linear(LayerNorm(x)); k, v = Linear_256_to_512(t)
      x = x + proj(attention(q, k, v))

2. Per-sheet offset, residual
      s = mean of x over each sheet            (B, 8, 256)
      x = x + gate * (s - mean of s over sheets), copied to all points on the sheet
      gate: a learned 256-vector, starts at zero

3. Sheet-level attention, residual   (new in `talk`)
      summary = Linear_256_to_64(LayerNorm(mean of x over each sheet))   (B, 8, 64)
      said = self-attention(summary), 4 heads
      x = x + Linear_64_to_256(said), copied to all points on the sheet
      the last Linear starts at zero

4. MLP, residual
      x = x + MLP(LayerNorm(x))     MLP: 256 -> 512 -> GELU -> 256

5. Neighbour filtering along each sheet, residual
      reshape each sheet to a 32 x 32 image with 256 channels
      x = x + depthwise 3 x 3 Conv2d (groups = 256), starts at zero
```

Because the extra parts start at zero, a new block initially behaves like the simpler one and only learns to use them if they help.

### 5.4 Size and parameter count

| Part | Parameters |
|---|---|
| Category embedding | 5,248 |
| Sheet codes | 128 |
| Query projection | 13,056 |
| Token LayerNorm + projection | 264,448 |
| Output head | 1,283 |
| Each cross-attention + MLP (Block base) | 527,104 |
| Sheet additions per block (gate + conv) | 2,816 |
| Talk additions per block | 50,240 |
| **`deep` total** | 2.39 M |
| **`sheet` total** | 2.40 M |
| **`talk` total** | **2.60 M** |

### 5.5 Design choices

- **Frozen encoder.** Token extraction is done once, so an epoch only runs the decoder. Fine-tuning DINOv3 has not been tried on this task.
- **No point-to-point self-attention.** 8,192 queries would be too expensive on this GPU. Points interact only through the sheet average, sheet-level attention and the 3 × 3 filter.
- **Mixed precision.** Blocks run in bf16 autocast. The output head runs in 32-bit float, because millimetre targets need the precision.
- **Global unit of 40 mm.** Predictions are in units of 40 mm and multiplied back at the end. The model learns shape and absolute size together, end to end.

### 5.6 Variants tried

| Name | What it adds |
|---|---|
| `atlas` | Original: one single-head cross-attention, two-layer MLP, no residual. Output directly in mm |
| `deep` | Four residual cross-attention blocks. Trains on a random half of the queries per step |
| `sheet` | `deep` + per-sheet offset + 3 × 3 filter along each sheet. Uses all queries |
| `talk` | `sheet` + attention between the 8 sheet summaries in every block |

## 6. Training

### 6.1 Loss

Squared Chamfer distance over all 8,192 predicted and 8,192 target points:

```
to_gt   = mean over predicted points of |p - nearest target|^2
to_pred = mean over target points of |t - nearest predicted|^2
loss    = to_gt + to_pred
```

- Nearest neighbours are exact. They are found in a chunked pass without gradients, then the loss is computed with gradients on those matches.
- **`--no-flip-loss` is on for all current models.** The older loss took the best of four PCA sign flips, which allowed arbitrary orientations.
- The loss is computed in the canonical frame without further alignment.

### 6.2 Optimizer and schedule

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | Peak 3e-4 |
| Schedule | OneCycle over 18,400 steps, 5% warm-up, cosine decay (PyTorch defaults: starts at 1.2e-5, ends about 1.2e-9) |
| Batch size | 4 |
| Epochs | 40 |
| Gradient clipping | None |
| Precision | bf16 in blocks, 32-bit float head |
| Hardware | NVIDIA RTX A3000 Laptop GPU, 6 GB |
| Time | About 50–60 minutes per 40-epoch run |

### 6.3 Weight averaging (EMA)

A second copy of the decoder tracks a running average of the weights:

```
d = min(0.999, (1 + step) / (10 + step))
shadow = d * shadow + (1 - d) * weights       after every step
```

Normal and EMA weights are checked on the holdout separately after each epoch, and each keeps its own best epoch. They are saved as `<name>.pt` and `<name>_ema.pt`.

### 6.4 Best-epoch selection

The best epoch is chosen on the 204-object holdout using the training loss, without any alignment. Typical best epochs are 22–29 for normal weights and 20–21 for EMA, out of 40. Holdout error rises slightly after that while training loss keeps falling. On `pl_t_s0`, for example:

| Epoch | Train | Holdout (normal) | Holdout (EMA) |
|---|---|---|---|
| 1 | 650.1 | 331.8 | 210.4 |
| 5 | 80.4 | 67.8 | 65.2 |
| 10 | 41.6 | 47.1 | 42.8 |
| 15 | 31.1 | 53.5 | 37.6 |
| 20 | 21.8 | 40.8 | 35.5 |
| 25 | 14.6 | 38.6 | 35.3 |
| 30 | 10.7 | 38.3 | 36.6 |
| 35 | 9.0 | 38.2 | 37.9 |
| 40 | 8.6 | 38.6 | 38.5 |

Training loss falls by a factor of 2.6 between epochs 20 and 40. Holdout error does not improve, which is the overfitting described in section 1.

### 6.5 Available training options

| Flag | Effect | Status |
|---|---|---|
| `--ema 0.999` | Weight averaging | **In use** |
| `--no-flip-loss` | No sign-flip search in the loss | **In use** |
| `--deep-sup 0.2` | Extra Chamfer loss on the first three blocks' outputs, through the shared output head, each weighted 0.2 | Training now |
| `--drop-path`, `--attn-drop` | DropPath (rises from 0 at the first block) and attention dropout | Did not help |
| `--grid-jitter` | Random noise on uv, in grid steps | Did not help |
| `--repulsion` | Hinge loss against clumped points | Implemented, not run after the plan changed |
| `--fourier` | uv + r² + low sin/cos harmonics | Did not stack on `sheet` |
| `--resample` | Fresh mesh samples each step | Did not help |
| `--token-drop` | Hide a share of the patch tokens each step | Did not help on `sheet` |
| `--all-queries` | `deep` on all queries | Did not help |
| `--drop-oversized`, `--reweight-oversized` | Handle the old bad labels | Obsolete after the fix |

## 7. Inference and ensembling

### 7.1 Single model

1. Resize and normalize the image, and run DINOv3 once.
2. Get the category from the folder name.
3. Run the decoder and multiply by 40.
4. Write 8,192 points to `geometry/<sample_id>.ply`.

There is no post-processing, test-time augmentation or size correction.

### 7.2 Point mix

Each model predicts its own 8,192 points. The mix takes an equal share of each:

```
mix = concatenate( cloud_0[0::k], cloud_1[1::k], ..., cloud_{k-1}[k-1::k] )[:8192]
```

No alignment is done between models, because they all predict in the same canonical frame. All members use `--no-flip-loss`, and earlier flip-invariant models needed alignment first.

**Other merge methods tested.** On the same five models (`pl_r_s0_ema`, `pl_r_s1_ema`, `tn_base_s1`, `pl_t_s0_ema`, `pl_e_s0_ema`):

| Merge method | Local strict | Stray | Missing |
|---|---|---|---|
| **Index stride (current)** | **31.2** | 14.9 | 16.3 |
| Equal share, random points | 31.9 | 14.9 | 17.0 |
| Drop 5% most-disputed points, then random | 38.7 | 12.3 | 26.4 |
| Farthest-point sampling of all pooled points | 40.4 | 32.7 | 7.8 |
| Drop 5% most-disputed points, then farthest-point | 39.2 | 23.2 | 16.0 |

Index stride is the best of these. Farthest-point sampling covers the surface well but selects too many misplaced points. Pruning removes real surface points that models disagree on.

**Stripes with an even number of models (found 2026-11-10).** Each sheet is a 32 × 32 grid stored row by row. When the number of models k divides 32 (2, 4, 8), `cloud[m::k]` keeps whole grid columns for each model, so each model contributes stripes with gaps. A staggered assignment (`(column + row + 3·sheet) mod k`) avoids this. On the same members:

| Models | Index stride | Staggered |
|---|---|---|
| 2 | 31.42 | **30.91** |
| 3 | **30.10** | 30.24 |
| 4 | 30.97 | **30.08** |
| 5 | **29.76** | 29.83 |
| 6 | **29.39** | 29.87 |
| 7 | **29.55** | 29.81 |

Staggering gains 0.5 with 2 models and 0.9 with 4, and nothing with odd counts. Rule: use an odd number of models (3, 5, 7), and no code change is needed. Differences of about 0.5 between assignments are noise. The older even mixes `v17` (2 models) and `v20` (4 models) carried this penalty.

## 8. Evaluation method

### 8.1 Three scores

| Name | How it is computed | Use |
|---|---|---|
| **Training check** | Chamfer on the 204 holdout objects, canonical frame, no alignment | Picks the best epoch |
| **Local strict** | Holdout, centre the prediction on the target's centroid, 10 iterations of point-to-point ICP from the predicted orientation, then Chamfer over all points | Compares models |
| **Leaderboard** | Grader on the 511 Public Eval objects | Real test |

Two components are always reported:

- **Stray:** mean squared distance from each predicted point to the nearest true point. High when points are misplaced.
- **Missing:** mean squared distance from each true point to the nearest predicted point. High when parts of the object are not covered.

### 8.2 How far local scores can be trusted

Local strict against leaderboard, corrected data:

| Zip | Local strict | Leaderboard | Ratio |
|---|---|---|---|
| `v22` | 30.1 | **46.35** | 1.54 |
| `v23` | 29.8 | 48.28 | 1.62 |
| `v19` | 32.2 | 50.76 | 1.58 |
| `v17` | 35.4 | 54.53 | 1.54 |
| `v21` | 34.3 | 55.74 | 1.63 |
| `v15` | 38.8 | 59.23 | 1.53 |
| `v16` | 40.2 | 66.84 | 1.66 |
| `v9` | 42.1 | 68.54 | 1.63 |
| `v14` | 45.8 | 78.55 | 1.72 |
| `v10` | 47.9 | **60.22** | 1.26 |

The local score orders five of these six single models correctly, with `v10` as the outlier. Across mixes, one local point has been worth about 1.2 leaderboard points, but a local difference of 0.3 between mixes is noise: `v23` was 0.3 better than `v22` locally and 1.9 worse on the leaderboard. We do not know why the leaderboard is about 1.5–1.7 times higher than local scores. The eval set may be harder, or the grader's alignment may differ from our ICP.

Known limits of the local score:

- **Seed noise is about 5–6 mm².** Same recipe, different seed: `fx_sheet` 42.1 against `tn_base_s1` 47.9, and that ordering reversed on the leaderboard.
- **Only 204 objects.** A bootstrap over objects (2,000 resamples) gives a confidence figure, but it does not cover training randomness.
- **Selection bias.** The same 204 objects choose the best epoch and compare the models, so local scores are slightly optimistic.
- **Training check and strict score can disagree.** `pl_el_s0_ema` has the best training check (34.1) but scores 41.4 strict, while `pl_t_s0_ema` is 35.1 and 34.3.
- **Different ground-truth sampling.** The local score uses 8,192 target points, the grader 10,000.

## 9. All experiments, in order

### 9.1 Before the label correction

Scores in this section are on the old labels and are not comparable with later sections. The local score here is strict Chamfer on normal objects only.

| Step | Change | Local | Leaderboard | Finding |
|---|---|---|---|---|
| Baseline | DINOv2 + AtlasNet | | 733 | Starting point |
| Reference | Teammate's `atlas3` | | 724.73 | Comparison point for early work |
| `v2` | Three-model mix | 175.8 | 761.74 | Worse than a single model on the leaderboard |
| Unit shape × size head | Separate shape and size | ≥566 (mean, old scale) | | Always loses to training in mm, which learns to spread points over uncertain sizes |
| `v3` | DINOv3-L/16 at 256 px | 90.6 | 403.43 | Large gain from the encoder |
| `v3b` | Original + DINOv3 mix | 124.2 | 551.31 | Mixing in a weaker model hurt |
| `v5` | Deep decoder, no oversized objects | 58.9 | 213.05 | Residual cross-attention blocks help a lot |
| `v6` | 256 px + 384 px deep mix | | 220.95 | No gain from flip-aligned mix |
| `v7` | No flip loss + Fourier uv | 49.0 | Not uploaded | |
| `v8` | Sheet decoder | 48.1 | 173.09 | Best on old labels |

Decoder experiments on the old labels. Base was the deep decoder with no flip loss, local 54.9. Pass rule: lower in at least 90% of bootstrap resamples.

| Run | Change | Local | Lower than base in | Result |
|---|---|---|---|---|
| `q1_fourier` | Fourier uv, harmonics 1–3 | 49.0 | 91.3% | Weak pass |
| `q2_resample` | Fresh surface samples | 55.9 | 39.9% | Fail |
| `q3_sheet` | Sheet decoder | 48.1 | 97.5% | Pass, partly a lucky seed |
| Rotation | Search 25 rotations | | | Dropped, gain only 1.5 |
| `q5_tokdrop` | Token dropout 0.15 | 52.9 | 74.1% | Fail |
| `q6_allq` | All queries on `deep` | 55.8 | 35.9% | Fail, so the sheet gain comes from sheet communication |
| `q7_reweight` | Cap oversized loss | 60.6 | 9.5% | Fail on normal objects |
| `s1_sheet_fourier` | Sheet + Fourier | 49.4 | 31.4% | Fourier adds nothing on top |
| `s4_sheet_tok` | Sheet + token dropout | 59.2 | 0.2% | Fails badly |
| `s5_sheet_cap` | Sheet + cap | 61.5 | 0.4% | Fails |
| `s6_sheet_seed1` | Sheet, seed 1 | 53.1 | 14.2% | Showed seed noise of about 5 |

Also found: flipping analysis showed the no-flip gain was all on asymmetric objects (54.5 to 46.3). Near-symmetric objects did not improve (64.7 to 65.9), and about 80 such objects could be worth about 4.5 points with a consistent orientation. No method for that exists yet.

### 9.2 On corrected data: single models

All use DINOv3-L/16, `sheet` or `talk`, `--no-flip-loss`, all 2,042 objects, 40 epochs, seed 0 unless noted. Scores are local strict on all 204 objects.

| Run | Setup | Training check (normal / EMA) | Strict normal | Stray / missing | Strict EMA | Stray / missing |
|---|---|---|---|---|---|---|
| `fx_sheet` | Sheet, seed 0 | 37.0 | 42.1 | | | |
| `tn_base_s1` | Sheet, seed 1 | 35.3 | 47.9 | 12.5 / 35.4 | | |
| `pl_r_s0` | Heavy regularization, EMA | 39.1 / 39.5 | 45.8 | 12.1 / 33.7 | 38.8 | 15.2 / 23.5 |
| `pl_r_s1` | Heavy regularization, EMA, seed 1 | 37.6 / 38.5 | 40.0 | | 40.2 | 15.5 / 24.7 |
| `pl_e_s0` | Plain + EMA | 37.3 / 36.5 | 39.8 | 12.9 / 26.9 | 37.8 | 14.0 / 23.8 |
| `pl_el_s0` | Light regularization + EMA | 34.9 / 34.1 | 48.6 | 13.0 / 35.6 | 41.4 | 14.0 / 27.3 |
| **`pl_t_s0`** | **Sheet attention + EMA** | 35.8 / 35.1 | 37.1 | 11.7 / 25.4 | **34.3** | 14.5 / 19.8 |
| `pl_ds_s0` | Sheet + EMA + deep supervision 0.2 | 39.0 / 36.5 | 40.1 | 15.1 / 25.0 | 37.5 | 16.5 / 21.0 |
| `pl_tds_s0` | Sheet attention + EMA + deep supervision 0.2 | 36.0 / 35.2 | 35.6 | 12.7 / 22.9 | 34.2 | 13.7 / 20.5 |

Overnight runs on top of sheet attention are in section 9.6.

- Heavy regularization: `--drop-path 0.15 --attn-drop 0.1 --grid-jitter 0.5`.
- Light regularization: `--drop-path 0.05 --attn-drop 0.05 --grid-jitter 0.25`.

Observations:

- **EMA shifts stray error into less missing error.** It always raises stray a little and lowers missing by more, with a net gain every time except `pl_r_s1` (40.0 to 40.2).
- **Against `fx_sheet`:** `pl_t_s0_ema` is lower in 99.9% of bootstrap resamples (mean −7.8, 5–95% −13.2 to −3.1). `pl_e_s0_ema` is lower in 95.7%.
- **Against each other:** `pl_t_s0_ema` is 3.5 below `pl_e_s0_ema`, within seed noise on one seed each.
- **The missing error is the larger component.** That is the main thing the model still has to fix.

### 9.3 On corrected data: mixes

| Mix | Members | Local strict | Stray / missing | Leaderboard |
|---|---|---|---|---|
| `v17` | `pl_r_s0_ema` + `pl_r_s1_ema` | 35.4 | | 54.53 |
| `v18` | `pl_r_s0_ema` + `tn_base_s1` | 35.9 | | Built, no score |
| **`v19`** | `pl_r_s0_ema` + `pl_r_s1_ema` + `tn_base_s1` | 32.2 | | **50.76** |
| `v20` | `v19` + `pl_e_s0_ema` | 33.6 | | Built, no score |
| Not built | `v19` + `pl_el_s0_ema` | 33.2 | 15.1 / 18.1 | |
| Not built | `v19` + `pl_t_s0_ema` | 32.7 | 15.1 / 17.7 | |
| Not built | `pl_t_s0_ema` + `pl_e_s0_ema` + `tn_base_s1` | 33.1 | 14.0 / 19.0 | |
| Not built | `v19` + `pl_e_s0_ema` + `pl_el_s0_ema` | 31.9 | 14.9 / 17.1 | |
| Not built | `v19` + `pl_t_s0_ema` + `pl_e_s0_ema` | **31.4** | 14.9 / 16.5 | |

Mixes mainly reduce missing error. Adding models of the same design has stopped giving much, and the local spread of 31.4–33.6 is about noise level.

**Greedy mix search (2026-11-10).** Starting from the best single model, add the model that lowers the local score most, up to 7, over the 9 models available at 02:00 (`queue/mix_search.py`; holdout predictions cached in `queue/clouds/`):

| Models | Local strict | Stray / missing | Added |
|---|---|---|---|
| 1 | 34.18 | 13.70 / 20.48 | `pl_tds_s0_ema` |
| 2 | 31.42 | 15.95 / 15.47 | + `pl_ds_s0_ema` |
| **3** | **30.10** | 15.82 / 14.28 | + `pl_r_s1_ema` → built as **`v22`** |
| 4 | 30.97 | 15.23 / 15.74 | + `pl_el_s0_ema` (even-count stripe penalty, section 7.2) |
| **5** | **29.76** | 15.71 / 14.05 | + `fx_sheet` → built as **`v23`** |
| 6 | 29.39 | 15.70 / 13.70 | + `pl_t_s0_ema` |
| 7 | 29.55 | 15.23 / 14.32 | + `tn_base_s1` |

- **The two deep-supervision models are the best partners.** Together they cut the score from 34.2 to 31.4. Deep supervision makes models that err differently, even though it does not improve a single model.
- **Selection bias.** Members were chosen on the same 204 objects that score them, so 29.8 and 30.1 are somewhat optimistic.
- **Expected leaderboard.** At about 1.2 leaderboard points per local point from `v19` (32.2 → 50.76), `v22` would be about 48.3 and `v23` about 47.8.
- **Actual leaderboard.** `v22` scored **46.35** and `v23` 48.28. Steps 4 and 5 of the search added the two weakest single models (`pl_el_s0_ema` 41.4, `fx_sheet` 42.1), and that gain did not hold on the leaderboard. Stop the search where the next candidate is a weak model.

**Swapping overnight models into the 5-model mix** (core: `pl_tds_s0_ema` + `pl_ds_s0_ema` + `pl_r_s1_ema`, plus two more):

| Added models | Local strict |
|---|---|
| `pl_el_s0_ema` + `fx_sheet` (`v23`) | 29.76 |
| `n_coord_ema` + `fx_sheet` | 29.38 |
| `n_hint_ema` + `fx_sheet` | 29.61 |
| `pl_el_s0_ema` + `n_hint_ema` | 29.79 |
| `pl_el_s0_ema` + `n_coord_ema` | 29.91 |
| `n_flip_ema` + `fx_sheet` | 29.96 |
| `pl_el_s0_ema` + `n_flip_ema` | 30.11 |
| `pl_el_s0_ema` + `n_t_s1_ema` | 30.16 |
| `n_t_s1_ema` + `fx_sheet` | 30.22 |

The best swap is 0.4 lower, within noise, so no new zip was built from it.

### 9.4 Pseudo-labelling (on hold)

Idea: predict Public Eval objects with `fx_sheet`, keep confident ones, and add them to training. To decide what counts as confident, we measured how well two uncertainty signals predict the true error on the holdout:

| Signal | Spearman with error | Relative to object size |
|---|---|---|
| Token-drop spread | +0.59 | +0.28 |
| Model disagreement (two models) | +0.59 | +0.39 |
| Object size alone | +0.58 | |

Neither signal beat object size by itself, so this is on hold. The next check would need four or more models. Pseudo-labels from Public Eval also never touch Private Test images directly.

### 9.5 Teammate runs

`t1_sheet_seed2`, `t2_base_seed1` and `t3_sheet_25ep` were trained on DINOv2 tokens and the **old labels**. On the corrected holdout they score 1247, 1816 and 1381. Zips `v11`–`v13` should not be uploaded. Reruns would need the corrected caches, `--encoder dinov3l` and `--ema 0.999`.

### 9.6 Overnight queue (2026-11-09 23:49 to 2026-11-10)

One change per run on top of sheet attention (`--arch talk --target mm --no-flip-loss --ema 0.999`, DINOv3-L/16, seed 0, 40 epochs). Base: `pl_t_s0_ema` at 34.3 (median 19.0). Each run takes about an hour when nothing else uses the GPU. Script: `queue/night.sh`, log `queue/night.log`. Runs that crash are rerun by `queue/retry.sh` once the queue is empty.

| Run | Change | Training check (normal / EMA) | Strict normal | Strict EMA | Stray / missing (EMA) | Median (EMA) | Lower than base in | Verdict |
|---|---|---|---|---|---|---|---|---|
| `pl_tds_s0` | Deep supervision 0.2 | 36.0 / 35.2 | 35.6 | **34.2** | 13.7 / 20.5 | 18.0 | | Same as base alone; best mix partner |
| `n_t_s1` | Seed 1 (no change) | 36.4 / 35.7 | 38.6 | 36.3 | 13.3 / 23.1 | 17.0 | 14.1% | Seed noise: same recipe, 2 points worse |
| `n_flip` | Flip augmentation: mirrored image tokens, same canonical target | 36.3 / 35.3 | 37.8 | 36.5 | 14.2 / 22.3 | 17.7 | 14.6% | No gain, within seed noise |
| `n_hint` | Size hint: log radius and similarity of the 2 nearest training objects in the same category (class-token cosine), added to the queries | 36.8 / 34.8 | 42.7 | 40.5 | 13.1 / 27.4 | **16.5** | 10.1% | Worse mean, better median |
| `n_patch` | Sheets talk through 4 × 4 regions each (128 tokens) | | | | | | | Crashed at epoch 10 (GPU driver error), retry queued |
| `n_coord` | Queries from 3D coordinates of a template | 35.5 / 34.7 | 39.8 | 37.1 | 13.6 / 23.5 | 19.8 | 6.7% | No gain |
| `n_sphere` | One 64 × 128 latitude-longitude grid | 54.1 / 54.5 | 50.5 | 51.0 | 19.7 / 31.3 | 24.5 | 0.0% | Clearly worse |
| `n_miss` | Loss weights stray 0.85, missing 1.15 | | | 37.6 | 15.6 / 22.0 | 16.6 | 23.2% | No gain |
| `n_cat` | Category dropout 0.15 (unknown category token) | | 47.3 | 38.8 | 14.0 / 24.8 | 20.9 | | No gain (run on Kaggle, section 9.7) |
| `n_ml` | DINOv3 layers 12 + 18 + last, averaged | | | | | | | Not run: training moved to the cloud before its token cache was built |
| `n_rep` | Point repulsion 1.0 | | 48.3 | 45.5 | 16.7 / 28.8 | 19.1 | | Clearly worse (Kaggle) |

"Lower than base in" is the share of 2,000 bootstrap resamples over objects in which the run beats `pl_t_s0_ema`. Ten runs of the queue had not finished when this was written.

Findings:

- **Seed noise is about 2 points for this recipe.** Seed 1 of the base scored 36.3. Flip augmentation (36.5) and coordinate queries (37.1) are within that range: no better, not clearly worse.
- **Size hints help typical objects and hurt a few large ones badly.** The median falls from 19.0 to 16.5 and stray error falls, but missing error rises from 19.8 to 27.4. A wrong neighbour size on a large object costs a lot. The fix would be to limit how far the hint can move the size, or to use it only when neighbours agree.
- **Training check and strict score disagree again.** `n_hint` and `n_coord` have the best EMA training checks (34.8, 34.7) but strict scores of 40.5 and 37.1.
- **The GPU runs at about 87 °C.** `n_patch` died with `CUDA_ERROR_UNKNOWN` mid-run; heat is a likely cause.

### 9.7 Cloud runs: single-model changes (2026-11-10 to 11)

From 2026-11-10 all training moved to Google Colab (L4) and Kaggle (2 × T4, three accounts), several runs in parallel (`baselines/colab/run_exps.py`, `push_kernel.sh`). Scores are local strict on the 204 holdout objects against real-handed shapes, EMA weights. Base `pl_t_s0_ema` 34.3 / median 19.0 (34.5 on real-handed shapes), seed 1 36.3.

| Change (switch) | Runs | Local mean (EMA) | Median | Leaderboard (single) | Verdict |
|---|---|---|---|---|---|
| Learned positional embedding on image tokens (`--pos-emb`) | seeds 0, 1, 2 | 36.6, 37.6, 37.7 | 17.1–18.8 | 55.35 (s0), 60.12 (s2) | Neutral |
| Right-handed targets: un-mirror the 49% of targets PCA mirrors (`--right-handed`) | seeds 0, 1, 2 | 33.9, 39.3, 37.1 | 17.8–20.1 | 57.77 (s0), 54.77 (s2) | At most a small gain |
| Positional embedding + right-handed | seeds 0, 1 | 34.2, 38.9 | 17.3–17.6 | **52.89** (s1, best single) | Best single, within noise |
| Separate size head, log-size offset from category mean, Gaussian NLL (`--size-head`) | 4 runs | 39.9–49.1 | 14.4–17.2 | | Better median, much worse mean |
| Sliced Wasserstein coverage term (`--swd`) | 4 runs | 40.8–43.8 | 19.4–24.3 | | Worse: missing error down, stray error up equally |
| L1 (un-squared) Chamfer (`--l1cd`) | 4 runs | 33.5–37.0 | 16.8–18.5 | 59.76 (s0, 40 epochs) | Best local, bad leaderboard |
| Gaussian noise on image tokens (`--feat-noise` 0.1 / 0.3) | 2 runs | 40.5, 40.6 | 19.7–21.3 | | Worse |
| Sheet attention at 384 px (`--encoder dinov3l384`) | seeds 0, 1 | 37.6, 40.2 | 17.7–19.3 | | No gain, 2× slower |
| No category input: queries from the image's class + mean patch token (`--no-cat`) | seed 0 | 38.4 | 21.8 | **54.01** | Works without category; second-best single |
| No category + Depth Anything V2-Small tokens and relative depth (`--no-cat --extra dav2s`) | seed 0 | 38.9 | 21.1 | not uploaded | No local gain |
| Image augmentation, 8 cached rotation/zoom/colour token sets (`--aug-sets`) | | | | | Built; the runs were lost with a Colab session |

Findings:

- **Local scores do not rank single models.** The best local models (`s_l1_s0` 33.5, `s_rh_s0` 33.9) scored 59.76 and 57.77; v27 (38.9 locally) scored best at 52.89. Same-recipe seeds differ by up to 5 on the leaderboard.
- **The decoder is at its limit.** No change moved a single model outside the 52.9–60.1 range.
- **The category is not needed.** `--no-cat` scored 54.01, better than the base recipe (55.74), so the private test does not depend on category folders.

### 9.8 Mixes with right-handed models (2026-11-10 to 11)

Right-handed models output the real handedness; older members output the mirror image for about half the objects. `baselines/tools/ens.py` aligns each member to a right-handed reference per object (best of 8 axis sign flips, mirrors included, then 10 ICP steps) before the equal point mix.

| Zip | Members | Leaderboard |
|---|---|---|
| **`v28_e5_v22plus_rh2` (RH-Mix5)** | `v22`'s three + v27 + v26 | **43.85 (best)** |
| `v28_e7_v22plus_rh4` (RH-Mix7) | `v22`'s three + four right-handed models | 44.75 |
| `v32_e7_mix5_l1x2` | RH-Mix5 + two L1 models | 45.05 |
| `v32_e6_mix5_l1` | RH-Mix5 + one L1 model | 45.45 |
| `v28_e7_all_strong`, `v28_e4_v22plus_v27`, `v28_e4_rh_only`, `v35_e6_mix5_nocat` | other plans | not uploaded |

Every addition to RH-Mix5 so far has made it worse: more members of the same family only dilute the strongest ones.

### 9.9 Pretrained image-to-3D generators as the shape source (2026-11-11)

Each generator turns the photo into a shape at an arbitrary size and pose. Scored on holdout objects (`baselines/tools/trellis_score.py`): (a) scaled to the **true** RMS radius and rotated onto the truth: shape quality only; (b) scaled to our model's predicted size (best global factor ×1.35) and rotated onto our predicted cloud: what a submission would get.

| Generator | Licence | Objects | Ours | (a) true size | Wins (a) | (b) our size |
|---|---|---|---|---|---|---|
| TripoSG (MIT), Kaggle 2 × T4, ~50 s/photo | MIT | 204 | 38.9 / median 17.4 | 29.6 / 11.7 | | 108.4 (3-size hedge 91.5) |
| **SAM 3D Objects via Fast-SAM3D**, Colab L4, ~7.5 s/photo | Meta SAM License (gated) | 92 | ~49 | **~23** (MetaFood3D 22.6 / 13.9; OmniObject3D 23.1 / 9.6) | **72 of 92** | ~97 |
| Same 20 objects | | 20 | 33.9 | SAM 3D 13.8, TripoSG 35.2 | | |
| IDOL (Apache-2.0), Kaggle | Apache-2.0 | | | running | | |
| TRELLIS.2-4B | MIT | | | never produced a shape (install, attention backend) | | |
| Hunyuan3D 2.1 | licence excludes the EU | not used | | | | |

Findings:

- **SAM 3D's shapes are much better than our decoder's:** about half the error at the true size, better on 78% of objects, on both data sources.
- **No generator knows the size.** SAM 3D puts every object at about 0.5 m. With our predicted size, every generator scores two to three times worse than our model.
- **Why our model survives an uncertain size and a crisp surface does not:** our models pile up to 43% of their points into dense clumps near the surface and spread the rest. Under squared Chamfer that is a hedge. A clean generator surface at a wrong size has no hedge. The model that clumps most (v27) scored best on the leaderboard; L1 Chamfer (0.3% clumped) scored worst.
- **Weak spots:** generators close open half-shell scans (durian), and TripoSG flattens bowls.
- Comparison images and point clouds: `f3d_triposg_compare/` (204), `f3d_sam3d_compare/` (92), each with `index.csv`.
- **Fine-tuning TripoSG on our training set (2026-11-12) cut its shape error at the true size from 29.6 to 24.5** on the 204 holdout objects (section 18). Size is still the problem there too.

## 10. Full submission history

| Zip | Method | Local | Leaderboard | Labels |
|---|---|---|---|---|
| Reference | `atlas3` (teammate) | | 724.73 | Old |
| `v0` | Retrained original | | No score recorded | Old |
| `v1b` | Retrieval + size head + hedge | | No score recorded | Old |
| `v2` | Three-model mix | 175.8 | 761.74 | Old |
| `v2b` | `v2` + reference | | No score recorded | Old |
| `v3` | DINOv3-Atlas | 90.6 | 403.43 | Old |
| `v3b` | DualDINO-Atlas | 124.2 | 551.31 | Old |
| `v4` | DINOv3, no oversized | 77.4 | No score recorded | Old |
| `v5` | DINOv3-DeepAtlas | 58.9 | 213.05 | Old |
| `v5b` | Deep at 384 px | | No score recorded | Old |
| `v6` | DualRes-DeepAtlas | | 220.95 | Old |
| `v7` | DINOv3-DeepAtlas-Fourier | 49.0 | Not uploaded | Old |
| `v8` | DINOv3-SheetAtlas | 48.1 | 173.09 | Old |
| `v9` | DINOv3-SheetAtlas-Fixed | 42.1 | 68.54 | Corrected |
| `v10` | DINOv3-SheetAtlas-Fixed-S1 | 47.9 | 60.22 | Corrected |
| `v11`–`v13` | Teammate, DINOv2, old labels | | Do not upload | Old |
| `v14` | DINOv3-SheetAtlas-Reg | 45.8 | 78.55 | Corrected |
| `v15` | DINOv3-SheetAtlas-EMA | 38.8 | 59.23 | Corrected |
| `v16` | DINOv3-SheetAtlas-EMA-S1 | 40.2 | 66.84 | Corrected |
| `v17` | DINOv3-SheetAtlas-EMA-Mix | 35.4 | 54.53 | Corrected |
| `v18` | Two-model mix | 35.9 | No score recorded | Corrected |
| **`v19`** | **DINOv3-SheetAtlas-Mix3** | 32.2 | **50.76** | Corrected |
| `v20` | DINOv3-SheetAtlas-Mix4 | 33.6 | No score recorded | Corrected |
| `v21` | DINOv3-SheetAtlas-Talk-EMA | 34.3 | 55.74 | Corrected |
| **`v22`** | **DINOv3-SheetAtlas-DS-Mix3** | 30.1 | **46.35** (best, below the competitor's 46.88) | Corrected |
| `v23` | DINOv3-SheetAtlas-DS-Mix5 | 29.8 | 48.28 | Corrected |
| `v24` | DINOv3-SheetAtlas-PosEmb-EMA | 36.6 | 55.35 | Corrected |
| `v25` | DINOv3-SheetAtlas-PosEmb-S2-EMA | 37.7 | 60.12 | Corrected |
| `v26` | DINOv3-SheetAtlas-RightHanded-S2-EMA | 37.1 | 54.77 | Corrected |
| `v27` | DINOv3-SheetAtlas-PosEmb-RH-S1-EMA | 38.9 | **52.89** (best single) | Corrected |
| **`v28_e5_v22plus_rh2`** | **DINOv3-SheetAtlas-RH-Mix5** | | **43.85** (best) | Corrected |
| `v28_e7_v22plus_rh4` | DINOv3-SheetAtlas-RH-Mix7 | | 44.75 | Corrected |
| `v29` | DINOv3-SheetAtlas-RightHanded-S0-EMA | 33.9 | 57.77 | Corrected |
| `v30` | L1 Chamfer, 25 epochs | 37.0 | Not uploaded | Corrected |
| `v31` | DINOv3-SheetAtlas-L1CD-EMA | 33.5 | 59.76 | Corrected |
| `v32_e6_mix5_l1` | DINOv3-SheetAtlas-RH-Mix6-L1 | | 45.45 | Corrected |
| `v32_e7_mix5_l1x2` | DINOv3-SheetAtlas-RH-Mix7-L1 | | 45.05 | Corrected |
| `v33` | DINOv3-SheetAtlas-NoCat-EMA | 38.4 | 54.01 | Corrected |
| `v34` | DINOv3-SheetAtlas-NoCat-Depth-EMA | 38.9 | Not uploaded | Corrected |
| `v35_e6_mix5_nocat` | DINOv3-SheetAtlas-RH-Mix6-NoCat | | Not uploaded | Corrected |

`v22_mix3_ds`: `pl_tds_s0_ema` + `pl_ds_s0_ema` + `pl_r_s1_ema`. `v23_mix5_ds`: those three + `pl_el_s0_ema` + `fx_sheet`. Both have 511 files of exactly 8,192 points, same names as `v9`, and carry their checkpoints, `atlas.py` and `commit.txt`. Description for both: "Equal point mix of independently trained decoders on frozen DINOv3 ViT-L/16 image tokens, including sheet attention and deep supervision variants, using EMA weights."

## 11. Where the error comes from

Measured for the best single model, `pl_t_s0_ema`, on the 204 holdout objects with the local strict score.

### 11.1 Overall

| Measure | Value |
|---|---|
| Mean | 34.3 (stray 14.5, missing 19.8) |
| Median | 19.0 |
| 90th percentile | 77.9 |
| Worst object | 440.3 |
| Share of total error from the worst 5% / 10% / 25% of objects | 28.8% / 41.3% / 66.0% |

The mean is 1.8 times the median, so a minority of objects decides the score. Taking the square root of half the mean gives about 4 mm of typical error in each direction.

### 11.2 Size accounts for a large part of the error

After ICP, each predicted cloud was resized about its centroid over a grid from ×0.70 to ×1.40, and scored at each size.

| Test | Mean | Median |
|---|---|---|
| As predicted (×1.00) | 34.3 | 19.0 |
| **Best size for each object, chosen using the true shape** | **19.1** | 11.0 |

| One constant size for all objects | ×0.90 | ×0.95 | ×1.00 | ×1.05 | ×1.10 |
|---|---|---|---|---|---|
| Mean | 39.1 | 34.9 | **34.3** | 37.0 | 42.8 |

- **Per-object size is worth up to 15 points, about 44% of the mean error.** This is a ceiling, because the test picks the size with knowledge of the answer. No real size predictor would reach it.
- **The sizes are not biased.** One constant multiplier cannot help, and ×1.00 is already the best constant. The median best size is ×0.98 (5th percentile ×0.70, 95th ×1.25). The problem is object-to-object spread: the model does not know each object's real size well enough.
- **The grid is slightly too narrow.** 8.8% of objects hit its edge, so the true ceiling is a little lower than 19.1.
- **Matching RMS radius is the wrong correction.** The predicted RMS radius is only 0.77 times the true value (median), yet the best size is ×0.98. Resizing each cloud to the true RMS radius makes the score worse (66.0). This fits predicted points being concentrated towards the inside of the object, though we have not measured that directly.

Note: separately predicting size and shape lost badly on the old labels (at least 566 against 457). That test replaced the model's size. A small correction to a size the model already gets approximately right has not been tried.

### 11.3 Orientation

Starting ICP from the best of the four PCA sign flips gives 32.3 mean and 16.9 median, against 34.3 and 19.0. Orientation errors are worth at most about 2 points in the mean, much less than size.

### 11.4 By object size

| True RMS radius (mm) | Objects | Mean | Median | Share of total error | Mean CD / radius² |
|---|---|---|---|---|---|
| 6.5–17.9 | 41 | 9.9 | 5.5 | 5.8% | 0.063 |
| 17.9–26.4 | 41 | 20.6 | 14.0 | 12.1% | 0.041 |
| 26.6–33.6 | 41 | 22.2 | 16.5 | 13.0% | 0.025 |
| 34.0–41.7 | 41 | 50.4 | 29.6 | 29.5% | 0.037 |
| 41.9–100.7 | 40 | 69.5 | 47.8 | 39.7% | 0.024 |

The two largest groups of objects produce 69% of the error. Spearman correlation between error and radius is +0.65. Relative to size, small objects are actually the worst, but they contribute little because Chamfer scales with size squared.

### 11.5 By group, source and training examples

| Group | Objects | Mean | Median | Share | Median radius (mm) |
|---|---|---|---|---|---|
| bakery | 35 | 48.1 | 27.6 | 24.0% | 32.3 |
| fruits | 44 | 47.5 | 13.4 | 29.8% | 28.8 |
| seafood | 7 | 35.5 | 24.2 | 3.5% | 25.8 |
| meat | 5 | 30.2 | 16.7 | 2.2% | 31.0 |
| vegetables | 61 | 26.0 | 19.2 | 22.7% | 34.0 |
| extra | 41 | 25.7 | 18.5 | 15.0% | 25.4 |
| packaged | 11 | 17.4 | 2.8 | 2.7% | 10.7 |

| Source | Objects | Mean | Median | Share | Median radius (mm) |
|---|---|---|---|---|---|
| MetaFood3D | 38 | 46.8 | 30.8 | 25.4% | 35.9 |
| OmniObject3D | 166 | 31.5 | 16.6 | 74.6% | 28.6 |

| Training objects in the same category | Holdout objects | Mean | Median |
|---|---|---|---|
| 0–3 | 14 | 38.8 | 30.8 |
| 4–10 | 15 | 32.2 | 18.0 |
| 11–25 | 151 | 34.7 | 16.9 |
| 26 or more | 24 | 30.5 | 22.9 |

- Fruits have a low median and a high mean, so a few bad objects drive that group.
- MetaFood3D objects are harder than OmniObject3D objects, partly because they are larger.
- Having more training examples of a category makes little difference above 3, which suggests category memory is not the main limit.

### 11.6 The 12 worst objects

| Category | CD | Stray | Missing | True radius | Predicted radius |
|---|---|---|---|---|---|
| watermelon | 440.3 | 245.1 | 195.2 | 50.9 | 65.3 |
| fig | 341.9 | 45.0 | 296.9 | 35.9 | 17.6 |
| pancake | 197.5 | 19.7 | 177.8 | 100.7 | 63.5 |
| waffle | 190.1 | 27.6 | 162.5 | 55.7 | 37.1 |
| cake | 181.7 | 13.1 | 168.6 | 39.3 | 16.2 |
| pineapple | 147.3 | 62.9 | 84.3 | 76.3 | 49.0 |
| waffle | 142.4 | 134.0 | 8.4 | 37.4 | 51.3 |
| durian | 136.5 | 27.1 | 109.4 | 60.7 | 48.5 |
| lotus_root | 125.4 | 26.6 | 98.8 | 45.1 | 25.0 |
| banana | 114.4 | 10.9 | 103.5 | 60.8 | 33.4 |
| steak | 101.7 | 7.2 | 94.5 | 58.9 | 35.9 |
| salmon_grill | 100.9 | 15.0 | 85.8 | 44.1 | 26.1 |

Ten of the 12 are predicted too small, with missing error far above stray. Two (watermelon and one waffle) are predicted too large, with high stray error. The worst objects are mostly size errors on large objects.

### 11.7 Mirrored training targets (checked 2026-11-09, harmless so far)

`canonicalize` fixes each axis sign independently and never checks that the result is a true rotation. On the holdout, **95 of 204 objects (46.6%) are mirror images of the real mesh**. The grader can only rotate and translate, so it cannot undo a mirror. We tested what this costs by scoring the predictions against the real, unmirrored shape:

| Test | Mean |
|---|---|
| Against the target as trained, all objects | 32.5 |
| Rotation only, against the real shape, all objects | 32.4 |
| A perfect shape against its own mirror image (mirrored objects only) | 7.5 (median 4.2) |

At the current error level the mirroring costs nothing measurable, so it does not explain why the leaderboard is 1.5–1.7 times higher than the local score. It would start to matter only if error fell close to that 7.5 floor. Inconsistent handedness in the training targets may still make learning harder, which has not been tested.

## 12. What we have learned

| Finding | Evidence strength |
|---|---|
| Training in millimetres end to end beats shape × separate size | Strong, every variant lost by a wide margin |
| DINOv3-L/16 beats DINOv2-B/14 | Strong (leaderboard 724 to 403 on old labels) |
| Multi-block residual decoder beats single cross-attention | Strong (403 to 213, also with fewer bad labels) |
| Corrected labels matter most of all | Strong |
| EMA helps | Strong within a run (`v14` 78.55 against `v15` 59.23 on the leaderboard) |
| Point mixes of no-flip models help | Strong (66.84 and 59.23 down to 54.53, then 50.76) |
| Diverse mix members matter | Moderate. `v10` scored worst locally but helped the mix |
| Sheet-level attention helps | Moderate. Best single model on the leaderboard (`v21` 55.74 against `v15` 59.23), one seed |
| DropPath, attention dropout, grid jitter do not help | Moderate. Consistent at both strengths, one or two seeds |
| Fourier uv, fresh surface samples, token dropout, all queries do not help on `sheet` | Moderate, old-label experiments |
| Deep-supervision models are the best mix partners | Strong. `v22` scored 46.35 on the leaderboard against 50.76 for `v19` |
| Do not add weak models to a mix | Strong. `v23` = `v22` + two models at 41–42 alone: 0.3 better locally, 1.9 worse on the leaderboard (48.28). Same as `v3b` on the old labels |
| Merging by consensus, ICP between members, pruning or farthest-point sampling does not beat the plain point mix | Moderate. All 30.2–40 against 29.8 on the same five models (section 7.2, `queue/proposals_test.log`) |
| A retrieved training shape is no substitute for the decoder | Strong. Most similar training object, in its real size: 74.6–89.8 against 34.3 |
| Use an odd number of models in a point mix | Moderate. Even counts lose 0.5–0.9 to stripe gaps (section 7.2) |
| Flip augmentation, coordinate queries, a spherical grid, size hints do not help a single model | Moderate. One seed each (section 9.6) |
| Align members to a right-handed reference before mixing, and add right-handed members | Strong. RH-Mix5 43.85 against `v22` 46.35 |
| More members of the same family make the best mix worse | Strong. Mix7 44.75, +L1 45.05 and 45.45, all against 43.85 |
| Local scores cannot rank single models | Strong. Best local (33.5, 33.9) scored 59.76 and 57.77; v27 (38.9 locally) scored 52.89 |
| The category input is not needed | Moderate, one seed. `--no-cat` 54.01 against 55.74 for the base recipe |
| Pretrained generators draw better shapes but not the size | Strong for shape (SAM 3D ~23 vs ~49 at the true size, 72 of 92 objects); size is unsolved (section 9.9) |
| Squared Chamfer rewards hedging size with dense point clumps | Moderate. Clump share ranks the three tested single models in leaderboard order |

## 13. Hurdles: why the score is hard to reduce

These are the obstacles we keep running into. Each comes with the evidence behind it and what might get past it.

### 13.1 One image does not show the object's real size

**The problem.** The camera auto-frames every object, so a 3 cm fig and a 10 cm pancake fill the image the same way, and there are no camera intrinsics. Size must be guessed from what the object is and how it looks. Chamfer is squared distance, so a size error costs in proportion to the object's size squared.

**Evidence.**

- Giving each object its best possible size cuts the best single model from 34.3 to 19.1 (section 11.2). That is up to 44% of the error, more than any other cause we have measured.
- The average size is already right: one constant scale factor cannot help, and ×1.00 is the best constant. The problem is object-to-object spread (best size ranges from ×0.70 to ×1.25).
- Ten of the twelve worst objects are predicted too small (section 11.6), for example a fig at 17.6 mm radius against 35.9 mm true.
- Every attempt to predict size directly has failed. A separate size head lost badly on the old labels (≥566 against 457). Size hints from similar training objects (`n_hint`) improved the typical object (median 19.0 to 16.5) but made the mean worse (34.3 to 40.5), because a wrong neighbour size on a large object costs far more than the gains on small ones.

**Why it is hard.** The size is partly not in the image at all. The model already learned to hedge: when it is unsure, it shrinks the cloud towards the inside of the object, which is the Chamfer-optimal answer to uncertainty (predicted RMS radius is 0.77 of the true value, yet the best scale is ×0.98). Any size correction has to beat that hedge, not just be right on average.

**What might get past it.** A size correction that can only move the size a little, used only when similar training objects agree. Using the volume model from the sister task as an extra input. Asking the organisers whether the checkerboard `reference_object` image exists at test time, which would give real scale.

### 13.2 A few objects decide the score

**The problem.** The mean is 1.8 times the median. The worst 10% of objects carry 41% of the error, and the largest 40% of objects carry 69% (sections 11.1 and 11.4).

**Why it is hard.** Any change is judged mainly by about 20 large, hard objects. A change that helps 180 typical objects can still lose if it hurts a handful of big ones, as size hints did. Changes that help typical objects are much easier to find than changes that help the tail.

**What might get past it.** Training that weights large objects by their share of the final score, or a mix member built for the tail, such as a model that spreads points more widely when unsure.

### 13.3 Too little training data, and regularization does not help

**The problem.** 1,838 training objects across 164 categories, about 11 per category. Final training loss is about 8.6 against holdout about 35, a factor of four.

**Evidence.** DropPath, attention dropout and grid jitter did not help at either strength. Token dropout hurt. Flip augmentation (36.5) did not help either. EMA and mixing are the only things that reliably narrow the gap, and they are forms of averaging, not of learning more.

**Why it is hard.** The DINOv3 tokens are cached once per image to make training fit on a 6 GB GPU, so normal image augmentation (crop, colour, rotation) means extracting a new token set for every variant. The encoder is frozen, so the model cannot adapt its image features to food.

**What might get past it.** Training the final models on all 2,042 objects (11% more data). LoRA fine-tuning of DINOv3, which helped on the sister volume task but needs more GPU memory. A few cached augmented token sets.

### 13.4 The decoder design looks close to its limit

**The problem.** After sheet attention, nothing we changed in the decoder improved a single model.

**Evidence.** Overnight (section 9.6): deep supervision 34.2, flip augmentation 36.5, coordinate queries 37.1, size hints 40.5, a spherical grid 51.0, against 34.3 for the base. On the old labels, Fourier uv, fresh surface samples, token dropout and all-point queries also failed.

**Why it is hard.** If most of the remaining error is size and data, a better shape decoder has little left to fix. The per-object best-size test supports this: with sizes right, the error would almost halve without any shape change.

### 13.5 We cannot measure small gains reliably

**The problem.** Most real improvements now are 1–3 points, and our measurements are not much finer than that.

**Evidence.**

- **Seed noise:** the same recipe with a different seed moves the local score by about 2 points (`pl_t_s0_ema` 34.3, `n_t_s1_ema` 36.3) and up to 5–6 points for older recipes.
- **Only 204 holdout objects,** and a few large ones dominate (section 13.2).
- **Selection bias:** the same 204 objects pick the best epoch, choose mix members and score them.
- **Local and leaderboard disagree:** the leaderboard is 1.5–1.7 times the local score and we do not know why. `v10` scored worst locally (47.9) but 60.22 on the leaderboard, better than models that scored better locally.
- **Training check and strict score disagree:** `n_hint` and `n_coord` have the best EMA training checks (34.8, 34.7) but strict scores of 40.5 and 37.1.

**Why it is hard.** Telling a real 2-point gain from noise needs several seeds per change, and each run takes about an hour on this machine.

**What might get past it.** Cross-validation folds instead of one holdout. Two or three seeds for anything that looks promising. Treating the leaderboard as the final judge for close calls.

### 13.6 Mixing has diminishing returns

**The problem.** Mixing has been the most reliable gain (`v17` 54.53, then `v19` 50.76), but each extra model now adds less.

**Evidence.** Greedy search (section 9.3): 3 models 30.1, 5 models 29.8, 6 models 29.4, 7 models 29.6. Swapping new models into the 5-model mix changed it by at most 0.4. Smarter merging (farthest-point sampling, pruning) lost to the simple index stride (section 7.2).

**Why it is hard.** A mix only helps where its models make different errors. Our models share the encoder, data and size hedge, so they tend to make the same size errors on the same large objects, and mixing cannot fix an error every member makes.

**What might get past it.** Members that differ in more than the seed: another encoder, another input resolution, or a pretrained image-to-3D model rescaled to millimetres.

### 13.7 Hardware

**The problem.** One laptop GPU (RTX A3000, 6 GB) does everything: training, scoring and building zips.

**Evidence.** About an hour per training run, one at a time. Running scoring alongside training slowed `pl_ds_s0` to about 2.5 hours. The GPU sits at about 87 °C, and `n_patch` crashed mid-run with a driver error. WSL restarts have killed runs twice.

**Why it is hard.** It limits us to about 10–12 experiments a day, one seed each. That is exactly what makes the measurement noise in section 13.5 hard to beat. It also rules out full fine-tuning of DINOv3-L and large pretrained 3D models.

### 13.8 Smaller issues

- **Orientation:** picking the best of four PCA sign flips would gain at most about 2 points (section 11.3).
- **Mirrored training targets:** 47% of targets are mirror images. This is harmless at the current error level, but would start to matter near 7.5 (section 11.7).
- **The category comes from the folder path:** if a private-test category is unseen, or the path is not allowed as input, the model has no fallback yet (`n_cat`, category dropout, is queued to test this).
- **The leaderboard uses 10,000 surface points against our 8,192,** and its exact alignment is unknown, which may explain part of the 1.5–1.7 ratio.

## 14. Open problems and untested ideas

The ideas below have not been run. They are ordered by my judgement of their value, not by measurements.

1. **Better per-object size.** Section 11.2 puts up to 15 points on this. Options: a small correction head on top of the current model, using nearest look-alike training objects for size (this was the strongest signal on the sister volume task), or feeding the predictions of our volume model in as a size hint. *Tested 2026-11-10:* a look-alike size hint added to the queries (`n_hint`) improved the median but worsened the mean (section 9.6). A bounded version is still untested.
2. **More seeds of sheet attention, then mix.** Cheap, and mixes have been the most reliable gain.
3. **Train the final models on all 2,042 objects.** Using the 204 holdout objects adds 11% more data. The epoch would have to be fixed in advance.
4. **Mirror augmentation.** *Tested 2026-11-10* (`n_flip`): mirrored image tokens with the same canonical target scored 36.5 against 34.3, no gain.
5. **Other image augmentation.** Colour, crop or small rotation with cached token sets, and test-time averaging over them.
6. **Fine-tuning DINOv3 with LoRA.** On the sister volume-estimation task a fine-tuned DINOv3 clearly beat frozen features on held-out folds.
7. **Better starting shape.** Per-category template point clouds, or a retrieved similar training shape as an extra input. On the volume task about a third of evaluation objects had a near-twin in training.
8. **Weighted mixing.** Random, pruned and farthest-point selection all lost to the current method (section 7.2). Weighting by member quality has not been tried.
9. **A wider sheet filter (5 × 5) or more sheet communication.** Small, cheap changes in the same direction as the one that worked.
10. **Consistent orientation for near-symmetric objects.** Worth about 4.5 points on the old-label analysis, but at most about 2 on the current model (section 11.3). No design yet.
11. **Pretrained image-to-3D models** rescaled to metric size, as a mix member that makes different errors.
12. **Multi-layer DINOv3 tokens** or higher input resolution. A 384 px mix did not help on old labels, but that was one test.
13. **Use of `reference_object`.** The checkerboard card might give real scale, but its image is not in our download. Worth asking the organisers whether it is available at test time.

Practical items before the private phase:

- `predict` needs a `test` split option. It currently reads only `val` or `eval`.
- Unseen categories need a proper fallback.
- Confirm the DINOv3 licence terms for a final submission.
- Confirm whether using the category from the folder path is allowed.
- The checkpoint, code version and commit behind any possible final zip must be archived. Several training switches are not yet committed, although each zip folder has a copy of the exact `atlas.py`.

## 15. Questions for a reviewer

0. A per-object size oracle would cut the error from 34.3 to 19.1, and average size is already unbiased. The camera auto-frames each object and there are no intrinsics. What is the best way to predict an object's real size from one image, and how should it combine with an end-to-end millimetre decoder?
1. With 1,838 training objects and a training-to-holdout gap of about 4×, what is the highest-value way to cut overfitting that does not rely on weight-level regularization?
2. The missing error is larger than stray error. How would you reduce it in an AtlasNet-style decoder with 8 sheets of 32 × 32 points?
3. Is there a better query design than category code + sheet code + uv?
4. What would you use instead of equal-share index-stride point mixing for Chamfer-optimal ensembles?
5. How should we handle orientation ambiguity on near-symmetric objects given a loss computed in a PCA frame?
6. Is frozen DINOv3 at 256 px the right trade-off on a 6 GB GPU, or would multi-layer features, fine-tuning or higher resolution be worth the cost?
7. Would a generative prior (for example a pretrained image-to-3D model, rescaled) be worth it at this error level (about 4 mm per direction, root mean square)?

## 16. Reproducing the current best

Run from the repository root.

```bash
# 1. Point clouds and DINOv2 tokens (needed for the point clouds)
uv run baselines/atlas.py cache

# 2. DINOv3 tokens
uv run baselines/atlas.py cache --encoder dinov3l

# 3. Train sheet attention + EMA, seed 0
uv run baselines/atlas.py train --target mm --encoder dinov3l --no-flip-loss \
    --arch talk --ema 0.999 --ckpt baselines/out/atlas/pl_t_s0.pt

# 4. Build the single-model zip (written to submissions/atlas/preds_geometry.zip)
HF_HUB_OFFLINE=1 uv run baselines/atlas.py predict --ckpt baselines/out/atlas/pl_t_s0_ema.pt

# 5. Build a mix (written to submissions/ensemble/preds_geometry.zip)
HF_HUB_OFFLINE=1 uv run baselines/atlas.py predict --ckpt \
    baselines/out/atlas/pl_r_s0_ema.pt \
    baselines/out/atlas/pl_r_s1_ema.pt \
    baselines/out/atlas/tn_base_s1.pt
```

The code is on branch `f3dr_generators` (commit `0e1494b`, pushed 2026-11-11), which contains all training switches, the cloud runners, the mix builder (`baselines/tools/ens.py`) and the generator scripts. Each zip folder holds the exact `atlas.py` used. The current best (RH-Mix5) is built with:

```bash
HF_HUB_OFFLINE=1 ENS_TAG=v28 uv run --script baselines/tools/ens.py $PWD eval e5_v22plus_rh2
```

## 17. Plan from 2026-11-11: generator shape + a learned size model

Tweaking our decoder has stopped paying off, and pretrained generators draw much better shapes but have no idea of size. So:

1. **Shape from a generator.** SAM 3D (best tested) or IDOL (running on Kaggle; fits a 12 GB card, Apache-2.0). Run it on the 204 holdout photos, all 2,042 training photos and the 511 public-test photos. SAM 3D scripts for a local 12 GB GPU: `baselines/sam3d/` (`--low-vram` untested).
2. **Size from a small model of our own.** Inputs: frozen DINOv3 image features (already cached) and about 10 size-free shape numbers from the generator (axis ratios, flatness, roundness). No category input and no look-alike lookup. Two-layer network, about 0.6 M parameters. Output: the object's size and its uncertainty. Trained on the 2,042 training objects with 5-fold cross-validation, keeping near-twin scans in the same fold.
3. **Hedge the size.** Scale the generator's shape to the predicted size; when unsure, split the points over a few sizes around it.
4. **Later:** feed the generator's shape into our decoder as an extra input, so it keeps its size hedge but starts from a much better shape; mix the result with RH-Mix5.
5. **Fine-tuning a generator** only if shape becomes the main error once size is solved. TRELLIS is the only one with released training code; IDOL lacks its shape encoder.

Open checks before relying on a generator in a final submission: whether the competition allows external pretrained models, and the SAM License terms for Bosch.

Update 2026-11-12: step 5 was tried early, with TripoSG (MIT licence), and it worked for shape (section 18). The size part of steps 2 and 3 is now the bottleneck, as planned.

## 18. Fine-tuned TripoSG as the shape source (2026-11-12)

TripoSG's transformer was adapted to our photos with LoRA, and the shape it draws was scored on the fixed 204 holdout objects. Size comes from a separate small head. Code: `baselines/generators/triposg/finetune/` (branch `f3dr_generators`, commit `678c98a`). Everything here is aggregate statistics.

### 18.1 Setup

| Item | Value |
|---|---|
| Split | 2,042 training objects: 1,838 train and the fixed 204 holdout (`baselines/sam3d/holdout_ids.txt`), the same 204 as in every other section. No train object shared an exact mesh with a holdout object (the excluded count was zero) |
| Cache | TripoSG VAE posterior, 2,048 latent tokens × 64 channels, 32,768 surface points per object, PCA pose, mesh unit 1.0 mm. Photo preprocessing: the full 1024 × 1024 RGB photo (these photos have no alpha mask), white square padding of 10% |
| Hardware | Colab A100 80 GB, one session (`tg`), stopped 2026-11-12 about 18:25 Vietnam time |
| Objective | Rectified flow: x = (1 − σ)·latent + σ·noise, target = latent − noise, model time 1000σ, σ uniform, 10% condition dropout, latent sampled from the VAE posterior. Matches the sampler in TripoSG's own scheduler (`x + (σ − σ_next)·v`, shift 1) |
| Size head | Small network on TripoSG's DINOv2 image features (257 × 1024 tokens), log-radius target. Best epoch 30 |
| Full fine-tune | All transformer weights, A100 config. 4 epochs run; validation loss best at epoch 0, so `full` = epoch 0 and `fulllast` = epoch 3 |
| LoRA | Rank 16, alpha 16, dropout 0.05, lr 5e-5, batch 1 × 16 accumulation, bf16, at most 20 epochs, patience 5, on `to_q`, `to_k`, `to_v`, `to_out.0`. Ran 7 full epochs (about 20 min each, 2 h 17 min in all), then we stopped it in epoch 7. **Best epoch 4**, exported as a 42 MB file |
| Inference | fp16, 50 steps, guidance 7, mesh grid depth 7/8 (256³), seed 42, then 8,192 points sampled from the mesh. 204 objects × 4 arms at once on one A100 took 88–91 min per arm (about 26 s per object per arm while sharing) |

### 18.2 The flow-matching loss barely moves, and it is not a bug

Validation loss on the 204 holdout objects, fixed noise per object, no condition dropout:

| Model | Validation loss |
|---|---|
| Pretrained TripoSG, untrained | 0.8847 |
| LoRA, best (epoch 4) | 0.8838 (−0.1%) |
| Full fine-tune, best (epoch 0) | 0.8824 (−0.3%) |

LoRA by epoch (0 to 6): 0.8846, 0.8844, 0.8839, 0.8842, 0.8839, 0.8841, 0.8840. Full fine-tune (0 to 3): 0.8824, 0.8838, 0.8851, 0.8842. Training loss is about 1.39–1.40 because of condition dropout and posterior sampling.

Checks run on the VM to rule out a bug:

| Check | Result |
|---|---|
| Loss with the sign of the target flipped | 3.06 (against 0.88), so the sign is right |
| Zero predictor (target variance) | 1.45, so the model is far from ignoring the input or time |
| Overfit one object, LoRA at lr 3e-4, 400 steps | Mean loss falls from 0.91 to about 0.38–0.43, so gradients reach the adapters |
| LoRA weights at epoch 4 | All 168 `lora_B` matrices are non-zero; the update is 0.65% of the weight norm (median, largest 1.2%) |

Pretrained loss by noise level σ (the lower the σ, the cleaner the input):

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---|---|---|---|---|
| Pretrained loss | 1.01 | 0.95 | 0.98 | 0.88 | 0.56 |
| Zero predictor | 1.45 | 1.45 | 1.45 | 1.46 | 1.45 |

We believe the loss is almost flat because the random noise in the target cannot be predicted (about 1.0 at low σ) and the part that depends on the photo is small. This is an explanation, not something we measured directly. The loss cannot tell good shapes from bad ones here: the shape error below fell fivefold while the loss fell 0.1%. Judge these runs on geometry, not on loss.

### 18.3 Holdout shape and score (`evaluate.py`, 204 objects)

CD-L2 in mm², lower is better. All arms use the same size head, so their size error is identical (radius error 55.9 mm²). "Predicted size" is a submission-like score; "true size" rescales each cloud to the real radius, so it measures shape alone.

| Arm | Predicted size, mean / median | True size, mean / median |
|---|---|---|
| Pretrained | 154.98 / 79.49 | 120.78 / 55.48 |
| **LoRA, epoch 4** | **79.27** / 33.41 | **24.17** / **11.87** |
| Full fine-tune, best (epoch 0) | 82.48 / 33.35 | 26.88 / 12.51 |
| Full fine-tune, last (epoch 3) | 84.47 / 31.96 | 26.23 / 12.39 |

LoRA beat the pretrained model on 167 of 204 objects at the predicted size (mean −75.7, median −23.8) and on 185 of 204 at the true size (mean −96.6, median −33.5). The full fine-tune (epoch 0) beat it on 193 of 204 at the true size. LoRA is at least as good as the full fine-tune on both means, from a 42 MB file instead of 5.7 GB.

`evaluate.py` is a local diagnostic: 24 rotation starts, 20 ICP iterations each, 8,192 truth points. It is not the report's strict scorer, so these numbers are not comparable with sections 8 to 11.

### 18.4 The report's own scorer on the LoRA clouds

`baselines/tools/trellis_score.py` on the same 204 objects, centre plus 10 ICP steps from the given pose:

| Arm | CD-L2 mean | Median |
|---|---|---|
| Atlas single model `v27` (a member of RH-Mix5) | 38.87 | 17.4 |
| LoRA at the true size, best of 24 rotations | **24.47** | 13.1 |
| LoRA at the RMS radius of Atlas's cloud, rotated onto Atlas's cloud | 189.80 | 109.9 |
| Same, then ICP with scale clamped to ±15% | 197.24 | 112.0 |

Atlas scores 38.87 here, matching the 38.9 of section 9.9, so this is the same scorer and the same objects. LoRA is better than Atlas on only 9% of objects at Atlas's size.

The "Atlas's cloud" rows use the cloud's own RMS radius. Atlas's clouds are hedged and come out about 1/1.35 of the true size, so this is the worst calibrated way to size a crisp surface. For zero-shot TripoSG the same arm gave 188.6, a global ×1.35 factor gave 108.4 and a 3-size hedge gave 91.5 (section 9.9). Those two variants have **not** been run for LoRA yet.

Shape at the true size, other generators (different object sets, so only roughly comparable): LoRA 24.5 / 13.1 on 204; zero-shot TripoSG 29.6 / 11.7 on 204, with BiRefNet masks; IDOL 24.1 / 11.5 on 204; SAM 3D about 23 on 92. Same-subset comparison with SAM 3D is still to do.

### 18.5 Why the predicted size fails

True radius of the holdout objects: median 29.1 mm. Radius error of three ways to guess the size:

| Size guess | Median error | Within 10% | Off by more than 30% |
|---|---|---|---|
| One size for everything (train median) | 28% | 16% | 47% |
| Median size of the object's food category | 13% | 42% | 19% |
| **Our size head** | 13% | 41% | 18% |

The size head is no better than the category median, so it has learned the category's typical size and nothing from the photo itself. Effect on LoRA's score, holdout objects grouped by size error:

| Size error | Objects | CD-L2 at predicted size | CD-L2 at true size |
|---|---|---|---|
| 0–10% | 84 | 23.4 | 19.8 |
| 10–20% | 54 | 56.8 | 28.6 |
| 20–30% | 29 | 94.0 | 26.2 |
| over 30% | 37 | 227.5 | 26.1 |

The shape scores about the same in every row (20–29), while the predicted-size score rises from 23 to 228. The 37 objects with the largest size error add about 40 points to the mean of 79.

**Correction (2026-11-13).** An earlier version of this paragraph said the size head was trained on oversized OmniObject3D labels. That was wrong. The size head trained on radii from the current, corrected meshes (recomputed from `mesh.ply` for all 2,042 objects, they agree with the holdout labels to 0.6%). The oversized radii (265 objects, 33 of the 204 holdout objects, more than 5 times the current value) are in the old `baselines/out/scale/scale.npz`, which is stale and which the TripoSG size head does not use (`analysis/stale_check.py`). So "drop the oversized labels" is not a fix for this size head.

### 18.6 Caveats

1. **The pretrained arm saw unmasked photos.** Our photos have no mask and the pipeline pads the full frame, while the zero-shot TripoSG run in section 9.9 used BiRefNet masks (29.6 at true size). So part of LoRA's fivefold gain over "pretrained" in 18.3 may be learning to ignore the background. The fair comparison is 24.5 (LoRA) against 29.6 (masked zero-shot), a gain of 17%, and the two use different scorers. Not yet tested: LoRA on masked photos, or the pretrained arm with a mask.
2. **One LoRA run, one seed (42).** Epoch 4's validation loss beats epochs 2 to 6 by at most 0.0003, so the choice of epoch 4 is within noise. The best epoch of each arm was picked on the same 204 objects that score it.
3. **Leaderboard number (updated 2026-11-13).** The 79.3 is a holdout figure and, by section 8.2, leaderboard scores have run 1.3 to 1.7 times local ones for Atlas. For this pipeline the ratio is now known from one point: the look-alike version (holdout 60.87) scored 102.82 on the leaderboard, a ratio of 1.69 (section 18.10). Because of that ratio, a holdout mean must be compared with RH-Mix5's holdout equivalent (about 26 to 29, from 43.85 divided by 1.5 to 1.69), not with 43.85, which is a leaderboard score.
4. **Full fine-tune weights.** The fp32 files (5.7 GB each) were lost when the VM was stopped. fp16 copies are in the Kaggle dataset below. Inference runs in fp16, so those copies give the same clouds.

### 18.7 Public test: status and how to run it

**Update 2026-11-13: run, on a Colab L4 instead of Kaggle.** The 511 clouds (LoRA epoch 4 and our size head, seed 42, 50 steps, depth 7/8) are merged in `baselines/out/triposg/public/lora_eval.npz` and `lora_eval.zip`: ids equal the 511 public photos, 8,192 finite points each, 511 `geometry/<id>.ply` in the zip. The first L4 VM was terminated after 2.5 h (we sent no stop; the cause is unknown) and its 157 finished photos were lost; the other 340 photos were rerun as 12-way shards in 2 h 31 min (about 2.2 photos per minute, about 25 compute units), see `finetune/colab/`. This zip is the size head's version, which scores 79.3 on the holdout and was not uploaded; the same clouds at the look-alike size were uploaded by the user and scored 102.82 (section 18.10).

What follows is the Kaggle route, which has not run for this public test. On 2026-11-12 Kaggle refused the kernel: this account had used 36.93 of its 30 GPU hours, and the quota resets at 2026-11-14 00:00 UTC (07:00 Vietnam time). Everything needed is in place:

| Item | Where |
|---|---|
| Private Kaggle dataset | `anhanhanhanh/f3dr-tg-infer`: `code.tar`, the 511 public-test photos (`eval.tar`, 511 `rgb_industrial.png` checked), `size.pt`, `lora.pt`, `full.pt`, `fulllast.pt` (the last two in fp16) |
| Kernel | `finetune/kaggle/run.py` and `kernel-metadata.json` (id `anhanhanhanh/f3dr-tg-public`, arm list `["lora"]`); two T4s, one shard each, then `merge.py` writes the zip |
| Push | `kaggle kernels push -p <folder with run.py and kernel-metadata.json> --accelerator NvidiaTeslaT4` |
| Local copies | `baselines/out/triposg/holdout/`: holdout clouds and scores for all four arms, size head, LoRA file, training logs (`logs/runlogs.tgz`), `metadata.json` |

The kernel has never run: its setup (torch 2.5.1 + cu124, `diso` built for the T4's sm_75) is only tested on Colab. Expect about 50 s per photo on a T4 (section 9.9), so roughly 3.5 hours on two T4s, which is an estimate and not a measurement. A Colab A100 would take about 1 to 2 hours (about 6.6 s per object of GPU time, measured above).

### 18.8 What to try next

1. **Free, on this machine:** how accurate is Atlas's own size on the same 204 objects? If it beats the category median, LoRA can use it.
2. **Free, on this machine:** place the LoRA shape at several sizes around Atlas's size (hedge), at a calibrated factor, and mix some LoRA points into RH-Mix5. Judge each against RH-Mix5's own holdout score under the same strict scorer.
3. **Size model:** the look-alike size of section 18.10 is better than the size head (squared radius error 39.5 against 55.9) and is the size model to build on. The earlier idea to "drop the oversized training labels" is withdrawn (correction in 18.5). A bigger network on the same inputs is unlikely to help, because the category median already matches the head.
4. **Only if one of these beats RH-Mix5 on the holdout:** build that combination as a zip from the public clouds, which now exist (18.7). Do not upload a lone LoRA zip: at the look-alike size it scores 60.9 on the holdout (about 29 is the bar), at the size head's size 79.3.
5. Remove the mask confound in 18.6 (masked photos) and add a second LoRA seed before relying on the 24.5.
6. **Mixing the LoRA into RH-Mix5** is the one route with a chance. It needs RH-Mix5's holdout clouds, which are not cached (`v27` and `v26` would have to be rebuilt from their checkpoints; `analysis/mix_lora.py` is a first draft, never run), and a per-object rule for when to trust the LoRA, for example only where the look-alikes are close and agree.

### 18.9 Practical notes from the run

- Colab's `exec` connection drops often. Run long jobs detached on the VM and poll them.
- Loading TripoSG from the Hugging Face repo id fails on its custom scheduler entry. `predict.py` now downloads a snapshot and loads from the local path.
- `predict.py` writes its output only after the last object, so a half-finished arm has no score. Scoring one arm locally takes 10 to 14 minutes on 14 CPU cores.
- `predict.py` has `--shard` and `--num-shards`, and `merge.py` joins the shards into the challenge zip (exactly 8,192 points per object, `geometry/{id}.ply`).
- Scripts behind the numbers are in `finetune/analysis/`: `fm_diag.py` and `lora_diag.py` (18.2, run on the GPU VM), `score_local.py` (18.3, parallel version of `evaluate.py`), `size_baselines.py` and `radius_check.py` (18.5). The last three run from `baselines/out/triposg/holdout/` and need `metadata.json` and the `*_val.npz` files there. The 18.4 numbers come from `trellis_score.py` run on `lora_val.npz`. The 18.10 numbers come from `size_lookalike.py` and `hedge_surrogate.py`, which run with `uv run --no-project --with numpy --with scipy --with trimesh python -I <script>` (the usage is at the top of each script).

### 18.10 Look-alike size for the LoRA shape (2026-11-13)

**The idea.** The LoRA draws the shape without a trustworthy size, so take the size from the training photos instead (the method of the sister volume task's B3.4). For each photo: (1) the 2 training photos of the same food type whose image features are closest (cosine on standardized SigLIP2 CLS features of three layers); their real radii, averaged in log space; (2) a ridge formula on the same features, as a correction around the food type's median radius; (3) the type median. The three are mixed in log space and the LoRA's unit-RMS cloud is scaled to that radius with `common.metric_points`, so only the size number changes. Chosen out-of-fold on the pool: SigLIP2 features (of 7 feature choices tried; the best four are within 2 mm² of squared radius error of each other, so the choice is not sharp), ridge alpha 100, weights 0 / 0.95 / 0.05 for type median / look-alikes / formula. The formula adds almost nothing: the ridge alone is barely better than the type median (101.5 against 111.0 mm² out of fold). The volume model's guess (listed as optional) was not used: its look-alike part is this same idea on volume, and its fine-tuned part has no out-of-fold guesses for our holdout.

**No leakage.** The 204 holdout objects are never in the pool (pool = the 1,838 train objects). The features, the ridge strength and the mix weights were chosen out-of-fold on that pool (leave-one-out for the look-alikes with identical meshes also left out, 10 folds by mesh for the ridge) and the holdout was judged once. For the public photos the pool is all 2,042 training objects with the same settings. The cached features come from the volume task; its photos are byte-identical to ours (checked for all 2,553). Closeness check, cosine similarity of the best same-type look-alike, percentiles 10 / 25 / 50 / 75 / 90:

| Queries and pool | 10 | 25 | 50 | 75 | 90 |
|---|---|---|---|---|---|
| Train pool, leave one out | 0.506 | 0.625 | 0.734 | 0.816 | 0.865 |
| Holdout, train pool | 0.534 | 0.622 | 0.723 | 0.814 | 0.858 |
| Public photos, all 2,042 | 0.488 | 0.617 | 0.731 | 0.812 | 0.868 |

The holdout is as close to its pool as the public photos are to theirs, so it does not flatter the method. Re-weighting the holdout errors to the public photos' closeness gives a squared radius error of 39.1 for the blend (39.5 measured) and 55.8 for the head (55.9 measured).

**Size accuracy, 204 holdout objects.**

| Size guess | Median error | Within 10% | Off by >30% | Squared radius error (mm²) |
|---|---|---|---|---|
| Type median | 13.2% | 42% | 19% | 87.5 |
| Our size head | 13.4% | 41% | 18% | 55.9 |
| Look-alikes, K = 2 | 8.5% | 55% | 11% | 40.2 |
| Ridge formula | 12.4% | 44% | 17% | 80.0 |
| **Blend (the plan)** | **8.2%** | **54%** | **11%** | **39.5** |

Out of fold on the 1,838-object pool the same blend gives 7.2% / 60% / 11% / 50.6, against 10.8% / 48% / 21% / 111.0 for the type median.

**Score, the local scorer of 18.3 (my harness reproduces the saved 79.27 and 24.17 to 0.01).**

| LoRA at | Mean CD-L2 | Median |
|---|---|---|
| Size head's guess | 79.27 | 33.41 |
| **Look-alike size** | **60.87** | **27.18** |
| True size (oracle) | 24.17 | 11.87 |

The look-alike size is better than the size head's on 124 of 204 objects. As before, the score lost to the size error is almost the squared radius error (36.7 against 39.5 here, 55.1 against 55.9 for the head). So a mean score under 43.85 needs a squared radius error of about 20, and a score near the roughly 29 of RH-Mix5 needs about 5.

**Where the error is.** The 10 worst objects carry 44% of the squared radius error, the 40 worst 82%. The 22 objects with a radius above 50 mm carry 40% (watermelon halves, a large pancake, nachos, steak, sausage). On several of the worst, the size head is closer than the look-alikes. An exploratory geometric mean of the head and the blend lowers the squared radius error from 39.5 to about 33, but the head has no out-of-fold guesses for the pool, so its weight cannot be chosen honestly without retraining it by fold.

**A size hedge** (`hedge_surrogate.py`, a fast stand-in for the scorer that keeps each cloud in its true-size pose and only rescales it; within about 5% of the real scorer on the arms checked: 25.5 against 24.2 at the true size, 75.0 against 79.3 at the head's size, 61.4 against 60.9 at the look-alike size). Mean CD on the 204 holdout objects:

| Cloud | Mean | Median |
|---|---|---|
| One size, look-alike | 61.4 | 29.7 |
| 3 copies at (0.9, 1, 1.1), a third of the points each | 52.7 | 25.4 |
| 5 copies at (0.84, 0.92, 1, 1.08, 1.16), weights 1:2:3:2:1 | **50.1** | 25.2 |
| 5 copies, narrower (0.90 ... 1.10), d = 0.05 | 51.8 | 25.0 |
| 5 copies, wider (0.76 ... 1.24), d = 0.12 | 52.8 | 30.7 |
| 5 copies, much wider (0.68 ... 1.32), d = 0.16 | 60.0 | 36.5 |
| Hedge width set per object from the look-alikes' closeness and agreement | 59 to 80 | |
| One size, radius times 0.97 | 60.0 | |

A narrow fixed hedge helps by about 18%; a per-object width does not. 50 is still far above the roughly 29 of RH-Mix5 on this holdout. Not scored by the real scorer.

**Verdict.** The look-alike size is a real gain on the size head (79.3 to 60.9) but does not meet the rule "upload only if it beats 43.85 on the holdout". It is also the wrong yardstick: 43.85 is the leaderboard score of RH-Mix5 on the public test, and leaderboard scores have run 1.3 to 1.7 times holdout scores (section 8.2), so on this holdout RH-Mix5 would score about 29 (measured later, 18.11: 27.4 with this scorer, 30.7 with the report's strict scorer). A standalone LoRA submission therefore does not look like a route to a better leaderboard score, even with a hedge (about 50). The file is `f3dr_submissions/v36_lora_lookalike_size/preds_geometry.zip` (511 clouds, 8,192 points each, ids equal to the public photos, radii from the 2,042-object pool; median new radius 31.4 mm against 30.8 mm for the head).

**Leaderboard result (2026-11-13, 09:27 Vietnam time): 102.8221.** The zip was uploaded by the user (not by me) against the recommendation above. It confirms the verdict: RH-Mix5 is 43.85. The ratio to the holdout is 102.82 / 60.87 = 1.69, the top of the 1.3 to 1.7 range seen for our Atlas models (section 8.2), so the local scorer and the holdout predict the leaderboard for this pipeline too. My written estimate before the upload was "roughly 80 to 100"; the real score is slightly worse. By the same ratio the size head's version (79.3 on the holdout) would have scored about 134, and the LoRA needs a holdout score of about 26 to 29 (43.85 divided by 1.5 to 1.69) to beat RH-Mix5.

**Agreement of the two look-alikes predicts the error.** The log-radius gap between the two look-alikes (0.056 and 0.174 are the terciles of the holdout) separates the objects. Out of fold on the 1,838-object pool, which was not used to pick the cuts: look-alikes within 6% of each other (685 objects) have a median size error of 4.1%, a squared radius error of 21.9 mm² and 4% off by more than 30%; 6 to 19% apart (601): 7.2%, 58.0, 11%; more than 19% apart (552): 13.8%, 78.3, 19%. On the 204 holdout objects the score at the look-alike size is, by group: agree (68 objects) mean 37.9 / median 19.9, partly agree (65) 32.4 / 19.0, disagree (68) 105.6 / 57.8, and only one look-alike in the training set (3 objects, a rare food type, one of them a steak at 473) 185 / 51. The grouping was chosen after looking at the holdout (similarity of the best look-alike separated the objects much less), so the holdout bands are indicative. The public test has 175 / 173 / 160 / 3 photos in these groups.

`visualize_public_lookalike.py` draws this for the 511 public photos (`baselines/out/triposg/public/lora_lookalike_view/index.html`): the LoRA output at the look-alike size, the two look-alike training photos with their similarity and real radius, and the holdout score band for the photo's agreement group. The public test has no ground truth, so the band is the score of holdout objects like this one, not the photo's own score.

**Open (superseded by 18.11, which answers items 1 and the shape question).** (1) Mix LoRA points into RH-Mix5, with a per-object rule based on the look-alikes (item 6 of 18.8): tested in 18.11, it does not help. (2) Out-of-fold size-head guesses for the pool, to weight the head and the look-alikes honestly. (3) Real-scorer numbers for the hedge.

### 18.11 Is the LoRA's shape better than RH-Mix5's, and can it help RH-Mix5? (2026-11-13)

The question came from the leaderboard result of 18.10 (102.82 against 43.85): "I thought the LoRA was good on our holdout". The 24.2 to 24.5 of 18.3 and 18.4 is the LoRA given each object's true size; RH-Mix5's holdout score is its honest score, with no help. This section compares them on equal terms. RH-Mix5's holdout clouds were rebuilt with the same code that builds its submission (`baselines/tools/ens.py root holdout_save OUT.npz`, new mode; members `v27`, `v26`, `pl_tds_s0_ema`, `pl_ds_s0_ema`, `pl_r_s1_ema`, aligned to `v27`, equal point mix). They were scored with the same scorer and the same truth samples as the LoRA (`size_lookalike.py`, `evaluate.aligned_cd`: 24 rotation starts, 20 ICP steps).

**RH-Mix5 on the 204 holdout objects, measured:** 27.39 mean / 16.24 median with the scorer of 18.3; 30.66 / 18.3 with the report's strict scorer (`ens.py holdout_save` prints it). Its leaderboard score is 43.85, so the leaderboard to holdout ratio is 1.43 (strict) or 1.60 (18.3 scorer). The earlier estimate of "about 29" was right.

**Shape on equal terms** (`shape_best_size.py`: each cloud aligned once, then resized about its centre from x0.50 to x1.60, best factor per object chosen with knowledge of the truth; neither cloud is moved when resized):

| CD-L2 | As given | Best size for each object | Median best factor |
|---|---|---|---|
| RH-Mix5 (as given = its own size) | 27.39 / 16.24 | **16.52** / 11.43 | x0.95 |
| LoRA (as given = the true RMS radius, an oracle) | 24.17 / 11.87 | 23.27 / 11.86 | x1.00 |

As given, the LoRA is 3.2 better (95% interval -8.7 to +2.0, so within noise) although it is handed the true size and RH-Mix5 is not. With the best size for each object, RH-Mix5 is better by 6.7 (interval +3.7 to +10.2). So the LoRA's shape is not better. Its typical object is about as good (median 11.9 against 11.4) but it has a heavier tail: crisp surfaces with failures such as the closed half-shells (watermelons, 119 to 179 even at the true size), where a hedged cloud loses less. The "24.5 against 38.9" of 18.3 and 18.4 compared the LoRA with the true size to Atlas at its own size; Atlas at its best size (section 11.2) was 19.1 on an older model, consistent with the 16.5 above for RH-Mix5.

**Per-object choice between the LoRA (at the look-alike size) and RH-Mix5** (`hybrid_rhmix.py`):

| Look-alikes | Objects | RH-Mix5 mean / median | LoRA mean / median | LoRA better on |
|---|---|---|---|---|
| agree | 68 | 21.8 / 14.9 | 37.9 / 19.9 | 24 |
| partly agree | 65 | 21.8 / 13.6 | 32.4 / 19.0 | 30 |
| disagree | 68 | 38.0 / 26.2 | 105.6 / 57.8 | 14 |
| only one look-alike | 3 | 35.4 / 11.0 | 185.3 / 50.6 | 0 |

RH-Mix5 is better on average in every group. Using the LoRA where the look-alikes' gap is below 0.03, 0.056, 0.10, 0.174, 0.30 gives a mean of 33.0, 32.8, 33.7, 36.1, 45.0 against 27.4 for RH-Mix5 alone, every one worse (95% intervals all above zero). A perfect per-object chooser would reach only 23.1. So handing photos to the LoRA does not help. Mixing its points into RH-Mix5 was not tested; it is not expected to help, because every addition to RH-Mix5 so far has made it worse (section 9.8) and the LoRA is much worse than RH-Mix5 at the sizes it can actually be given.

**What is left: the look-alike size as a nudge for RH-Mix5's own size** (`rhmix5_size_nudge.py`, a stand-in scorer from the resize curves, about 5% off). RH-Mix5's clouds have a median RMS radius of 0.76 of the true one and a best resize of about x0.95; up to 11 points (27.4 to 16.5) sit in its size. The look-alike radius carries some information about that resize: the correlation of log(look-alike radius / RH-Mix5 radius) with the best resize is +0.33 over all objects, +0.49 where the look-alikes agree, +0.56 where they agree or partly agree, and +0.09 where they disagree. A bounded resize (clipped to x0.85 to x1.15, strength fitted on 9/10 of the holdout and judged on the rest, 20 shuffles) gives 26.2 against 27.4 (26.6 when applied only where the look-alikes agree or partly agree). That is about 1.2 points, within the seed noise of 2 (section 13.5), and the 11-point ceiling stays almost untouched because the look-alike size is not accurate enough. Not built as a zip.

**Verdict.** The LoRA path is closed: on its own (102.82 on the leaderboard), as a hybrid with RH-Mix5 (worse than RH-Mix5 alone for every gate) and as a shape (not better than RH-Mix5's when sizes are equal). The look-alike size is a weak lead for correcting RH-Mix5's size, worth at most about a point on the holdout.

# Experiment plan (written 2026-11-05)

Current best: **blend probe 37.42 board** (B3.4 0.6 + B4.0 folds 0,1 mean 0.4). B3.4 alone 38.37, B4.0 alone ~40.5-41.5.
Lesson: blending different models helps (diversity), single-fold gaps < ~2 pts are noise, B3.4's board edge comes from kNN with the full-train bank.

Gates (all runs): adopt only if it gains >= 0.5 CV on seeds 0-2, holds on fresh seeds 3-5, or (for fine-tuned runs) >= 1 pt on the same 2 folds vs B4.0. Board slots (5/day) are for confirming, never for tuning weights.

| ID | Question | Where | Cost | Status / gate |
|----|----------|-------|------|---------------|
| B5.1 | Recalibrate P(oversized) on OOF | local CPU | done | NO GAIN (+0.08 / -0.12). Dropped. |
| B5.2 | Does a frozen DINOv3 ConvNeXt-B / -L add to B3.4 (alone vs concat)? | local CPU | ~15 min | running; adopt if concat gains >= 0.5 on both seed sets |
| B4.0e | 5-fold B4.0 average in the blend (folds 0-4) | local GPU | fold 4 ~1 h | OOF blend check on 5 folds, then 1 board slot |
| B4.2 | kNN on fine-tuned embeddings (val queries vs full train bank, per fold model) | local GPU, inference only | ~30 min | board-only evidence (OOF is leaky); 1 slot |
| B5.3 | Frozen MoGe-2 shape/depth stats added to B3.4 | local GPU/CPU | ~1 h | adopt if >= 0.5 CV |
| B4.1 | Bundle: patch-grid conv+attn-pool head + mesh-derived aux targets (convexity, compactness, PCA ratios, silhouette) + 20 epochs + EMA + head warm-up | Kaggle GPU | ~2 h/fold | folds 0,1 vs B4.0 (40.34 / 38.02); need >= 1 pt on BOTH; then ablate |
| B6.1 | Fine-tune the CNN (DINOv3 ConvNeXt-B, LoRA-free full FT or last-stages) with the B4 heads | Kaggle GPU | ~2 h/fold | only if B5.2 shows the CNN features are decorrelated/competitive; folds 0,1 gate as B4.1 |
| B6.2 | Re-render train meshes (new views/lighting) as augmentation | local CPU render + Kaggle train | days | first check render matches test look on 5 meshes; stop if not |
| B4.3 | Diversity members: other seeds/augment, ViT-H+ LoRA (fits on a 16 GB Kaggle T4) | Kaggle GPU | ~2-3 h/fold | adopt a member if it improves the 5-fold OOF blend by >= 0.5 |

## Order
1. Finish what is running: B5.2 scores, fold 4 -> B4.0e OOF blend check (5 folds) -> one board slot for the 5-member blend.
2. B4.2 (inference only) and B5.3 (MoGe-2) while Kaggle is being set up.
3. Kaggle: B4.1 on folds 0,1; in parallel decide B6.1 from the B5.2 result.
4. Stack the winners (B4.1 members, CNN members, 5-fold B4.0e) in one OOF blend; final weights from OOF, not the board.

## Not doing
Second-stage mode classifier (B5.1), expected-sMAPE decoder (equivalent to a threshold), 3D-encoder distillation (image has no scale), external dataset copies (rules unclear), board-tuning weights.

## Open items
Folder category at inference (ask organisers); DINOv3 licence for handover (keep DINOv2 fallback); keep the host's Kaggle approval in writing; Kaggle credentials + HF token secret needed.

---
# Finetune and CNN integration (added 2026-11-06)

Board so far: B3.4 38.37 -> blend(B3.4 + folds 0,1) 37.42 -> blend(B3.4 + 5 folds) **37.15 (best)**. Frozen ConvNeXt-L added to B3.4: 40.32 (worse; CNN line dropped, B6.1 cancelled).

## How the conv head (B4.1) works  -- design proposal, not yet built

```
 image 512x512
     |
 DINOv3 ViT-L/16  (LoRA r16 on q/k/v, 24 layers, bf16, grad-checkpointing)
     |
     +--> CLS token ........................................ [1024] ------------------+
     |                                                                                |
     +--> 4 register tokens (dropped)                                                 |
     |                                                                                |
     +--> 1024 patch tokens  [1024 tokens x 1024 dim]                                 |
              |  reshape to a grid (32 x 32 patches, 1024 channels)                   |
              v                                                                       |
        1x1 conv  1024 -> 128      (cheap channel reduction)                          |
        3x3 conv  128 -> 128  + GroupNorm + GELU        (local shape: edges, thickness)|
        3x3 conv  128 -> 128, stride 2 + GN + GELU      (grid 32x32 -> 16x16)          |
              |                                                                       |
        attention pooling: 4 learned queries attend over the 16x16 map -> [4 x 128]   |
              |  flatten -> [512]                                                     |
              +-----------------------------------------------------------------------+
                                        concat [1024 + 512]
                                                |
                                  LayerNorm -> Linear 512 -> GELU -> Dropout 0.1
                                                |
                  +--------------+--------------+-----------------+
                  |              |                                |
           r  (ln size      oversized                aux targets (training only):
           correction)      logit                    sorted ln bbox dims (3), ln fill,
                            (BCE)                    + NEW mesh targets: convexity,
                                                     compactness, PCA ratios (2),
                                                     silhouette fraction
```
- Why: CLS + mean-patch (B4.0) throws away WHERE things are; the grid conv lets the head read outline/thickness cues.
- Size: roughly 0.6-0.9M new parameters (my estimate). Risk: overfit on ~2k objects -> keep LoRA dropout, head dropout, EMA, 20 epochs.
- Decode unchanged: cluster value + clipped r, oversized cluster if p > 0.5.

## Run order
1. Frozen C-RADIOv4-H screen (`nvidia/C-RADIOv4-H`, ~3-4 GB, local): adopt only if it adds >= 0.5 CV on seeds 0-2 AND 3-5. Read the model card for licence first.
2. Mesh targets (local CPU) -> upload small file to the private Kaggle dataset `anhanhanhanh/fve-bosch-data`.
3. B4.1 on DINOv3-L, folds 0,1, Kaggle T4 (clean test of head + mesh targets + 20 epochs + EMA). Need >= 1 pt over B4.0 (40.34 / 38.02) on BOTH folds.
4. B4.2 = same recipe on C-RADIOv4-H (only if step 1 shows signal and step 3 works). Keep backbone switchable in the script so only one thing changes per test.
5. Stack winners in a 5-fold blend; weights from OOF, not the board.

## Geometry models (MoGe-2 etc.)
Agent rejected them (assume intrinsics/scene context); we test scale-free depth ratios on ~50 images first: do they predict fill (volume / bbox volume) better than DINO features? Pi3 weights are non-commercial: avoid.

## REVISION 2026-11-06: conv head re-designed after reading the MoGe-2 paper (arXiv 2507.02546, HTML, Appendix A.1, Sec. 3.2)
Verified MoGe-2 facts: DINOv2 encoder (ViT-L); features from layers 6, 12, 18 and final, projected to a common dim, reshaped to (h,w) and SUMMED; a lightweight conv neck = residual conv blocks + transpose convs (k2,s2) upsampling to 16h x 16w, normalisation layers removed, UV positional encoding injected at every neck scale; metric scale head = 2-layer MLP on the CLS token only, hidden = input dim, exp output, trained with L2 in log space (online-optimal scale target, stop-grad). Channel widths / number of levels NOT stated. (MoGe-2 repo: MIT; DINOv2 part Apache-2.0; HF `Ruicheng/moge-2-vitl` 326M, `-vitb-normal` 104M.)

What we borrow / do not borrow (our task is ONE global number, not a dense map):
- BORROW multi-layer fusion: take patch tokens from layers 6/12/18/24 (B4.0 used the last layer only; our frozen pipeline already gained from 3 layers).
- BORROW coordinate (UV) channels concatenated to the grid (position in the auto-framed image is informative for silhouette/placement).
- BORROW residual conv blocks; BORROW CLS-token MLP for the size/scale-like output.
- DO NOT borrow transpose-conv upsampling (dense output only); we DOWNSAMPLE (stride-2) and pool.
- DEVIATE: keep GroupNorm (MoGe removes norm for latency; with 2k objects norm should help stability). Untested.

```
 image 512 -> DINOv3 ViT-L/16 (LoRA) ->  patch tokens from layers 6, 12, 18, 24   [4 x 32x32x1024]
                                           |  per-layer 1x1 conv 1024->128 ; SUM          (MoGe-2 style fusion)
                                           v
                                    grid 32x32x128  + 2 UV coordinate channels
                                           |  residual conv block (3x3, GN, GELU)
                                           |  3x3 conv stride 2 -> 16x16
                                           |  residual conv block
                                           v
                              attention pooling (4 learned queries) -> [512]
 CLS token (last layer) [1024] ------------+-> concat [1536] -> LN -> Linear 512 -> GELU -> Drop
                                                  -> r | oversized logit | aux (bbox, fill, mesh shape targets)
```
Ablate after a win: (a) multi-layer vs last layer, (b) UV channels, (c) conv grid vs CLS+mean.

# Experiment log

CV = out-of-fold sMAPE (5-fold, 3 seeds, seeds 0-2 select / 3-5 fresh). Board = public leaderboard (511 objects), as scored by the
platform. Lower is better. B0-B5 were trained on an older copy of the data (labels with oversized unit errors, meshes); their CV did
not transfer to the board (CV ~40, board 47-54). B6 is trained on the official FVE zip, where CV and board agree.

| ID | What | Script | CV | Board |
|----|------|--------|----|-------|
| EDA | labels, clusters, duplicates | `eda/01_labels.py`, `eda/b3_1_duplicates.py` | | |
| B0 | global median | `baselines/b0_b1.py` | 94.07 | |
| B1 | category median | `baselines/b0_b1.py` | 52.59 | |
| B2 | per-category sMAPE-argmin constant | `baselines/b2.py` | 51.4 | |
| B3 | frozen DINOv2-L-reg@518 + cluster + ridge + logistic | `baselines/b3.py` | 48.95 | 53.99 |
| B3.2 | tuning (CLS only, no PCA, C=0.01, clip 2.5) | `experiments/b3_2_tune.py` | 47.37 | |
| B3.3 | + DINOv3-H+/16@512 | `experiments/b3_3_dinov3.py` | 44.71 / 44.35 | 50.05 |
| B3.4 | + within-category kNN (k=2, blend .75) | `experiments/b3_4_knn.py` | 40.82 / 40.19 | 48.69 |
| B3.5 | decoding tweaks (tau, offset) | `experiments/b3_5_decode.py` | no gain | |
| B3.6 | SigLIP2-so400m instead of / next to DINOv3 | `experiments/b3_6_siglip2.py` | 41.33 / 40.27 (+0.31 fresh) | not adopted |
| B4.0 | LoRA DINOv3-L, fold 0 only | `experiments/b4_0_finetune.py` | fold 40.34 | 50.37 |
| B4.0 | same, fold 1 only | | fold 38.02 | 50.16 |
| B4.0e | 5 folds, OOF | | alone 40.94 vs B3.4 40.53; blend w=.5 **39.31 (+1.22)** | |
| blend | B3.4 + mean(B4.0 folds 0,1), w=.4 | `b4_0_blend_submit.py` | | 47.70 |
| blend5 | B3.4 + mean(B4.0 folds 0-4), w=.5 | `b4_0_blend_submit.py 0.5 0,1,2,3,4 ...` | | 47.46 |
| B5.1 | recalibrate P(oversized) on OOF (stage 2) | `experiments/b5_1_mode_calib.py` | +0.08 / -0.12 | dropped |
| B5.2 | frozen DINOv3 ConvNeXt-B/-L alone / concat | `experiments/b5_2_cnn_features.py` | alone 43.6; concat +0.24 / +0.38 | ConvNeXt-L concat 49.38 (worse) |
| B5.3 | frozen C-RADIOv4-H alone / concat | `features/extract_radio.py`, `b5_2_cnn_features.py` | alone 42.83 / 42.68; concat 40.91 / 40.37 (worse than B3.4 40.82 / 40.19) | dropped |
| B4.2 | LoRA C-RADIOv4-H + conv head (Kaggle, folds 0,1) | `kaggle_pkg/b4_2_train.py` | fold 0 43.05, fold 1 40.56 (B4.0: 40.34, 38.02); 3-way blend adds only 0.2 | |
| B4.1 | LoRA DINOv3-L + conv head + mesh aux (local, folds 0,1) | `experiments/b4_1_chain.sh` | running | |
| B4.3 | B4.2 + regularisation (Kaggle) | `kaggle_pkg/kernel/launcher.py` | running | |
| B6 | LoRA DINOv3-7B + query decoder, aux from gt.json only, official zip, folds 0,1 | `kaggle_pkg/b4_2_train.py --head decoder`, `experiments/b6_submit.py` | fold 42.04 / 38.75 | 39.21 |
| **B6s** | B6 + residual shrink n/(n+4) by category train count | `b6_submit.py ... 4` | fold 41.12 / 37.65 | **38.80 (best)** |

## How each stage works (input -> output)

**B1/B2 (category statistics).** Input: category of the object. Output: the stored constant for that category (median, or the sMAPE
argmin for the normal cluster). Image unused. Shows that category alone is worth ~51%.

**B3 / B3.2 / B3.3 (frozen features + ridge/logistic).** Input: cached frozen features (DINOv2 and/or DINOv3), category. Stages:
cluster table -> ridge residual -> logistic oversized probability -> decode. Output: ln volume. The image carries real within-category
size signal (residual R2 ~0.29 versus ~0.02 for shuffled labels) and a visible oversized mode (within-category AUC ~0.88).

**B3.4 (kNN).** Adds the within-category nearest-neighbour step. Near look-alikes in the same category share their size and their
oversized flag, so the neighbour residual and oversized rate beat the ridge alone (kNN alone 42.2 vs ridge).

**B3.5 / B3.6 / B5.1 (negative results).** Decoding tweaks, SigLIP2 features, and a second-stage mode classifier did not clear the
adoption gate. The mode decision is already calibrated.

**B4.0 (fine-tuning).** Input: the image, fold-specific targets. LoRA on DINOv3-L, small head. Output: head values -> decode. On its
own it only matches B3.4 (no kNN, 80% of the data), but its mistakes differ, so averaging with B3.4 helps.

**Blending.** Input: B3.4 ln predictions and the B4.0 ln predictions (mean over fold models for val). Output: weighted ln average with
the B3.4 cluster call kept when the models disagree by >= 30x. The 5-fold OOF gain (+1.22, CI +0.57..+1.88) was confirmed on the board
(48.69 -> 47.46).

**B5.2 (frozen CNN features).** Extra ConvNeXt features concatenated into the B3.4 stages. CV looked slightly better, but the
board was ~0.7 points worse: small CV gains on frozen concatenations are not reliable.

**B4.2/B4.1/B4.3 (conv head).** Patch-grid conv head over multi-depth tokens, EMA, mesh-derived auxiliary targets. B4.2 (C-RADIOv4-H)
overfit (train loss ~0.09, mode loss 0.001) and its errors correlate 0.90 with B4.0, so it adds only 0.2 to the blend.

## Decisions and rules
* Do not switch to a CNN; frozen CNN features were a negative result. Do not tune blend weights on the board.
* Never hedge between clusters. (Wrong-cluster calls dominated on the older labels; on the official zip only 4 of 818 held-out
  objects are in two-cluster categories, so the error is within-category size.)
* Open questions for the organisers: whether the folder category may be used at inference (we use it), whether external copies of
  OmniObject3D / MetaFood3D are allowed (we do not use them), DINOv3 licence for the code hand-over (keep a DINOv2 fallback).

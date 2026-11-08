# Viable methods for Food 3D Reconstruction

The grader aligns a prediction to the scanned mesh by rotation and translation only.
Scale stays in the score. The pose stored on `mesh.ply` means nothing. A method that
copies a training mesh, or that supervises the raw file pose, does not match the task.

Three published methods fit this setting.

1. **Point Set Generation Network** (Fan, Su, Guibas, CVPR 2017). A conv net reads one
   image and emits a point cloud. The loss is Chamfer distance, the same family as the
   challenge CD-L2. This repo trains it on a principal-axis frame in millimetres, so the
   loss sees shape and size and not the arbitrary scan pose. Command:
   `uv run baselines/baseline.py train`.

2. **AtlasNet** (Groueix et al., CVPR 2018), which is what `baselines/atlas.py`
   trains. A frozen DINOv2 ViT-B/14 reads the image. Each of the 8192 output
   points attends to the patch tokens, then a small MLP maps that vector to
   millimetres. A learned category vector is added to the query. It is not a
   retrieved mesh. The checkpoint is the best holdout epoch. ViT-B/14 at 224
   fits an L4.

3. **Arbitrary-scale mesh, then a learned volume scale** (Gao, Coburn, Raghavan, He,
   Zhu, "Size Matters", 2026). One-2-3-45 reconstructs a mesh in unknown units from one
   view. A second network predicts `V_true / V_mesh`, and the mesh is scaled by the cube
   root. They evaluate on MetaFood3D and OmniObject3D, the two sources of this challenge.
   Their own controls are the dataset-mean volume and the category-mean volume. This
   needs the external reconstructor and its weights. It is the upgrade when (1) cannot
   represent a thin surface such as a shrimp.

## Improvements on this branch

The score keeps scale and squares millimetres, so a size error on a large object costs
more than any shape error. Each change below is one commit.

1. **Separate size head** (`baselines/scale.py`). The decoder now draws a unit-size
   shape: targets are divided by their RMS radius. A second model predicts that radius
   in mm from frozen image features: the FVE volume pipeline retargeted from log volume
   to log radius (category/cluster table, ridge offset, P(oversized)), without kNN.
   The output size is the posterior mean of the radius, because squared Chamfer error
   grows with (s - s_true)^2, and settings are picked on out-of-fold mm^2 error.
   MoGe-2 (Wang et al., 2025) makes the same split between an affine-invariant shape
   and a separate metric scale. `atlas.py evaluate` scores the 10% holdout with the
   predicted and with the true size, which splits the error into size and shape parts.
2. **Hedged output** (`geom.hedge`, `scale.py hedge`). When P(oversized) is unsure, the
   cloud keeps all points at the likely size and adds a share at the other cluster size.
   Wrong extra points only cost their share of one Chamfer half; if that size was right
   they save the other half, which would be the full size error. The threshold and the
   share are tuned on every training object with its out-of-fold size and its true
   shape. This comes from our own reasoning and a toy simulation, not from a paper.
3. **Loss fixes** (`atlas.chamfer`). PCA fixes the axes but not their signs, and a
   near-symmetric axis flips between similar objects. The grader removes rotation, so the
   loss takes the minimum over the 4 proper sign flips of the target. It also uses exact
   nearest neighbours over all 8192 points instead of 2048 random ones per side, and,
   since change 1, it is computed on size-normalised targets.
4. **Retrieval baseline** (`atlas.py predict --shape retrieve`, `evaluate --retrieval`).
   Copy the unit shape of the most similar training image (DINOv2 class token and mean
   patch token, cosine, same category) and give it the size head's size. Tatarchenko et
   al. ("What Do Single-view 3D Reconstruction Networks Learn?", CVPR 2019) show this
   matches or beats single-view decoders, which mostly recognise and recall. Earlier
   versions of this file ruled retrieval out as memorising; it is kept here as the bar
   the decoder has to clear, and the predict flag makes it a submission if it wins.

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

Not in this baseline: retrieving a training mesh by image similarity and scaling it.
That memorizes the training shapes. It is also someone else's leaderboard entry.

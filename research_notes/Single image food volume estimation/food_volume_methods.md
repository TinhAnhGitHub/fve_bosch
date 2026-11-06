# Single-image food volume / portion / nutrition estimation: methods and datasets (excluding MetaFood challenge writeups)

Scope note: research done 2026-11-04. "Verified" = numbers read in the paper HTML/PDF or the repo page during this session. "Secondary" = taken from a search-result summary or aggregator and not checked against the paper. Repo existence was checked by HTTP status on 2026-11-04 (200 = exists). All volume numbers below are **MAPE or MAE, not sMAPE**, and use **different splits** from the competition, so they are not directly comparable to the leaderboard metric.

Competition relevance in one line: the competition image is auto-framed (object always about 20% of the frame) with no reference object or intrinsics. So the image carries shape, category and texture/appearance but almost no direct metric size cue. The closest published setting is **"Size Matters" (Vinod et al., arXiv 2601.20051, Jan 2026)**, which trains a learned scale regressor on MetaFood3D and OmniObject3D, the same two sources as the competition.

---

## Q1. Main method families and how well each works without a reference object

### Takeaway
On MetaFood3D, the strongest reference-free results are hybrid methods: an image or a scale-less 3D/point-cloud proxy plus a **learned regression of scale or volume** (Size Matters 35.8% vol. MAPE; MFP3D 41.4%; PortionNet 17.4%, unreviewed and on a different split). Pure geometric methods such as depth integration, stereo or voxels are poor without a metric reference (79–278% MAPE). A per-category mean is a surprisingly strong baseline, especially on OmniObject3D, where it nearly matches the best learned method.

### Cited Findings
**Family A: direct regression from RGB (Nutrition5k style)**
- Nutrition5k (CVPR 2021): InceptionV2 backbone, 256x256 center-cropped overhead image, JFT-300M pretraining, multi-task MAE heads. 2D direct prediction reaches calorie MAE 70.6 kcal (26.1% of mean) and mass MAE 40.4 g (18.8%). The mean-prediction baseline is 150.8 kcal (60.2%) and 124.6 g (58.5%). The portion-independent model reaches calories per gram MAE 9.5%. This is verified from the arXiv PDF. Note that the % here is "MAE as % of mean GT", not MAPE. — [Nutrition5k arXiv 2103.03375](https://arxiv.org/abs/2103.03375)
- Nutrition5k states that "predicting direct calories compared to calories per gram increases the MAE nearly 3x from 9.5% to 26.1%". Portion (size) is therefore the dominant error source, not food identity. Verified. — [Nutrition5k arXiv 2103.03375](https://arxiv.org/abs/2103.03375)
- In Size Matters, the "RGB Estimation" baseline (Nutrition5k-style ResNet regression) gives volume MAPE 86.84% on MetaFood3D (MAE 161.70 mL) and 158.74% on OmniObject3D (MAE 91.38 mL). Verified from paper HTML. — [Size Matters arXiv 2601.20051 (HTML)](https://arxiv.org/html/2601.20051v1)
- In MFP3D Table 1, "RGB Only" energy regression on MetaFood3D has MAPE 1,124.90% (MAE 1,932 kcal). This is worse than the mean baseline (1,287% MAPE, 221 kcal MAE), which suggests direct RGB regression is unstable on the small MetaFood3D training set. Verified. — [MFP3D arXiv 2411.10492 (HTML)](https://arxiv.org/html/2411.10492v1)
- NutritionVerse (Waterloo) reports that direct prediction with Nutrition5k weights performed best on synthetic NV-Synth (calorie MAE 128.7, mass MAE 77.2 g). On NV-Real, direct prediction with ImageNet weights trained on real images was best (calorie MAE 296.9, mass 115.9 g). Depth did not help direct prediction. Verified from HTML v2. — [NutritionVerse arXiv 2309.07704](https://arxiv.org/html/2309.07704v2)

**Family B: depth-based integration (sensor depth or predicted monocular depth)**
- Nutrition5k with sensor depth: calorie MAE 47.6 kcal (18.8%) using depth as a 4th channel. The volume-scalar approach uses known camera height 35.9 cm and per-pixel area 5.957e-3 cm², sums per-pixel volume over a food mask, then feeds the result to InceptionV3 mass regression. It improves mass MAE from 38.1 g (29.5%) to 29.4 g (13.7%), and the end-to-end calorie MAE is 41.3 kcal (16.5%). The scale comes from **known capture geometry**, not from the image. Verified. Inconsistency: the image-only mass MAE is 40.4 g / 18.8% in Table 3 but 38.1 g / 29.5% in Table 4. — [Nutrition5k arXiv 2103.03375](https://arxiv.org/abs/2103.03375)
- DPF-Nutrition (Han et al., *Foods* 2023; arXiv 2310.11702) predicts monocular depth and fuses it with RGB through a cross-modal attention block and a multi-scale fusion network. It reports mean PMAE 17.8% on Nutrition5k, an 11.3% improvement over Thames et al. Secondary, from a search summary; no code found. — [DPF-Nutrition arXiv 2310.11702](https://arxiv.org/abs/2310.11702); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10706621/)
- "Depth Reconstruction" baseline in Size Matters: Metric3D monocular depth, voxelization, then **category-mean calibration using GT**. It gives volume MAPE 277.65% on MetaFood3D and 87.84% on OmniObject3D. Monocular "metric" depth alone does not recover food scale. Verified. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
- Graikos `food_volume_estimation` (GitHub, MIT, no paper or accuracy numbers): a monocular depth net (Godard-style self-supervised, trained on EPIC-Kitchens and fine-tuned on smartphone food videos), Mask R-CNN food segmentation fine-tuned on UNIMIB2016, and a point cloud built from depth, mask and intrinsics. Scale comes from **user-supplied camera FoV plus an expected camera-to-food distance (`gt_depth_scale`)**, or optionally from a plate diameter found by ellipse detection. Volume converts to weight using a FAO/INFOODS density database. Verified from README; repo exists. — [github.com/AlexGraikos/food_volume_estimation](https://github.com/AlexGraikos/food_volume_estimation)
- "Image Based Food Energy Estimation With Depth Domain Adaptation" (Vinod, Shao, Zhu; IEEE MIPR 2022; arXiv 2208.12153) has two parts: an energy density map estimated from RGB, and a feature adaptation module that transfers depth-domain features to RGB. It reportedly reaches 13.29 kcal average error (about 13.57%), with Nutrition5k cited as the evaluation dataset. Secondary, from a search summary. — [arXiv 2208.12153](https://arxiv.org/abs/2208.12153)
- "Towards Learning Food Portion From Monocular Images With Cross-Domain Feature Adaptation" (Shao et al., Purdue; arXiv 2103.07562) does single-image energy estimation on a dietitian-labeled real-life dataset with 11.47% MAPE. Secondary. — [arXiv 2103.07562](https://arxiv.org/abs/2103.07562)

**Family C: 3D reconstruction plus learned scaling (the best reference-free family)**
- Size Matters is detailed under Q2. On MetaFood3D (stratified split) it reaches volume MAE 59.09 mL and MAPE 35.83%, compared with 86.84% for RGB regression, 63.25% for GPT-4o with context, and 202.88% for the category mean. Verified. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)

**Family D: geometric template / 3D model pose fitting (needs a reference)**
- "Food Portion Estimation via 3D Object Scaling" (Vinod, He, Shao, Zhu; CVPRW 2024, pp. 3741–3749) estimates camera and object pose, then renders a known 3D food model in those poses to recreate the eating occasion. **Scale comes from a physical reference in the scene** (a checkerboard, according to Size Matters' description of this baseline). It reports 31.10 kcal (17.67%) average energy error on SimpleFood45. Verified (abstract). — [arXiv 2404.12257](https://arxiv.org/abs/2404.12257); [CVF open access](https://openaccess.thecvf.com/content/CVPR2024W/MTF/html/Vinod_Food_Portion_Estimation_via_3D_Object_Scaling_CVPRW_2024_paper.html)
- Run on MetaFood3D, this "3D Assisted" method gives volume MAPE 79.33% (MFP3D Table 3), 104.07% (Size Matters Table I) or 83.26% (Size Matters Table II). It works well only when its reference assumptions hold: on SimpleFood45 it reaches 14.01% volume MAPE, the best there. Verified. — [MFP3D HTML](https://arxiv.org/html/2411.10492v1); [Size Matters HTML](https://arxiv.org/html/2601.20051v1)

**Family E: point cloud / voxel learning**
- MFP3D is detailed under Q2. Its point clouds come from ZoeDepth or TripoSR, encoded with CurveNet plus a ResNet50 image branch, and regressed with L1. On MetaFood3D it reaches volume MAPE 41.43% (62.60 mL) and energy MAPE 68.05%. Verified. — [MFP3D HTML](https://arxiv.org/html/2411.10492v1)
- PortionNet (arXiv 2512.22304, Dec 2025) distills a PointNet teacher into an RGB-only student. It reports volume MAPE 17.43% and energy MAPE 15.36% on MetaFood3D (unreviewed; no code). Verified from HTML. — [PortionNet arXiv 2512.22304](https://arxiv.org/html/2512.22304v1)
- Point2Volume (Lo et al., IEEE Trans. Industrial Informatics 2020, 16(1):577–586, DOI 10.1109/TII.2019.2942831) uses UNet/VNet to complete partial point clouds from **depth input** (view synthesis) and regress volume in cm³. Its predecessor (Lo et al., *Nutrients* 2018) does view synthesis from a single depth map on irregular YCB objects. Scale comes from the depth sensor. Secondary (abstracts only); error numbers not checked; no code found. — [Point2Volume record](https://researchr.org/publication/LoSQL20/bibtex); [Nutrients 2018 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6316017/)
- Stereo and voxel reconstruction baselines on MetaFood3D give volume MAPE of 210.90% (stereo) and 104.07% (voxel) in MFP3D, or 214.95% and 96.31% in Size Matters Table II. Verified. — [MFP3D HTML](https://arxiv.org/html/2411.10492v1); [Size Matters HTML](https://arxiv.org/html/2601.20051v1)

**Family F: VLM/LLM** (see Q4). In Size Matters, GPT-4o gives volume MAPE 153.80% on MetaFood3D, or 63.25% with context. On OmniObject3D it gives 169.81%, or 155.28% with context. Verified. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)

### Inferences
- Every pure-geometry method needs a metric anchor (sensor depth with known camera height, a checkerboard, a plate diameter, FoV plus distance). The competition renders remove all of these, and auto-framing removes apparent size too. Families B and D are therefore unusable as published. Only learned priors (Families A, C, E with learned scale, and F) apply.
- The size of the category-mean baseline matters for the competition. In Size Matters, the category mean reaches 87.46% MAPE on OmniObject3D against 85.96% for the best method. On MetaFood3D it reaches 202.88% against 35.83%. A plausible reading is that OmniObject3D's within-category size variance is low relative to label noise, while MetaFood3D foods vary more within category but carry learnable appearance cues. That reading is mine, not the authors'.
- MAPE punishes over-prediction without bound, and some baselines exceed 800%. sMAPE is bounded at 200%. Rankings under sMAPE can differ, especially for objects with roughly 1000x unit errors.

### Gaps
- No paper found reports **sMAPE** on MetaFood3D or OmniObject3D volume.
- Fang et al. (Purdue, ICIP 2018, GAN "energy distribution" maps) and older single-view template methods (Purdue TADA, goFOOD/Bern) were not verified in this session and are omitted.
- DPF-Nutrition, depth-domain-adaptation and cross-domain-adaptation numbers are secondary only.

---

## Q2. What each named work does, and which report MetaFood3D numbers

### Takeaway
Three Purdue VIPER papers report single-image volume numbers on MetaFood3D: MFP3D (ICPR 2024 MADiMa workshop), Size Matters (arXiv Jan 2026) and the multi-food benchmark (SSIAI 2026). PortionNet (VIT, arXiv Dec 2025) reuses MFP3D's split and reports the lowest MAPE, but it is unreviewed and has no code. Size Matters is the only paper found that also reports OmniObject3D volume prediction.

### Cited Findings
**Nutrition5k** (Thames et al., Google; CVPR 2021). 5,006 plate scans with 4 side-angle Raspberry Pi videos (30°/60°) and overhead RealSense RGB-D. Raw depth is 16-bit at 10,000 units per meter, capped at 0.4 m. Labels are dish mass, calories, macronutrients and per-ingredient mass (USDA). Official dish-ID splits are provided, CC BY 4.0. The repo has eval scripts but **no model code**. Results are as in Q1. A human survey found 53% mean percent error for non-nutritionists and 41% for nutritionists, and the model beat both. Repo exists (archived). Verified. — [Nutrition5k repo](https://github.com/google-research-datasets/Nutrition5k); [arXiv 2103.03375](https://arxiv.org/abs/2103.03375)

**MetaFood3D** (Chen, He, Vinod, ... Wong, Delp, Zhu; arXiv 2409.01966, v1 Sep 2024, v2 Dec 2024; no venue listed on arXiv). 743 scanned and labeled 3D food objects across 131 categories. Each has textured mesh, RGB-D videos, segmentation masks, weight, nutrition values and food codes linked to a nutrition database (FNDDS per Size Matters). Data host: lorenz.ecn.purdue.edu/~food3d/. That host timed out from this machine, so its existence was not confirmed. Downstream papers use a 637-object / 108-category subset. Verified (abstract). — [arXiv 2409.01966](https://arxiv.org/abs/2409.01966)

**MFP3D** (Ma, Zhang, Vinod, Raghavan, He, Zhu; 9th MADiMa Workshop at ICPR 2024; arXiv 2411.10492). Verified from HTML:
- Input: single RGB image, SAM segmentation, no depth sensor, no reference, intrinsics not mentioned. **No explicit scale-recovery step.** Scale is learned implicitly in regression.
- Pipeline: ZoeDepth depth map, back-projected to a 1,024-point cloud (TripoSR mesh sampling as an alternative). The point cloud goes to CurveNet (chosen over PointNet), the image to ResNet50 (512-d). Features are concatenated and a linear head regresses volume or energy with L1 loss.
- Data: MetaFood3D 637 objects / 108 categories, 510 train / 127 test, plus SimpleFood45.
- MetaFood3D volume: MAE 62.60 mL, **MAPE 41.43%**. Energy: 77.98 kcal, 68.05%.
- SimpleFood45 volume: 25.83 mL / 16.15%, behind 3D-Assisted's 14.01%.
- Ablation: GT point cloud plus RGB gives 15.59% volume MAPE (upper bound). Normalized (scale-removed) GT point cloud plus RGB gives 42.19%, about the same as depth point cloud plus RGB at 41.43%. TripoSR point cloud plus RGB gives 39.45%. The authors conclude "the true scaling factor carries critical information."
- Code: [github.com/jingema99/MFP3D](https://github.com/jingema99/MFP3D) (exists).
- Sources: [MFP3D HTML](https://arxiv.org/html/2411.10492v1); [arXiv abs](https://arxiv.org/abs/2411.10492v1)

**Size Matters: Reconstructing Real-Scale 3D Models from Monocular Images for Food Portion Estimation** (Vinod, Coburn, Raghavan, He, Zhu; arXiv 2601.20051, 27 Jan 2026; no venue listed). Verified from HTML:
- Pipeline: SAM segmentation, then zero-shot One-2-3-45 single-view mesh with arbitrary scale (TripoSR is worse). The mesh is rendered in Blender at polar angles −45°, 0° and 45° with even azimuths (75 renders; m = 25 is best).
- A frozen CLIP ViT-L/14@336px encodes the input image f(x) and each render f(r_i). The concatenated 1536-d vector goes through a 3-layer MLP that predicts per-view **v_scale = V_gt / V_reconstructed**. Predictions are averaged over views and the mesh is scaled by the cube root of v_scale.
- **Loss: L1 normalized by 1/v_scale (relative error)**, which penalizes small items more.
- Training: 300 epochs, batch 64, lr 1e-4 decayed 0.7 every 10 epochs, 4x GTX 1080 Ti.
- Data: MetaFood3D 637 items (535 train / 102 test) and OmniObject3D 3,417 models (2,763 / 654), stratified by category.
- Table I volume results:
  - MetaFood3D: MAE 59.09 mL, **MAPE 35.83%**, R² 0.75.
  - OmniObject3D: MAE 70.49, **MAPE 85.96%**, R² 0.88.
  - Baselines on MetaFood3D / OmniObject3D MAPE: dataset mean 845.69 / 844.29; category mean 202.88 / 87.46; RGB est. 86.84 / 158.74; Depth recon 277.65 / 87.84; 3D Assisted 104.07 / –; GPT-4o 153.80 / 169.81; GPT-4o with context 63.25 / 155.28.
- Table II (MetaFood3D): volume 61.24 mL / 37.82%. Energy from FNDDS density is 163.7 kcal / 42.73%.
- Splits and transfer:
  - A random (non-stratified) MetaFood3D split gives **82.46% MAPE**, against 299.52% for RGB-only.
  - Training on OmniObject3D and testing on MetaFood3D gives MAE 104.08 mL, against 174.57 for RGB-only.
- Ablations:
  - Image-only feature f(x): 76.10% MAPE. Render-only f(r): 90.80%. The pair: 35.83%.
  - CLIP encoders beat non-CLIP ones. DeiT-S is weakest at 67.75%.
- Internal inconsistencies: 59.09 vs 61.24 mL, and 3D-Assisted at 123.34 vs 186.45 mL.
- Code: [gitlab.com/viper-purdue/size-matters](https://gitlab.com/viper-purdue/size-matters) (exists).
- Source: [Size Matters HTML](https://arxiv.org/html/2601.20051v1); [abs](https://arxiv.org/abs/2601.20051)

**Food Portion Estimation via 3D Object Scaling** (Vinod et al., CVPRW 2024). Covered in Q1 Family D: a pose-and-render template method with a physical reference, introducing the SimpleFood45 dataset (45 items with volume, weight and energy; [data](https://lorenz.ecn.purdue.edu/~gvinod/simplefood45/)). It reports 31.10 kcal / 17.67% energy error. The CVF open-access abstract says "SimpleFood40" while arXiv says "SimpleFood45", an unresolved discrepancy (secondary). Code: [gitlab.com/viper-purdue/monocular-food-volume-3d](https://gitlab.com/viper-purdue/monocular-food-volume-3d) (exists). — [arXiv 2404.12257](https://arxiv.org/abs/2404.12257); [CVF](https://openaccess.thecvf.com/content/CVPR2024W/MTF/html/Vinod_Food_Portion_Estimation_via_3D_Object_Scaling_CVPRW_2024_paper.html)

**NutritionVerse family** (Waterloo VIP lab, Alexander Wong):
- NutritionVerse-3D (Tai et al., arXiv 2304.05619, Apr 2023) has 105 high-resolution 3D food models with weight, food name and nutrition, hosted on Kaggle. Secondary. — [arXiv 2304.05619](https://arxiv.org/abs/2304.05619)
- NutritionVerse empirical study (arXiv 2309.07704, v2 Sep 2024; no venue on page): NV-Synth has 84,984 rendered images from roughly 7,082 scenes, 105 items across 45 food types, with depth and masks, rendered in Omniverse Isaac Sim. NV-Real has 889 iPhone 13 Pro Max images of 251 dishes.
- The study compares indirect methods (Mask2Former, Mask R-CNN, UOAIS-Net, scaled by pixel count times per-pixel nutrient average) with direct Inception-ResNet regression. Results are as in Q1. Data is on [Kaggle](https://www.kaggle.com/nutritionverse/datasets). **No GitHub code found.** Verified. — [arXiv 2309.07704 HTML](https://arxiv.org/html/2309.07704v2)

**Depth domain adaptation** (Vinod, Shao, Zhu, MIPR 2022) and **DPF-Nutrition** (Foods 2023): covered in Q1 Family B (secondary).

**Graikos food_volume_estimation:** covered in Q1 Family B (verified README, no metrics).

**Point2Volume:** covered in Q1 Family E (secondary).

**PortionNet** (Bright, Raj, Keisham, VIT; arXiv 2512.22304, Dec 2025). Verified from HTML:
- Inference uses RGB only, with ViT-B/16 and ResNet-18 features. The training teacher is PointNet with **bounding-box dimension embeddings** (R³ to R⁶⁴) for explicit scale, plus bidirectional cross-attention.
- The student adapter is distilled with MSE + cosine + KL losses. Training alternates 70% multimodal and 30% RGB-only batches. Heads: 108-class classification, volume and energy.
- MetaFood3D (MFP3D 80/20 split, 12,752 images): volume MAE 25.39 ± 2.59 mL, **MAPE 17.43 ± 0.81%**; energy 15.36%; R² 0.926. Mean ± std over 3 seeds.
- SimpleFood45 transfer: volume MAPE 23.51%, worse than 3D-Assisted's 14.01%.
- No code link.
- Source: [arXiv 2512.22304](https://arxiv.org/html/2512.22304v1)

**Implicit-Scale 3D Reconstruction for Multi-Food Volume Estimation from Monocular Images** (Chen, Vinod, ..., Zhu, He; IEEE SSIAI 2026; arXiv 2602.13041). This is a benchmark and challenge from the MetaFood 2025 workshop. Food items are drawn from the MetaFood3D object set and placed in multi-food scenes with no explicit metric references, only implicit cues (plates, utensils).
- The top approach reaches **0.21 MAPE (21%) volume** and 5.7 L1 Chamfer.
- "Vision-language baselines performed competitively, but geometry-based methods were more accurate and robust."
- Kaggle-hosted; no code. Verified (abstract).
- Overlaps with the MetaFood challenge material another researcher covers.
- Source: [arXiv 2602.13041](https://arxiv.org/abs/2602.13041)

**MetaFood CVPR 2024 challenge report** (arXiv 2407.09285): checkerboard-referenced 3D food reconstruction. Writeups are out of scope here. — [arXiv 2407.09285](https://arxiv.org/abs/2407.09285v1)

### Inferences
- **The Size Matters target and loss map almost one-to-one onto the competition.** They predict a scale ratio and use a relative (1/y-normalized) L1 loss, which approximates a percentage-error objective. A competition analogue: render or reconstruct a unit-normalized shape, predict log(volume) or log(scale), and train with a relative loss suited to sMAPE.
- The competition provides the true training meshes, so V_reconstructed is computable exactly from the normalized mesh in training. At test time a reconstruction (or a learned shape-volume proxy) is still needed.
- Because the object always fills about 20% of the frame, Size Matters' "f(x) only" ablation (76% MAPE on MetaFood3D) is closest to a plain image-to-volume regressor on these renders. The pairing with render features helped mainly by separating shape from scale.
- MFP3D's ablation shows that scale-normalized GT point clouds plus RGB (42.19%) perform about as well as depth point clouds plus RGB (41.43%) on MetaFood3D. This suggests most of MFP3D's accuracy comes from the RGB/appearance branch learning a category-and-size prior, not from metric geometry. That fits the competition's no-scale-cue setting.
- **Splits matter.** Size Matters' random split (82.46% MAPE) is much worse than its stratified split (35.83%). The competition test is a random split with the same categories, so expect published stratified numbers to be optimistic.
- The MetaFood3D numbers are mutually comparable only roughly. MFP3D and PortionNet use 510/127; Size Matters uses 535/102. PortionNet uses about 20 images per object (12,752 images / 637 objects), so per-image versus per-object evaluation may differ. That is an inference I have not verified.

### Gaps
- Whether MetaFood3D has appeared at a peer-reviewed venue: arXiv lists none, and none was found.
- PortionNet's large improvement (17.4% vs 41.4%) has no independent reproduction or code. Whether it evaluates per image with multiple views of test objects was not confirmed.
- No 2024–2026 paper other than Size Matters reporting OmniObject3D volume prediction was found. The search was not exhaustive.

---

## Q3. Methods designed for "single image, no reference, no intrinsics", and their scale tricks

### Takeaway
Only MFP3D, Size Matters and PortionNet target a single image with no reference and no intrinsics on scanned-food data. All three recover scale by **learning it from appearance features plus supervised GT volume**, not by geometry. Other tricks seen: category-mean calibration, bounding-box size embeddings as a training-only teacher signal, portion-independent factorization (energy density times mass/volume), and multi-view render features of a scale-free reconstruction.

### Cited Findings
- **Size Matters, predicted scale factor:** v_scale = V_gt / V_recon, regressed from paired CLIP features of the input image and renders of a scale-free One-2-3-45 mesh. The mesh is rescaled by the cube root of the prediction. Relative L1 loss. No reference or intrinsics are used by the method itself. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
- **MFP3D, implicit scale through regression:** depth-derived point cloud plus RGB feature, regressed directly to volume or energy. The authors say monocular reconstruction "cannot obtain actual size". — [MFP3D HTML](https://arxiv.org/html/2411.10492v1)
- **PortionNet, bounding-box dimension embeddings:** used only in the training-time PointNet teacher and distilled into an RGB-only student, so metric size information enters during training only. — [PortionNet HTML](https://arxiv.org/html/2512.22304v1)
- **Category priors:** the "Category Mean" baseline and the category-mean calibration in Depth Reconstruction (both in Size Matters) use GT category statistics. Category mean alone gives 87.46% MAPE on OmniObject3D, close to the best 85.96%. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
- **Energy density / portion-independent factorization:**
  - Nutrition5k predicts calories per gram from RGB at 9.5% MAE, compared with 26.1% for absolute calories. — [Nutrition5k](https://arxiv.org/abs/2103.03375)
  - Size Matters converts volume to energy using FNDDS density. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
  - The depth-domain-adaptation paper predicts pixel-wise energy density maps (secondary). — [arXiv 2208.12153](https://arxiv.org/abs/2208.12153)
- **Implicit contextual cues:** the 2026 multi-food benchmark removes explicit references but keeps plates and utensils, so scale must come from those plus prior knowledge. — [arXiv 2602.13041](https://arxiv.org/abs/2602.13041)
- **User-supplied priors (not image-based):** Graikos requires camera FoV and an expected camera-food distance, or a plate diameter. — [Graikos repo](https://github.com/AlexGraikos/food_volume_estimation)

### Inferences
- Under auto-framing, apparent pixel size is constant. That leaves three scale signals:
  - category identity (prior mean and variance);
  - within-category appearance cues that correlate with size, such as texture granularity, seed and crumb size, skin pores, and specular highlight scale relative to object extent;
  - shape and aspect ratio. Shape correlates with size in some categories, for example a whole fruit versus a slice.
- A sensible model structure is therefore log V = category prior + learned residual from appearance and shape features, trained with a relative or log loss. This is my inference, consistent with Size Matters' ablations.
- MetaFood3D weight_g offers a further training-only auxiliary target (mass ≈ density × volume), similar to PortionNet's training-only signals. Weight is unavailable at test time, so it can only regularize training.

### Gaps
- No published method uses *renderer framing statistics* (auto-framing) explicitly. The competition's fixed 20% fill is unusual, and no literature addresses it directly.
- No paper found evaluates texture-scale cues (for example frequency analysis) for absolute food size.

---

## Q4. Recent (2024–2026) VLM/LLM portion and volume estimation, and accuracy

### Takeaway
Off-the-shelf VLMs (GPT-4V, GPT-4o, GPT-5.x) do poorly at absolute food volume without references. On scanned single objects, Size Matters measured GPT-4o at 154–170% volume MAPE, improving to 63% on MetaFood3D with context. Dietary-assessment studies report better agreement, but only for simple items or with scale cues. Context and metadata prompting helps substantially.

### Cited Findings
- GPT-4o on MetaFood3D renders/images: volume MAE 84.95 mL, MAPE 153.80%, r 0.58; with context 83.65 mL, 63.25%. On OmniObject3D: 169.81%, or 155.28% with context. Verified. — [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
- "Dietary Assessment with Multimodal ChatGPT: A Systematic Analysis" (Lo et al., Imperial; arXiv 2312.08592, Dec 2023, preprint): GPT-4V reaches up to 87.5% food recognition and uses "surrounding objects as scale references" for portions. A search summary reports a portion MAE of 54.6 g and r = 0.64 against human estimates. The MAE and correlation are secondary; they are not in the abstract. — [arXiv 2312.08592](https://arxiv.org/abs/2312.08592)
- ACETADA benchmark (Coburn, He, ..., Zhu; IEEE BHI 2025 extended, arXiv 2507.07048) evaluates 8 LMMs (4 open, 4 closed) on dietitian-verified nutrition and portion size. Adding contextual metadata (GPS venue type, timestamp meal type, food items present) significantly reduces MAE and MAPE compared with image-only prompting. The gain holds across CoT, multimodal CoT, "Scale Hint", few-shot and expert-persona prompts. Exact numbers are not on the abstract page. Data: skynet.ecn.purdue.edu/~coburn6/ACETADA/. — [arXiv 2507.07048](https://arxiv.org/abs/2507.07048)
- Multi-food implicit-scale benchmark (SSIAI 2026): "vision-language baselines performed competitively, but geometry-based methods were more accurate and robust". — [arXiv 2602.13041](https://arxiv.org/abs/2602.13041)
- The following VLM results are secondary (search summaries only) and come from nutrition-journal or non-CV venues:
  - An MDPI *Nutrients* benchmark of open-source VLMs reports GPT-4o as best, at about 58 g NMAE per food and 157 g per meal, with weak correlation (r ≈ 0.13–0.25). — [Nutrients 18(21):3582](https://www.mdpi.com/2072-6643/18/21/3582)
  - OmniFood-Bench (arXiv 2607.08423, 2026) reports that even the best models struggle to reach portion MAPE below 50% in complex categories, attributed to scale ambiguity without references. — [arXiv 2607.08423](https://arxiv.org/pdf/2607.08423)
  - Customized Diabot-GPT-4o, validated against weighed food records in Taiwan, gave portion estimates within ±10–15%. — [PubMed 41138916](https://pubmed.ncbi.nlm.nih.gov/41138916/)

### Inferences
- On the competition's reference-free, auto-framed renders, a zero-shot VLM would mostly supply a category-typical size. It is likely useful as a prior or as a feature (for example "typical volume of a whole X"), not as a final predictor. Size Matters' GPT-4o numbers are in that range.
- VLM size priors could replace or smooth category means for rare categories with few training objects. I have not tested this.

### Gaps
- No found paper fine-tunes a VLM for volume regression on MetaFood3D or OmniObject3D.
- MDPI, OmniFood-Bench and ChatGPT MAE numbers were not read in full; the MDPI page returned 403.

---

## Q5. Datasets with mesh/volume ground truth usable as priors, and leakage flags

### Takeaway
Mesh or volume GT exists in MetaFood3D (743 objects / 131 categories), OmniObject3D (6,000 objects / 190 categories, CVPR 2023), NutritionVerse-3D (105 food meshes) and SimpleFood45 (45 items with volume, weight and energy). Nutrition5k has mass but not volume (overhead depth only). **External copies of MetaFood3D and OmniObject3D (and anything derived from them, such as the 2026 multi-food benchmark, or Size Matters / MFP3D checkpoints trained on them) would leak competition test objects and must not be used as extra training data or for pretrained weights.**

### Cited Findings
- **MetaFood3D:** 743 objects across 131 categories with textured meshes, RGB-D videos, masks, weight, nutrition and food codes. Papers use a 637-object / 108-category subset. — [arXiv 2409.01966](https://arxiv.org/abs/2409.01966); [MFP3D HTML](https://arxiv.org/html/2411.10492v1)
- **OmniObject3D** (Wu et al.; CVPR 2023 award candidate per repo): 6,000 real-scanned objects in 190 daily categories, with textured meshes, point clouds, multiview renders and videos, CC BY 4.0. Neither the arXiv page nor the README states mesh units, which fits the competition's note about unit errors. Size Matters used 3,417 OmniObject3D models. — [arXiv 2301.07525](https://arxiv.org/abs/2301.07525); [repo](https://github.com/omniobject3d/OmniObject3D); [Size Matters HTML](https://arxiv.org/html/2601.20051v1)
- **NutritionVerse-3D:** 105 3D food models with weight and nutrition (Kaggle). They are the source meshes for NV-Synth's 84,984 images. NV-Real has 889 images of 251 dishes with weighed ingredients. — [arXiv 2304.05619](https://arxiv.org/abs/2304.05619); [arXiv 2309.07704](https://arxiv.org/html/2309.07704v2)
- **SimpleFood45:** 45 food items with volume, weight and energy. — [arXiv 2404.12257](https://arxiv.org/abs/2404.12257)
- **Nutrition5k:** 5,006 plates with mass, calories and macronutrients, overhead RGB-D and side videos. There is no mesh, but a depth-derived volume can be computed from the known 35.9 cm camera height. — [repo](https://github.com/google-research-datasets/Nutrition5k); [arXiv 2103.03375](https://arxiv.org/abs/2103.03375)
- **Multi-food implicit-scale benchmark (2026):** built from MetaFood3D objects, so it is a **leakage risk**. — [arXiv 2602.13041](https://arxiv.org/abs/2602.13041)
- **ACETADA:** dietitian-verified nutrition and portion labels (real meal photos). — [arXiv 2507.07048](https://arxiv.org/abs/2507.07048)

### Inferences
- **Leakage flags:**
  - Do not use downloaded MetaFood3D or OmniObject3D, the MetaFood 2024/2025 challenge data (built on MetaFood3D objects), or released checkpoints from MFP3D, Size Matters or PortionNet trained on those sets as training data or initialization. They may contain the 851 test objects.
  - Generic pretrained encoders (CLIP, DINO), generic image-to-3D models, and depth models such as One-2-3-45, TripoSR, ZoeDepth and Metric3D are not food-specific. Their pretraining sets (for example Objaverse) could include OmniObject3D-like scans. That deserves a check but is a weaker concern. This is my inference; I did not verify their training data.
- Leakage-safe external **priors** could come from:
  - NutritionVerse-3D meshes (105 items; check category overlap);
  - SimpleFood45 volumes;
  - published food density tables (FNDDS, FAO/INFOODS), used to relate MetaFood3D weight_g to volume;
  - VLM or common-knowledge category size priors.
- Adding external objects to training is low-value, because the competition categories already match the test set. Category-level summary statistics are the safer use.
- Published MetaFood3D results train on about 510–535 objects. The competition's 2,042-object training set is larger and mixes both sources, so published per-dataset MAPEs are a rough guide at best.

### Gaps
- Whether OmniObject3D meshes are in metric units by default: no statement found. The competition reports roughly 10% of labels oversized by about 1000x, which suggests mixed mm/m conventions. That is an inference.
- ECUSTFD, UNIMIB2016 and other older food-volume datasets (coin/plate references, water-displacement volumes) were not verified in this session.

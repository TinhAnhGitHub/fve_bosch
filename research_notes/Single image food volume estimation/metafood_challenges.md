# MetaFood Workshop Challenges (CVPR 2024/2025/2026) and Related Food-Volume / Nutrition Benchmarks: What Top Solutions Did, and Lessons for Single-Image, No-Reference Volume Regression

Scope note: research current to early November 2026. Target setting: predict `volume_cm3` from one 1024x1024 Blender render of one food item. The view is canonical and auto-framed, so the object always fills about 20% of the frame. There are no intrinsics and no reference object. Training data: MetaFood3D plus OmniObject3D meshes, category, bbox_mm and weight_g. Metric: sMAPE. "Fact" bullets carry a source. "Inferences" are my interpretation.

---

## Q1. MetaFood CVPR 2024: "Physically Informed 3D Food Reconstruction" (report arXiv 2407.09285), the winning teams, their single-image (hard level) methods, VolETA, and repos

### Takeaway
In 2024 the size reference was explicit: a 1.2 cm checkerboard was in every image and dataset depth maps were also available. All top teams recovered metric scale from the checkerboard, often refined with depth, and used neural or Gaussian surface reconstruction for multi-view objects. For the single-image hard level, teams used zero-shot image-to-3D models (One-2-3-45, ZeroNVS, LGM, InstantMesh) for shape only. Scale still came from the checkerboard and depth. None of the 2024 methods solves no-reference scale.

### Cited Findings
**Challenge setup**
- Task: reconstruct volume-accurate 3D food models from 2D images, using a visible checkerboard as the size reference. The data is 20 MetaFood3D food items, each 3D-scanned and filmed with a checkerboard and pattern mat — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Difficulty levels: Easy (8 objects, about 200 frames each), Medium (7 objects, 30 images), Hard (5 objects, objects 16–20, one monocular top-view image). Depth images were provided for each frame, and teams used them to refine scale — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Objects 12 (steak) and 15 (chicken nugget) were excluded after Phase I because of data-quality issues. 16 teams submitted in the final phase, and the report describes only the top 3 — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Metrics: Phase I is volume MAPE. Phase II is Chamfer distance after alignment with submitted transforms, for top Phase I teams only. The final rank combines both phases — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Challenge site: [sites.google.com/view/cvpr-metafood-2024](https://sites.google.com/view/cvpr-metafood-2024)

**1st place: VolETA (AIBA Lab, Universitat de Barcelona; AlMughrabi, Haroon, Marques, Radeva)**
- Pipeline:
  - Keyframes: near-duplicate removal (perceptual hash, Hamming threshold 12) and FFT-based blur rejection, keeping about 34.8% of frames.
  - Masks: a SAM click prompt, propagated with XMem++.
  - Poses: PixSfM with SuperPoint and SuperGlue.
  - Mesh: NeuS2 (15k iterations, 512 marching-cubes resolution), with removal of fragments smaller than 5%.
  - Scale: checkerboard blocks measured manually in MeshLab (0.012 m), refined with depth plus food and reference masks.
  — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Single-view (hard): One-2-3-45 on the RGBA segmented image, cleanup, then a reused, previously chosen scale factor. Stated limitations are manual prompts and manual scale identification, and the need for depth and masks — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Reported results: Phase I MAPE 0.10973 (10.97%). Phase II Chamfer averages 0.00726 with the transformation matrix and 0.09529 without — [GCVCG/VolETA-MetaFood README](https://github.com/GCVCG/VolETA-MetaFood)
- Per-object example: banana_2 predicted 153.76 cm³ vs ground truth 163.09 cm³ — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- VolETA paper (arXiv 2407.01717): "one- and few-shot food volume estimation" from one or a few RGBD images, reporting 10.97% MAPE on the MTF dataset. I did not fetch the paper directly; this is from the VolTex paper's related-work description and the repo — [VolTex arXiv 2506.02895](https://arxiv.org/html/2506.02895v1), [VolETA repo](https://github.com/GCVCG/VolETA-MetaFood)
- Code: [github.com/GCVCG/VolETA-MetaFood](https://github.com/GCVCG/VolETA-MetaFood) (verified HTTP 200; GPL-3.0; submodules PixSfM, SAM, XMem++, NeuS2). The README describes no `main.py` steps.
- Follow-up: VolETA++, a "fully automatic depth-free radiance-field framework", CVPRW 2026 pp. 9482–9491. It uses FoodMem masks, NeuS2 or Instant-NGP, and checkerboard scaling with corner-detector binarisation threshold 240. The authors note this threshold is sensitive, with some scenes needing 210–220 — [GCVCG/VolETAPP](https://github.com/GCVCG/VolETAPP)

**2nd place: ININ-VIAUN**
- Scale method ("HR-CMS"): COLMAP dense poses and point cloud, corner detection (threshold 240), projection into the images. Scale = 0.012 / mean of per-image median nearest-corner distances — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Multi-view: COLMAP plus DiffusioNeRF and NeRF2Mesh, keeping the best result; then MeshFix and Laplacian smoothing (λ=0.2, 10 iterations); alignment by centroid, then ICP, then gradient descent — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Single-view (objects 16–20): compared Zero123, Zero123++, One2345, ZeroNVS and DreamGaussian, and chose ZeroNVS. Intrinsics were borrowed from object 15 and extrinsics refined by reprojection error. Scale came from dataset depth plus the checkerboard. Reported single-image errors include object 16 at 32.77% and object 18 at 34.01% — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Code: [github.com/BITyia/cvpr-metafood](https://github.com/BITyia/cvpr-metafood) (verified HTTP 200; no licence stated, per [VolETAPP README](https://github.com/GCVCG/VolETAPP)). VolETA++ adapted this team's `scale_estimate/find_scale.py`.
- Unit inconsistency: the report's table lists volumes in mm³ while the caption says cm³ — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)

**Best 3D mesh award: FoodRiddle**
- Multi-view: COLMAP with SuperPoint and SuperGlue (for weakly textured food such as bread), then 2D Gaussian Splatting with depth-distortion and normal-consistency regularisation. Next: TSDF, outlier removal, projection of lower vertices onto the support-plane contour, checkerboard scale, and Poisson reconstruction to a watertight mesh — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Single-view (hard): LGM, InstantMesh and One-2-3-45 generate a prior mesh, which is jointly corrected with depth. Scale comes from the checkerboard, assuming the object and board share a plane. The model is reprojected into the image to check scale — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Reported Phase II Chamfer totals: multi-view 0.036362 and single-view 0.019232, the lowest of the three. Comparable totals are ININ-VIAUN 0.041552 / 0.027889 and VolETA 0.071921 / 0.058726 — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Code: [github.com/jlyw1017/FoodRiddle-MetaFood-CVPR2024](https://github.com/jlyw1017/FoodRiddle-MetaFood-CVPR2024) (verified HTTP 200)

### Inferences
- The mapping of the brief's three affiliations is my inference, and I did not confirm it from the report author list:
  - Universitat de Barcelona = VolETA (confirmed by the repo).
  - Beijing Institute of Technology = probably ININ-VIAUN, given the `BITyia` GitHub owner.
  - Baidu/XPeng/BUPT = probably FoodRiddle.
- On the hard level, image-to-3D shape was paired with an external metric anchor (checkerboard plus depth). Single-image errors of about 30% persisted even with that anchor (ININ-VIAUN objects 16 and 18). Generated geometry alone (back side, thickness) is therefore a large error source even when scale is known.
- For the target task there is no checkerboard and no depth, so 2024 scale methods do not transfer. The transferable piece is the shape side: segmentation, then a single-view generator (One-2-3-45, InstantMesh, LGM, ZeroNVS), then mesh cleanup and watertight volume. That gives a scale-free volume, or a shape descriptor, that a learned scale model can use.

### Gaps
- Full 16-team leaderboard and per-team Phase I MAPE (other than VolETA's 10.97%): not published in the report.
- VolETA arXiv 2407.01717 was not fetched directly, so per-level (hard/single-image) MAPE for VolETA is unconfirmed.
- Exact affiliations of ININ-VIAUN and FoodRiddle were not confirmed from a primary source.

---

## Q2. MetaFood CVPR 2025: tasks, the "Implicit-Scale" benchmark (arXiv 2602.13041), top methods, MAPE, and implicit-scale inference

### Takeaway
2025 Challenge 1 removed all explicit references. Each multi-food scene came as one monocular image, and scale had to come from implicit cues: plates, utensils and known food sizes. Every reported method used Hunyuan3D for shape and differed mainly in how it recovered scale. Results:
- Winner (Hillo World, "MDMS", MonoBite): MAPE 0.21. It aligned the generated mesh to a Depth Pro metric point cloud, then applied a global correction from web-crawled plate and utensil size statistics.
- Runner-up (OneVol, Universitat de Barcelona): MAPE 0.31, using plate and utensil known dimensions.
- Single plate-pixel-diagonal heuristic: worst, at MAPE 0.46.
- GPT-5.2 VLM baseline (no geometry): MAPE 0.34.

### Cited Findings
**Challenge 1: 3D Reconstruction From Monocular Multi-Food Images**
- Task: reconstruct 3D food models in real-world units from a single 2D eating-occasion image of a multi-food scene, with no explicit physical reference. The organisers recommend an image-to-3D generator for the mesh plus a metric depth model to calibrate size. Scale must be inferred from plates, utensils or foods of known size — [MetaFood 2025 Challenge 1](https://sites.google.com/view/cvpr-metafood-2025/challenge-1)
- Scoring: Phase I volume MAPE (55%). Phase II L1 Chamfer per DTU protocol (45%), for the top 3–5 teams. The final score is a weighted sum of per-phase ranks. Opened April 8, 2025; winners announced May 11, 2025 — [MetaFood 2025 Challenge 1](https://sites.google.com/view/cvpr-metafood-2025/challenge-1)
- Winners:
  1. Team Hillo World (Songen Gu, Lina Liu, Binjie Liu, Lei Yang)
  2. Team OneVol (AlMughrabi, Haroon, Marques, Radeva)
  3. Team wuhu648648 (Shipeng Liu, Xiangji Kong, Runfeng Lv, Yatao Xu, Wei She)

  Reports and videos are on Google Drive, and no numeric scores are on the page — [MetaFood 2025 Challenge 1](https://sites.google.com/view/cvpr-metafood-2025/challenge-1)
- Kaggle: [3d-reconstruction-from-monocular-multi-food-images](https://www.kaggle.com/competitions/3d-reconstruction-from-monocular-multi-food-images/data)

**Benchmark paper (arXiv 2602.13041)**
- Authors: Yuhao Chen, Gautham Vinod, Siddeshwar Raghavan, Talha Ibn Mahmud, Bruce Coburn, Jinge Ma, Fengqing Zhu, Jiangpeng He. The paper reframes portion estimation as implicit-scale 3D reconstruction from monocular views, with physical references and metric annotations removed and plates and utensils provided as context — [arXiv 2602.13041](https://arxiv.org/abs/2602.13041)
- Data: 10 curated multi-food scenes with 24 scanned objects, and a single image per scene — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1). This conflicts with the OneVol README, which says "14 food scenes with 34 food items" — [OneVol README](https://github.com/GCVCG/OneVol-MetaFood). The paper's numbers may describe a curated subset.
- Metrics: MAPE, and L1 Chamfer after manual 6D alignment, so pose error is excluded — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1)
- Results by method category:

  | Method | MAPE (std) | L1 Chamfer (std) |
  |---|---|---|
  | MDMS | 0.21 (0.16) | 5.70 (2.66) |
  | SGPS | 0.31 (0.25) | 7.07 (3.85) |
  | PSHS | 0.46 (0.25) | 11.89 (6.44) |
  | GPT-5.2 (VLM, no geometry) | 0.34 (0.18) | n/a |

  — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1)
- MDMS (metric-depth-driven multi-stage scaling; cites Gu et al., "MonoBite", PRCV):
  - Hunyuan3D reconstructs the whole scene as one mesh, and k-means clustering separates the objects in 3D.
  - Depth Pro supplies monocular metric depth.
  - A coarse-to-fine 6D alignment fits a similarity transform between mesh points and the depth point cloud, giving a local scale.
  - Web-crawled plate and utensil size statistics give a global correction factor.
  - Final scale = local × global.

  — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1); MonoBite citation: Songen Gu, Lina Liu, Binjie Liu, Sanyi Zhang, Lei Yang, Yanwei Fu, "MonoBite: Scale-Aware 3D Reconstruction and Volume Estimation from Monocular Multi-food Images" — [researchr](https://researchr.org/publication/GuLLZYF25)
- SGPS: reconstructs food, plates and utensils together, and derives one global scale from known plate and utensil dimensions — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1)
- PSHS: segments and reconstructs each food separately, with scale = assumed real plate diameter / the plate's pixel-space diagonal — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1)
- Paper lessons:
  - "leveraging multiple reconstructed reference objects yields more reliable implicit scale estimation"
  - "pixel-based scaling relies on bounding-box measurements that are sensitive to bounding box noise and object orientation"
  - geometry-based methods give "higher accuracy and greater robustness"
  - GPT-5.2 is competitive on volume but produces no geometry

  — [arXiv 2602.13041 HTML](https://arxiv.org/html/2602.13041v1)
- OneVol (2nd place): FoodMem segmentation plus Hunyuan3D-2 (with rembg and trimesh/pymeshlab tooling). Reported Phase I MAPE 0.31 and Phase II Chamfer 7.0716 — [GCVCG/OneVol-MetaFood](https://github.com/GCVCG/OneVol-MetaFood) (repo verified via GitHub API). The README does not detail the scale step.

**Challenge 2 (2025): "Dishcovery" VLM image–caption matching**
- 400k food image–caption pairs, not volume estimation. Winners: SunnyLX, Desayuno, e0nia. No methods or scores are on the page — [MetaFood 2025 Challenge 2](https://sites.google.com/view/cvpr-metafood-2025/challenge-2)

### Inferences
- OneVol's numbers (0.31 / 7.07) exactly match the "SGPS" row, so SGPS is very likely OneVol: a global scale from plate and utensil known dimensions. MDMS is very likely Hillo World (MonoBite): the paper cites Gu et al., and Gu leads Hillo World. PSHS is plausibly third-place wuhu648648 or another team; unconfirmed.
- Ranking of scale sources in 2025, best to worst:
  1. Metric depth fused with multiple context-object priors.
  2. Multiple known-size context objects.
  3. A single plate-pixel heuristic.
- That still meant about 21% MAPE on only 24 objects, with high variance (std 0.16).
- GPT-5.2 at 0.34 MAPE suggests VLM world knowledge of typical food sizes is a meaningful prior, worse than geometry with context references but better than a naive heuristic. This matters for the target task, where no context references exist and prior knowledge is the main scale source.
- In the target task, 2025-style scale recovery cannot work: there are no plates or utensils, and auto-framing removes apparent size. Metric depth on a render with no intrinsics and a normalised framing is also unlikely to carry absolute size; this is my inference and was not tested in any source.

### Gaps
- The Hillo World and wuhu648648 technical reports (Google Drive links) were not read, so their exact details (Depth Pro settings, plate and utensil statistic values) are unverified beyond the benchmark paper.
- The MonoBite PRCV abstract and text were not found online; no code repos for Hillo World or wuhu648648 were found.
- The Kaggle private leaderboard numbers were not retrieved.

---

## Q3. MetaFood CVPR 2026: eating/bite challenge (PerBite, arXiv 2606.02021), GCVCG repos (VolTex etc.), other 2026 work

### Takeaway
2026 Challenge 1 ("Continuous 3D Reconstruction While Eating") used egocentric eating videos with no explicit references. Top teams kept the same recipe as 2025:
- open-vocabulary or SAM-family segmentation;
- a generative watertight mesh (Hunyuan3D or SAM 3D);
- metric scale from plate-size priors, with MoGe-2 or Pi3 metric geometry as a backup or a second path.

Winner PerBite reports 33.87% state-level volume MAPE and 53.74% consumed-volume MAPE. The most transferable 2026 item is not the challenge: it is "Size Matters" (arXiv 2601.20051, Purdue). That paper learns a scale factor from CLIP features of the input image plus renders of a scale-free reconstruction, using exactly MetaFood3D and OmniObject3D (see Q5).

### Cited Findings
**Challenge setup**
- Task: reconstruct evolving 3D food models from monocular egocentric eating videos, with hand and utensil occlusion. Scale must be inferred from plates, utensils or foods of known size. Recommended approach: image-to-3D generator plus metric depth model. Ground truth is 3D-scanner models — [MetaFood 2026 Challenge 1](https://sites.google.com/view/cvpr-metafood-2026/challenge-1)
- Scoring:
  - Phase I (55%): the equal-weighted average of MAPE on before-eating volume and MAE on percentage consumed.
  - Phase II (45%): L1 Chamfer (DTU protocol) for the top 3–5 teams.
  - Dates: opened March 22, 2026; winners announced May 14, 2026.
  - Prize: USD 1,500, sponsored by Cronometer.

  — [MetaFood 2026 Challenge 1](https://sites.google.com/view/cvpr-metafood-2026/challenge-1)
- Winners: 1st PerBite (UB/GCVCG). 2nd weiwei11123 (Northwestern Polytechnical University), "CrossRecon". 3rd is not listed on the page — [MetaFood 2026 Challenge 1](https://sites.google.com/view/cvpr-metafood-2026/challenge-1); Kaggle: [mtf-challenge-1-continuous-3-d-reconstruction-while-eating](https://www.kaggle.com/competitions/mtf-challenge-1-continuous-3-d-reconstruction-while-eating/overview)

**PerBite (1st)**
- Pipeline:
  - SAM 3 segments the food and the plate.
  - Hunyuan3D or SAM 3D generates a dimensionless food mesh.
  - Plate diameter gives the metric scale.
  - Blender removes the plate; then hole-filling, watertight repair and volume integration.
  - MoGe-2 is used only as a backup dish-diameter estimator when plate measurement is uncertain.

  — [arXiv 2606.02021](https://arxiv.org/abs/2606.02021)
- Results:
  - Average Chamfer 8.31 over 34 meshes (rigid ICP, no scale correction).
  - 33.87% state-level volume MAPE on 17 before/after pairs, with zero monotonicity violations.
  - 53.74% consumed-volume MAPE.

  The authors argue that reconstruction quality, metric scale, mesh cleanup, watertight integration and consumption consistency should be evaluated separately — [arXiv 2606.02021](https://arxiv.org/abs/2606.02021)
- Code link given on the challenge page and in the paper: `github.com/GCVCG/PerBite-CVPR-MetaFood-2026`. It returned HTTP 404 on 2026-11-04 and is not among the public GCVCG repos (checked via the GitHub API). The paper calls the code "forthcoming".

**CrossRecon (2nd, Northwestern Polytechnical University)**
- Path A ("real observation"): GroundingDINO plus SAM2 masks, then Pi3 feed-forward sparse-view geometry, plate/support-plane alignment, and a height-field mesh.
- Path B ("generative"): GPT Image synthesises extra views from a crop, and Hunyuan3D builds a watertight mesh.
- Fusion: V = α·V_A + (1−α)·V_B, where α grows with Path A visibility and with the after-eating state (generative priors over-complete eaten food), plus a monotonicity guard.
- Scale from plate and table-object priors.
- Reported Phase I composite score 0.37 and Phase II Chamfer 8.39.

— [WZ-CS/CVPR-2026-3rd-MetaFood-Workshop-challenge-2nd-Place-CrossRecon](https://github.com/WZ-CS/CVPR-2026-3rd-MetaFood-Workshop-challenge-2nd-Place-CrossRecon) (README fetched via GitHub API; repo exists)

**Other GCVCG (UB) repos and papers (all checked to exist)**
- VolTex (arXiv 2506.02895): text-guided segmentation (the user types a food name), NeRF reconstruction, and voxelised volume, on MetaFood3D/MTF — [arXiv 2506.02895](https://arxiv.org/html/2506.02895v1), [GCVCG/VolTex](https://github.com/GCVCG/VolTex)
- Other public GCVCG repos (descriptions from the GitHub API): VolETAPP (VolETA++, CVPRW 2026); VolE ("A Point-cloud Framework for Food 3D Reconstruction and Volume Estimation"); FoodMem (near-real-time food segmentation); BenchSeg (multi-view food video segmentation benchmark); RASS (NeurIPS 2026, per the VolETAPP README); and OneVol-MetaFood — [api.github.com/users/GCVCG/repos](https://api.github.com/users/GCVCG/repos), [VolETAPP README](https://github.com/GCVCG/VolETAPP)

**2026 Challenge 2: "Dishcovery: Mission II"**
- VLM image–caption matching, with paid proprietary models forbidden. Not a volume task — [MetaFood 2026 Challenge 2](https://sites.google.com/view/cvpr-metafood-2026/challenge-2)

### Inferences
- From 2024 to 2026 the MetaFood community converged on one recipe: foundation segmentation (SAM 2/3, GroundingDINO, FoodMem), then a generative mesh (Hunyuan3D-2, SAM 3D), then explicit scale from a context object (plate) and/or metric depth (Depth Pro, MoGe-2, Pi3). Even so, volume MAPE stayed around 21–34% on real images with context cues.
- Generative mesh priors bias toward "complete" objects (CrossRecon's after-state over-completion; PerBite's hole-fill and watertight steps). Volume errors from shape hallucination are therefore not negligible even when scale is right.

### Gaps
- PerBite code is not public (404). The 2026 3rd place and Kaggle numeric leaderboard were not found.
- A 2026 MetaFood challenge report (an analogue of 2407.09285 or 2602.13041) was not found.

---

## Q4. Other related competitions and benchmarks (Nutrition5k, Kaggle, ACM MM, FGVC, physical-quantity regression with MAPE): what winners did

### Takeaway
Nutrition5k is the canonical single-image food-quantity benchmark. Its results show that portion size, not food identity, is the bottleneck:
- Per-gram calorie density is predicted with 9.5% error, but absolute calories from RGB alone have 26.1% error.
- Adding depth, or a volume estimate, cuts the error to 16.5–18.8%.

I found no Kaggle or other leaderboard competition that matches the target task (single-image volume regression with sMAPE/MAPE). Image-regression Kaggle competitions such as CSIRO Image2Biomass standardise log targets and DINOv2/v3 backbones, but winner details were not verifiable.

### Cited Findings
**Nutrition5k (Thames et al., CVPR 2021)**
- Data: about 5k real dishes, with overhead RGB-D (Intel RealSense D435), side-angle videos, a plate scale for weights, and a 10% test split — [Nutrition5k paper](https://arxiv.org/pdf/2103.03375)
- Model: InceptionV2 backbone with multi-task heads (calories, mass, macronutrients), MAE losses, and a weighted multi-task loss — [Nutrition5k paper](https://arxiv.org/pdf/2103.03375)
- Test results, MAE / % of mean (Table 3):

  | Model | Calories | Mass |
  |---|---|---|
  | Mean-prediction baseline | 150.8 kcal / 60.2% | 124.6 g / 58.5% |
  | 2D portion-independent (per-gram) | 24.1 / 9.5% | – |
  | 2D direct prediction | 70.6 / 26.1% | 40.4 / 18.8% |
  | Depth as 4th channel | 47.6 / 18.8% | 40.7 / 18.9% |
  | Volume scalar input | 41.3 / 16.5% | 29.4 / 13.7% |

  — [Nutrition5k paper](https://arxiv.org/pdf/2103.03375)
- Code and data: [google-research-datasets/Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (verified HTTP 200)
- Later Nutrition5k claims are self-reported and not peer-reviewed. Examples: a ResNet-50 model card with an Atwater energy-consistency loss reports 17.5% calorie / 13.2% mass error, and a 2026 preprint "NutriVision" (RGB-D fusion) reports 13.60% mean percentage MAE. I found no authoritative RGB-only leaderboard — [NutriConsistNet card](https://huggingface.co/arunapb/nutriconsistnet), [search summary](https://chatpaper.com/chatpaper/paper/351773)

**CSIRO Image2Biomass (Kaggle, 2025–26; pasture biomass regression from images)**
- First place: Team 卷不动了 (China); CSIRO gave no architecture details — [CSIRO news, April 2026](https://www.csiro.au/en/news/All/News/2026/April/New-AI-tools-find-smarter-ways-to-measure-pasture)
- The dataset protocol transforms all five targets with log(1+y). Academic papers on the data use DINOv2-Giant or DINOv3 encoders; one reports +5.0 R² points from DINOv2→DINOv3 — [arXiv 2603.07819](https://arxiv.org/pdf/2603.07819), [search-surfaced papers](https://www.citedrive.com/en/discovery/dynamic-lora-fine-tuning-of-dinov3-for-multi-component-pasture-biomass-estimation/)

### Inferences
- The Nutrition5k ratios suggest that, for food, identity- or density-type quantities are learnable from appearance, while absolute size is the dominant residual. In the target task, auto-framing removes the main RGB size cue (apparent size). Error will therefore depend on how well category and appearance (texture granularity, shape) predict absolute size.
- The usual image-regression playbook (pretrained ViT features such as DINOv2/v3 or CLIP, log target, ensembling) is the reasonable default. But I found no verified winner write-up for a closely analogous sMAPE task.

### Gaps
- ACM MM food challenges and CVPR FGVC food tracks (iFood) were not researched in this pass. iFood tracks are classification, not regression (background knowledge; not verified here).
- No Kaggle calorie or food-volume competition with a sMAPE/MAPE metric and a published winning solution was found.
- No verified Kaggle winner write-up with backbones, losses, TTA or log targets for a physical-quantity image regression was retrieved. The Image2Biomass winner's method is unverified.

---

## Q5. Lessons: what worked and what failed for scale recovery without a reference object, and what transfers to the target task

### Takeaway
Every MetaFood winner recovered scale from some external metric anchor: a checkerboard (2024), plates and utensils plus metric depth (2025/26). The target task has no anchor, and its auto-framing deliberately removes apparent size. The closest published analogue is "Size Matters" (arXiv 2601.20051), which uses the same two source datasets (MetaFood3D plus OmniObject3D). It learns the scale from CLIP ViT-L/14 features of the input image plus renders of a scale-free One-2-3-45 reconstruction, trained with a relative (1/v-weighted) L1 loss. It reaches 35.83% MAPE on MetaFood3D, against 63.25% for GPT-4o and 104% for a 3D-assisted baseline. That paper is the most directly reusable reference.

### Cited Findings
**Size Matters (Vinod, Coburn, Raghavan, He, Zhu; Purdue/IU; arXiv 2601.20051, Jan 2026)**
- Pipeline: SAM-segmented input, then One-2-3-45 for an arbitrary-scale mesh, then renders (three polar angles). A frozen CLIP ViT-L/14 encodes the input and each render; the concatenated features go to an MLP that regresses the volume scale factor V_gt / V_recon. The prediction is averaged over views and applied as a cube root to the mesh. No depth model or VLM is used for scale — [arXiv 2601.20051](https://arxiv.org/html/2601.20051)
- Data and training:
  - MetaFood3D: 637 items (535 train / 102 test). OmniObject3D: 3,417 objects (2,763 / 654).
  - Loss: normalised L1, with each error weighted by 1/v_scale.
  - Schedule: 300 epochs, batch 64, lr 1e-4, decayed ×0.7 every 10 epochs.

  — [arXiv 2601.20051](https://arxiv.org/html/2601.20051)
- MetaFood3D results (MAE mL / MAPE):

  | Method | MAE (mL) | MAPE |
  |---|---|---|
  | GPT-4o with context | 83.65 | 63.25% |
  | 3D-assisted | 123.34 | 104.07% |
  | Size Matters | 59.09 | 35.83% |

  Pearson r 0.87, R² 0.75. On OmniObject3D: MAE 70.49 mL vs 91.38 mL for the next best — [arXiv 2601.20051](https://arxiv.org/html/2601.20051)
- Ablations:

  | Ablation | Result |
  |---|---|
  | Input + render features | 35.83% |
  | Input only | 76.10% |
  | Render only | 90.80% |
  | CLIP ViT-L/14@336 vs DeiT/ViT and smaller CLIP | CLIP L/14@336 best |
  | Renders: up to ~25 views | improved |
  | Renders: beyond ~25 views | plateau or worse |
  | One-2-3-45 vs TripoSR | MAE 59.09 vs 88.80 |
  | Random split: theirs vs RGB-only | 82.46% vs 299.52% MAPE |
  | Cross-dataset OmniObject3D→MetaFood3D | MAE 104.08 vs 174.57 mL |

  The paper has internal inconsistencies: 59.09 vs 61.24 MAE, 34.83% vs 35.83% MAPE, and a 75-render row at 38.42% — [arXiv 2601.20051](https://arxiv.org/html/2601.20051)
- Code: [gitlab.com/viper-purdue/size-matters](https://gitlab.com/viper-purdue/size-matters) (link from the paper; not independently opened)

**What worked or failed in the challenges**
- Worked: multiple context references over a single one. Pixel-bbox scaling is noise- and orientation-sensitive (PSHS 0.46 vs SGPS 0.31 vs MDMS 0.21) — [arXiv 2602.13041](https://arxiv.org/html/2602.13041v1)
- Worked: metric depth (Depth Pro) aligned to the generated mesh by a similarity transform, then corrected with size statistics (MDMS 0.21) — [arXiv 2602.13041](https://arxiv.org/html/2602.13041v1)
- VLM size priors: moderately competitive. GPT-5.2 scored 0.34 MAPE on the 2025 benchmark — [arXiv 2602.13041](https://arxiv.org/html/2602.13041v1). GPT-4o with context scored 63.25% MAPE on MetaFood3D single items — [arXiv 2601.20051](https://arxiv.org/html/2601.20051)
- Failure mode: generative priors over-complete partially eaten or occluded food; CrossRecon added a fusion weight and a monotonicity guard — [CrossRecon README](https://github.com/WZ-CS/CVPR-2026-3rd-MetaFood-Workshop-challenge-2nd-Place-CrossRecon). PerBite needed separate hole-fill and watertight steps, and found consumed-volume MAPE (53.74%) much worse than state MAPE (33.87%) — [arXiv 2606.02021](https://arxiv.org/abs/2606.02021)
- Failure mode: checkerboard corner-detection thresholds are fragile (240 vs 210–220 on some scenes) — [VolETAPP README](https://github.com/GCVCG/VolETAPP)
- Even with known scale, single-image reconstructions had about 30% per-object error in 2024 (ININ-VIAUN objects 16 and 18) — [arXiv 2407.09285](https://arxiv.org/html/2407.09285)
- Nutrition5k: an explicit volume estimate improved mass error from 18.8% to 13.7% — [Nutrition5k](https://arxiv.org/pdf/2103.03375)

### Inferences (my interpretation, for the target task)
1. **Scale is the whole problem, and auto-framing removes the usual cue.** With the object at about 20% of the frame regardless of size, apparent size carries little information. The remaining signals are:
   - category (strongest; the same categories appear in train and test);
   - within-category appearance cues of absolute size: texture granularity (seeds, crumbs, grain), surface detail density, specular highlight size relative to object, and rendering/defocus effects if present;
   - shape and aspect ratio, which correlate with size within some categories.

   A regressor on frozen foundation features (CLIP ViT-L/14@336 performed best in Size Matters; DINOv2/v3 is a common alternative) plus a category embedding or category prior is the analogue of the Size Matters scale regressor.
2. **Factor volume as "shape × scale", following Size Matters and Nutrition5k.**
   - Shape: predict a scale-free shape term, e.g. volume / bbox-volume fill ratio, or a normalised volume from a generated mesh (One-2-3-45 beat TripoSR there; Hunyuan3D-2 is the 2025/26 challenge favourite).
   - Scale: predict a size term separately, e.g. log of the bbox diagonal in mm, which bbox_mm in the training data allows as an auxiliary target.

   The training data's meshes and bbox_mm support multi-task learning (bbox dims, fill ratio, log volume), like Nutrition5k's multi-task heads. weight_g (MetaFood3D only) can support an auxiliary density/mass head. All of this is my inference.
3. **Loss and metric alignment.** Size Matters used a 1/v-weighted L1, i.e. a relative error that matches MAPE. For sMAPE, training on log-volume with L1, then possibly a post-hoc multiplicative calibration, is the analogous choice. This is my inference; no source tested sMAPE specifically.
4. **The about 1000× oversized OmniObject3D labels.** sMAPE is bounded at 200% per sample, so a conventional prediction for one of these costs nearly the full 200%. The two options:
   - Down-weight these samples or treat them as a separate mode in training, accepting the loss.
   - Add a classifier for "unit-error-prone" objects, using source dataset, category or other metadata cues, and predict ×1000 when confident. Since sMAPE is symmetric and bounded, a wrong ×1000 guess costs about as much as missing a true one; predict it only when P(oversized) > 0.5.

   Size Matters reports no handling of such outliers in OmniObject3D. This is a gap and my inference.
5. **VLM priors as a feature or fallback.** GPT-class models reach about 34–63% MAPE on food volume with context. They are useful as an auxiliary prior or ensemble member, but worse than learned regressors trained on in-distribution categories.
6. **Methods that do not transfer:** checkerboard, depth-map, plate/utensil and metric-depth (Depth Pro, MoGe-2) scale recovery. Without intrinsics and with normalised framing, metric depth's absolute scale on synthetic renders is unverified and likely unreliable.

### Gaps
- No source evaluates single-image food volume under strictly normalised framing (constant apparent size), so the size of the remaining within-category appearance signal is unknown.
- No source reports sMAPE-specific training choices for volume regression, or handling of mesh unit errors in OmniObject3D.
- Size Matters' random-split result (82.46% MAPE) is ambiguous in the extracted text: it may refer to a harder generalisation split. Check against the PDF before relying on it.

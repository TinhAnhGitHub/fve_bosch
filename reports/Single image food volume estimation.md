# Scale-blind renders reward priors, bins and decoding

No published method recovers absolute food scale from a single auto-framed, reference-free render. This competition is therefore decided by three things, not by the backbone: a well-pooled category prior, a distributional head over log-volume that is decoded with sMAPE's own Bayes rule, and an explicit, calibrated bet on the oversized OmniObject3D mode. Every MetaFood challenge winner from 2024 to 2026 took metric scale from an external anchor: a checkerboard, plates and utensils, or metric depth. The only paper trained on our two data sources, "Size Matters", learns scale from CLIP appearance features, and on OmniObject3D it barely beats a per-category mean (**85.96% vs 87.46% MAPE**) ([Size Matters](https://arxiv.org/html/2601.20051v1)). Two derived facts set the score range. Written in log space, sMAPE equals **2·tanh(|Δln V|/2)**, so a predictor that knows only the category floors at about **40% sMAPE** when the within-category log-std is 0.52. That matches our measured 41% in-sample on base-mode objects. Every object predicted in the wrong unit mode costs about **199%**. The recommended primary design has four parts: DINOv2-L/14-reg plus SigLIP 2 features with LoRA; a category-conditioned HL-Gauss head with 128 log-volume bins; a mode head with Beta-Binomial category priors and mesh-derived auxiliary targets; and decoding by argmin of expected sMAPE. Image features earn their place only if they explain within-category residuals. Weeks 1–2 must test, with a permutation null, whether the renders carry any scale cue at all. My estimate of a realistic private-test result is **33–42% sMAPE**. Going lower requires a scale cue that has not been shown to exist.

## Every published scale trick needs an anchor these renders remove

The literature splits cleanly into methods that get scale from geometry plus a metric anchor and methods that learn a scale prior from appearance. The competition removes every anchor: there are no intrinsics, no reference object and no plate, and auto-framing destroys apparent size (r≈0.01 between real and on-screen size). Only the second family transfers. The tables below map the field by group. "Transfers?" judges each item against the competition constraints. Venues marked * come from the researchers' prior knowledge and were not re-verified this session.

### Food volume and portion methods

| Venue, year | Method | Input → output | Scale handling | Key result | Code / licence | Transfers? |
|---|---|---|---|---|---|---|
| CVPR 2021 | Nutrition5k ([paper](https://arxiv.org/abs/2103.03375)) | Overhead RGB(-D) → kcal, mass, macros (InceptionV2, multi-task MAE) | Known 35.9 cm camera height for the depth volume; RGB-only learns scale implicitly | Per-gram kcal 9.5% vs absolute kcal 26.1% of mean; a volume scalar cuts mass error to 13.7% | [Repo](https://github.com/google-research-datasets/Nutrition5k), eval code only, CC BY 4.0 | Insight only: portion size, not food identity, is the bottleneck |
| ICPR 2024 MADiMa workshop | MFP3D ([paper](https://arxiv.org/html/2411.10492v1)) | RGB → ZoeDepth point cloud (1,024 pts, CurveNet) + ResNet50 → volume (L1) | Implicit, inside the regression | MetaFood3D volume **41.43% MAPE**; GT metric point cloud 15.59%; scale-normalised GT point cloud 42.19% | [jingema99/MFP3D](https://github.com/jingema99/MFP3D), licence unchecked | The ablation transfers: with scale removed, geometry adds nothing |
| arXiv, Jan 2026 | Size Matters ([paper](https://arxiv.org/html/2601.20051v1)) | RGB → One-2-3-45 mesh → 25 renders + input → frozen CLIP L/14@336 → MLP predicts V_gt/V_recon | Learned scale ratio, relative (1/v-weighted) L1 | MetaFood3D **35.83%**; OmniObject3D 85.96% vs category mean 87.46%; image-only features 76.10% | [gitlab viper-purdue/size-matters](https://gitlab.com/viper-purdue/size-matters) | The closest analogue. The recipe transfers; its checkpoints are leakage |
| arXiv, Dec 2025 (unreviewed) | PortionNet ([paper](https://arxiv.org/html/2512.22304v1)) | RGB → ViT-B/16 + ResNet-18 student distilled from a PointNet teacher with bbox-dimension embeddings | Bbox dimensions are privileged, teacher-only | MetaFood3D **17.43 ± 0.81%** volume MAPE, 3 seeds | No code | The privileged-distillation idea transfers; the number is unverified |
| CVPRW 2024 | 3D Object Scaling ([paper](https://arxiv.org/abs/2404.12257)) | RGB + estimated pose → render a known 3D food model | Physical reference in the scene | SimpleFood45 energy 17.67%; on MetaFood3D 79–104% volume MAPE | [gitlab monocular-food-volume-3d](https://gitlab.com/viper-purdue/monocular-food-volume-3d) | No: needs a reference |
| arXiv 2023–24 | NutritionVerse ([paper](https://arxiv.org/html/2309.07704v2)) | RGB(+depth) → nutrients, direct or via segmentation | Pixel counts or direct regression | Depth did not help direct prediction | Data on Kaggle, no code | Weak |
| Foods 2023 | DPF-Nutrition ([paper](https://arxiv.org/abs/2310.11702)) | RGB + predicted depth, cross-modal fusion | Implicit | PMAE 17.8% on Nutrition5k (secondary source) | No code | Weak |
| GitHub only | Graikos food_volume_estimation ([repo](https://github.com/AlexGraikos/food_volume_estimation)) | Monocular depth + Mask R-CNN → point cloud → volume → weight | User-supplied FoV and distance, or plate ellipse | No metrics | MIT | No |

Two datasets define the competition. **MetaFood3D** has 743 objects in 131 categories, with weight and nutrition ([arXiv 2409.01966](https://arxiv.org/abs/2409.01966)). **OmniObject3D** (CVPR 2023, CC BY 4.0) has 6,000 scans in 190 categories, and states no mesh units ([arXiv 2301.07525](https://arxiv.org/abs/2301.07525)). The missing unit convention is consistent with our oversized mode. Using any external copy of either dataset, or any checkpoint trained on them (Size Matters, MFP3D, PortionNet), would leak test objects.

### MetaFood challenges and winners

| Venue, year | Team / method | Input | Scale source | Key result | Code | Transfers? |
|---|---|---|---|---|---|---|
| CVPR 2024 MTF ([report](https://arxiv.org/html/2407.09285)) | VolETA, 1st (U. Barcelona) | Multi-view RGB-D; One-2-3-45 for single images | 1.2 cm checkerboard + depth | Phase I MAPE 10.97% | [GCVCG/VolETA-MetaFood](https://github.com/GCVCG/VolETA-MetaFood), **GPL-3.0** | No |
| CVPR 2024 | ININ-VIAUN, 2nd | COLMAP + NeRF; ZeroNVS for single images | Checkerboard corners + depth | Single-image objects still 32.77% and 34.01% error | [BITyia/cvpr-metafood](https://github.com/BITyia/cvpr-metafood), no licence | No; shows shape error of ~30% even with known scale |
| CVPR 2024 | FoodRiddle, best mesh | 2DGS; LGM, InstantMesh or One-2-3-45 for single images | Checkerboard | Lowest single-view Chamfer, 0.019232 | [FoodRiddle repo](https://github.com/jlyw1017/FoodRiddle-MetaFood-CVPR2024) | No |
| CVPR 2025 ([benchmark](https://arxiv.org/html/2602.13041v1)) | MDMS (≈ Hillo World / MonoBite), 1st | One multi-food image → Hunyuan3D + Depth Pro | Similarity alignment to metric depth × web plate/utensil statistics | MAPE 0.21 (std 0.16) | None found | No |
| CVPR 2025 | SGPS (≈ OneVol), 2nd | Hunyuan3D-2 + FoodMem | Known plate/utensil dimensions | MAPE 0.31 | [GCVCG/OneVol-MetaFood](https://github.com/GCVCG/OneVol-MetaFood) | No |
| CVPR 2025 | PSHS | Per-food reconstruction | Plate diameter / pixel diagonal | MAPE 0.46 | — | No |
| CVPR 2025 | GPT-5.2 baseline | Image → volume | VLM world knowledge | MAPE 0.34 | — | Partly: as a prior |
| CVPR 2026 ([PerBite](https://arxiv.org/abs/2606.02021)) | PerBite, 1st | Egocentric video → SAM 3 + Hunyuan3D / SAM 3D | Plate diameter; MoGe-2 as backup | 33.87% state MAPE, 53.74% consumed | Announced repo returns 404 | No |
| CVPR 2026 | CrossRecon, 2nd | Pi3 path + GPT-Image/Hunyuan3D path, fused | Plate and table priors | Phase I score 0.37 | [CrossRecon repo](https://github.com/WZ-CS/CVPR-2026-3rd-MetaFood-Workshop-challenge-2nd-Place-CrossRecon) | No |
| CVPRW 2026 | VolETA++ | Radiance fields, FoodMem | Checkerboard; corner threshold 240 is fragile | — | [GCVCG/VolETAPP](https://github.com/GCVCG/VolETAPP) | No |

The community converged on a three-part recipe: foundation segmentation, a generative mesh, and explicit scale from context or metric depth. Even so, volume MAPE stayed at **21–34% on real images that had context cues** ([benchmark](https://arxiv.org/html/2602.13041v1); [PerBite](https://arxiv.org/abs/2606.02021)). Two observations matter for us. First, every 2025–26 winner used Hunyuan3D, and its licence excludes the EU, so that recipe is unusable for a Bosch entry in any case ([Hunyuan3D 2.1 licence](https://raw.githubusercontent.com/Tencent-Hunyuan/Hunyuan3D-2.1/main/LICENSE)). Second, GPT-5.2's 0.34 MAPE with no geometry shows that world-knowledge size priors are meaningful when context is removed ([benchmark](https://arxiv.org/html/2602.13041v1)).

### Monocular metric geometry

| Venue, year | Model | Output | Scale handling | Cost | Licence | Use here |
|---|---|---|---|---|---|---|
| NeurIPS 2025 (verified by the team from the proceedings) | MoGe-2 ([repo](https://github.com/microsoft/MoGe), [arXiv](https://arxiv.org/abs/2507.02546)) | Metric point map, normals, FOV | DINOv2 ViT-L encoder + affine-invariant point-map head + CLS-token MLP log-scale head; trained on 24 datasets / 8.9M frames, **excluding OmniObject3D and MetaFood3D** | 326M params (ViT-L); ~29 ms FP16 at 1,200 tokens on A100 (paper) vs "60 ms" (repo) | Code MIT (DINOv2 parts Apache-2.0); HF weights MIT (team-verified) | Leakage-clean feature branch and scale-cue probe |
| CVPR 2025 | MoGe-1 | Affine-invariant point map | Relative only | 314M | MIT | Shape features |
| ICLR 2025* | Depth Pro ([repo](https://github.com/apple/ml-depth-pro)) | Metric depth + focal length | Estimates focal length itself | 0.3 s for 2.25 MP | Custom Apple licence ("personal ... license"), needs legal review | Probe only |
| CVPR 2024 / under submission | UniDepth v1/v2 ([repo](https://github.com/lpiccinelli-eth/UniDepth)) | Metric depth + intrinsics | Self-prompting camera module | — | **CC BY-NC 4.0** | Excluded |
| ICCV 2023 / TPAMI 2024 | Metric3D v1/v2 ([repo](https://github.com/YvanYin/Metric3D)) | Metric depth | **Needs focal length** | — | BSD-2* (unverified) | No |
| NeurIPS 2024 | Depth Anything V2 ([repo](https://github.com/DepthAnything/Depth-Anything-V2)) | Relative depth; domain-specific metric heads | Relative | Small 24.8M | Small Apache-2.0; **B/L/G CC-BY-NC** | Small only |
| arXiv, Nov 2025 | Depth Anything 3 ([repo](https://github.com/ByteDance-Seed/Depth-Anything-3)) | Depth, rays, poses | Metric-Large needs focal in px; Nested outputs metres | 0.08–1.40B | Small/Base/Metric-L/Mono-L Apache-2.0; others CC BY-NC | Probe only |
| CVPR 2024 | Marigold v1-0 ([card](https://huggingface.co/prs-eth/marigold-depth-v1-0/raw/main/README.md)) | Affine-invariant depth | Relative | Diffusion, multi-step | Apache-2.0 (v1.1 licence disputed) | Shape only |
| CVPR 2025 | VGGT ([repo](https://github.com/facebookresearch/vggt)) | Cameras, depth, point maps | Up to scale | 1B | Original non-commercial; a commercial checkpoint requires an application | Low value |
| arXiv 2026 (ECCV 2026 per a snippet) | WildMoGe ([paper](https://arxiv.org/abs/2606.02379)) | MoGe-2 fine-tuned on MetricScenes | Corrects "scale-collapse" | — | Weights licence not stated | Evidence only |

The WildMoGe paper shows why metric heads cannot be trusted here. MoGe-2 estimates the Arc de Triomphe at **18.8 m against a real 44.8 m**, because its metric output reflects the context mix of its training data rather than geometric evidence ([arXiv 2606.02379](https://arxiv.org/abs/2606.02379)). The geometric argument is stronger still. Real size is proportional to depth only at a fixed focal length ([Size-to-Depth](https://arxiv.org/pdf/1801.04461)). With auto-framing, camera distance scales with object size, so two objects of different size render identically up to texture detail. Any "metric" number from these models on our renders is therefore an appearance-and-category prior, which is the same thing a supervised head learns directly from 2,042 labels.

### Image-to-3D reconstruction

| Venue, year | Model | Output | Scale | Cost | Licence | Use here |
|---|---|---|---|---|---|---|
| Tech report, Dec 2025 | TRELLIS.2-4B ([card](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/README.md)) | PBR 3D asset | Unit-normalised | ~17 s at 1024³ on H100; ≥24 GB; Linux | **MIT** (verified) | Shape-factor feature (Alternative C) |
| CVPR 2025 | TRELLIS v1 ([repo](https://github.com/microsoft/TRELLIS)) | Mesh / 3DGS | Normalised | ~10–20 s* | MIT* (unverified) | Backup |
| Nov 2025 | SAM 3D Objects ([repo](https://github.com/facebookresearch/sam-3d-objects)) | Shape + layout (R, t, s) | Layout scale comes from MoGe point maps | Undocumented | SAM License, royalty-free | Shape; its scale is only as good as MoGe's |
| 2025 | Hunyuan3D 2.1 ([repo](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)) | Shape + texture | Normalised | 10 GB shape-only | **Territory excludes EU/UK/South Korea** | Excluded |
| 2024 | SF3D / SPAR3D ([licence](https://stability.ai/license)) | Textured mesh | Normalised | ~0.5 s* | Free only below US$1M revenue | Excluded |
| 2024 | InstantMesh, TripoSR, OpenLRM | Mesh / triplane | Normalised | Seconds* | Apache / MIT / CC BY-NC weights* (all unverified) | Verify before use |

Size Matters found One-2-3-45 better than TripoSR for scale-ratio regression (MAE 59.09 vs 88.80 mL) ([Size Matters](https://arxiv.org/html/2601.20051v1)). Every generator outputs unit-normalised geometry, so it can supply only the scale-free shape factor, never the scale.

### Size priors and VLMs

| Venue, year | Work | Finding | Relevance |
|---|---|---|---|
| ECCV 2020 | Single View Metrology in the Wild ([paper](https://arxiv.org/pdf/2007.09529)) | Absolute scale comes from co-occurring categorical size references (humans, cars) | Our renders contain no co-occurring references |
| CVPR 2023 | Omni3D / Cube R-CNN ([repo](https://github.com/facebookresearch/omni3d)) | Category size priors for 3D boxes | CC-BY-NC; idea only |
| arXiv 2026 | Physically guided mass estimation ([paper](https://arxiv.org/html/2601.20303)) | Appearance is "only weakly correlated with absolute object size" | Caps the expected within-category image signal |
| arXiv 2026 | Size Matters GPT-4o baseline ([paper](https://arxiv.org/html/2601.20051v1)) | 153.80% MAPE on MetaFood3D (63.25% with context); 169.81% on OmniObject3D | Zero-shot VLM is far worse than supervised |
| SSIAI 2026 | GPT-5.2 on the implicit-scale benchmark ([paper](https://arxiv.org/html/2602.13041v1)) | 0.34 MAPE, behind geometry with plate cues (0.21) | VLM priors are competitive when context is gone |
| CVPR 2025 | VSI-Bench ([site](https://vision-x-nyu.github.io/thinking-in-space.github.io)) | GPT-4o object-size MRA roughly 44–69 (column mapping uncertain) | Coarse size knowledge only |
| arXiv 2025 | SIRI-Bench ([paper](https://arxiv.org/pdf/2506.14512)) | Best VLM gets only about 31% of answers within 40% error | Poor metric precision |
| arXiv 2026 | ReVSI ([paper](https://arxiv.org/pdf/2604.24300)) | Benchmark rebuilt so models cannot rely on typical-size priors | Here, a typical-size prior is exactly what we want |
| IEEE BHI 2025 | ACETADA ([paper](https://arxiv.org/abs/2507.07048)) | Contextual metadata significantly reduces LMM portion error | Text priors help |

### Regression formulation and imbalanced regression

| Venue, year | Method | Key idea | Code | Use here |
|---|---|---|---|---|
| ICML 2024* | HL-Gauss / "Stop Regressing" ([paper](https://arxiv.org/html/2403.03950)) | Gaussian-smoothed CDF-binned targets + cross-entropy; tune σ/ς, default 0.75; degrades gracefully with label noise | Paper appendix only | **Primary head** |
| ICML 2018* | Imani & White ([paper](https://arxiv.org/abs/1806.04613)) | Origin of the histogram loss; no representational benefit in small supervised regression | — | Tempers expectations |
| CVPR 2018* | DORN ([paper](https://arxiv.org/abs/1806.02446)) | Ordinal regression over log-uniform (SID) bins | [hufu6371/DORN](https://github.com/hufu6371/DORN) | Bin layout |
| CVPR 2021* | AdaBins ([paper](https://arxiv.org/abs/2011.14141)) | Adaptive bins with expectation decoding | [AdaBins](https://github.com/shariqfarooq123/AdaBins) | Avoid its expectation decoding |
| PRL 2020 / PAA 2023* | CORAL / CORN ([CORAL](https://arxiv.org/abs/1901.07884), [CORN](https://arxiv.org/abs/2111.08851)) | Rank-consistent ordinal heads | [coral-cnn](https://github.com/Raschka-research-group/coral-cnn) | Ablation |
| ICCV 2023* | Pintea et al. ([paper](https://arxiv.org/abs/2308.10603)) | An added classification loss helps mainly under imbalance | — | Motivates coarse-bin auxiliary |
| ICLR 2023* | Ordinal Entropy ([paper](https://arxiv.org/abs/2301.08915)) | Regulariser for higher-entropy features | [OrdinalEntropy](https://github.com/needylove/OrdinalEntropy) | Low priority |
| NeurIPS 2020* | Deep Evidential Regression ([paper](https://arxiv.org/abs/1910.02600)) | Unimodal Student-t predictive | — | Cannot represent two modes |
| ICML 2021 | LDS/FDS ([paper](https://arxiv.org/abs/2102.09554)) | Kernel-smoothed label-density reweighting; feature smoothing | [imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression) | Mild LDS only |
| CVPR 2022 | Balanced MSE ([paper](https://arxiv.org/abs/2203.16427)) | Corrects for the training label prior | [BalancedMSE](https://github.com/jiawei-ren/BalancedMSE) | Redundant with a classifier head |
| ICML 2022 | RankSim ([paper](https://arxiv.org/abs/2205.15236)) | Matches feature-space neighbour ranks to label ranks | [ranksim](https://github.com/BorealisAI/ranksim-imbalanced-regression)* | Cheap regulariser |
| ICLR 2024* | ConR ([paper](https://arxiv.org/abs/2309.06651)) | Contrastive label-similarity regulariser | [ConR](https://github.com/BorealisAI/ConR) | Needs large batches |
| CVPR 2024 | HCA ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Deep_Imbalanced_Regression_via_Hierarchical_Classification_Adjustment_CVPR_2024_paper.html)) | Coarse-to-fine hierarchical classifiers | [HCA](https://github.com/xhp-hust-2018-2011/HCA) | Coarse-bin auxiliary head |
| JASA 2011 | Gneiting ([paper](https://arxiv.org/abs/0912.0902)) | The point forecast must be the Bayes rule of the scoring function | — | **Decoding** |
| ICML 2017* | Temperature scaling ([paper](https://arxiv.org/abs/1706.04599)) | One scalar on the logits, fit by validation NLL | — | Calibrates the mode probability |

### Backbones and transfer

| Venue, year | Model / recipe | Key fact | Licence | Use here |
|---|---|---|---|---|
| TMLR 2024 | DINOv2 ([repo](https://github.com/facebookresearch/dinov2)) | Linear-probe IN-1k accuracy: L 86.3, g-reg 87.1; k-NN close to linear | **Apache-2.0** | Primary backbone |
| arXiv 2025 | DINOv3 ([repo](https://github.com/facebookresearch/dinov3)) | Strong frozen dense features; top CSIRO teams used it | Custom, gated licence ([licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) | Optional, after legal review |
| arXiv 2025 | SigLIP 2 ([card](https://huggingface.co/google/siglip2-so400m-patch16-naflex)) | Semantics; NaFlex variable resolution | Apache-2.0 | Category signal |
| Tech report 2025 | Perception Encoder ([repo](https://github.com/facebookresearch/perception_models)) | The best features sit in intermediate layers | Apache-2.0 | Alternative |
| CVPR 2024/25; 2026 report | C-RADIOv3/v4 ([repo](https://github.com/NVlabs/RADIO)) | Distils DINOv3 + SigLIP2 + SAM3 | NVIDIA Open Model License (commercial OK); non-C variants NSCL | Alternative |
| — | MAE, ConvNeXt V2 IN weights ([MAE](https://github.com/facebookresearch/mae), [CNv2](https://github.com/facebookresearch/ConvNeXt-V2)) | MAE features are weak when frozen | **CC-BY-NC** | Excluded |
| ICLR 2022* | LP-FT ([paper](https://arxiv.org/abs/2202.10054)) | Probe first, then fine-tune, to limit feature distortion | — | Fine-tune recipe |
| CVPR 2022* / ICML 2022* | WiSE-FT, model soups ([WiSE](https://arxiv.org/abs/2109.01903), [soups](https://arxiv.org/abs/2203.05482)) | Weight interpolation and averaging | [wise-ft](https://github.com/mlfoundations/wise-ft) | Within-family ensembling |
| ICLR 2022* / ICML 2024* | LoRA / DoRA ([LoRA](https://arxiv.org/abs/2106.09685), [DoRA](https://arxiv.org/abs/2402.09353)) | Parameter-efficient fine-tuning | [PEFT](https://github.com/huggingface/peft) | Primary fine-tune |
| Kaggle 2025–26 | CSIRO Image2Biomass ([frozen solution](https://github.com/Forrest-xlz/CSIRO-kaggle), [1st place](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution)) | Frozen DINOv3-7B multi-layer features at 256/512/768 px + stacked ridge; the 1st place used interval classification | — | Recipe precedent |
| Kaggle 2021 | PetFinder Pawpularity ([summary](https://zenn.dev/kohecchi/articles/efc6d291dec289)) | Frozen features + SVR + hill-climbing won | — | Small-data precedent |
| ICLR 2016* | Generalized distillation ([paper](https://arxiv.org/abs/1511.03643)) | A teacher sees privileged information; the student sees images only | — | Mesh teacher |
| NeurIPS 2019* | FixRes ([paper](https://arxiv.org/abs/1906.06423)) | RandomResizedCrop changes apparent object size | — | Avoid scale augmentations |

### What transfers and what does not

| Component | Transfers? | Why |
|---|---|---|
| Checkerboard, plate/utensil, pixel-diagonal scaling (all MetaFood winners) | **No** | The renders contain no reference object |
| Metric depth as a scale source (Depth Pro, MoGe-2, Metric3D, DA3) | **No as truth; yes as a feature** | No intrinsics, and auto-framing equalises geometry. The output is an appearance prior, with documented scale-collapse ([WildMoGe](https://arxiv.org/abs/2606.02379)) |
| Template pose-fitting (3D Object Scaling) | **No** | Needs a reference |
| Learned scale from appearance (Size Matters, MFP3D's RGB branch) | **Yes** | The only mechanism left. MFP3D shows normalised geometry adds nothing once scale is gone (42.19% vs 41.43%) ([MFP3D](https://arxiv.org/html/2411.10492v1)) |
| Shape × scale factorisation with image-to-3D | **Partly** | A scale-free shape factor is useful; its gain is bounded by within-category fill variance (see EDA) |
| Relative L1 loss (Size Matters) | **Yes, adapted** | L1 on ln V targets the median, which is sMAPE's unimodal Bayes rule |
| Privileged teacher (PortionNet) and multi-task heads (Nutrition5k) | **Yes** | Meshes, bbox_mm and weight_g exist for training only |
| Generative-mesh recipe (Hunyuan3D) | **No** | Licence excludes the EU |
| VLM size priors | **Partly** | Text-only, offline category priors are safe. Per-image cloud calls are an open rules question |
| External MetaFood3D / OmniObject3D data or checkpoints trained on them | **Forbidden in practice** | Leakage of the 851 test objects |

## sMAPE's log-space form sets a 40% floor and a 199% mode tax

The metric does more to fix the design than any backbone result. For positive values with ratio r = v̂/v, the competition sMAPE term 2|v̂−v|/(|v̂|+|v|) equals **2·tanh(|ln r|/2)**. Our notes checked this identity numerically. sMAPE is therefore symmetric in log space and bounded at 200%. It behaves like L1 for small log errors and saturates like a 0–1 loss for large ones. The often-quoted asymmetry (Goodwin–Lawton; Hyndman's correction that too-small forecasts are penalised more) concerns equal *absolute* deviations, not equal ratios ([Hyndman](https://www.r-bloggers.com/2014/04/errors-on-percentage-errors/)). Gneiting's rule, that the point forecast must be the Bayes rule of the scoring function, then gives the decoder ([Gneiting](https://arxiv.org/abs/0912.0902)). For a unimodal log-symmetric predictive distribution the sMAPE optimum is the **median**, exp(μ), with no exp(σ²/2) correction. The MAPE optimum, exp(μ−σ²), would shrink predictions by about 24% at σ = 0.52, which is the wrong correction here. For a multimodal distribution the optimum is a mode-seeking weighted median, with weights ½·sech²((t−z_k)/2) in log space. A grid search over bin centres finds it.

The floors follow directly. I simulated E[2·tanh(|σZ|/2)] for normal log-residuals. A predictor that knows only the category scores **39.8% at σ = 0.52**, 34.8% at 0.45, 31.2% at 0.40, 27.4% at 0.35, and 107.9% at 1.89, which is the global spread. Our measured in-sample figures are 94% for the global median and 41% for the true-category median excluding the large mode. They sit on this curve. That tells us the category prior has already extracted nearly all of the between-category signal. Every further point must come from explaining within-category variance: lowering σ from 0.52 to 0.40 is worth about 9 points. The in-sample numbers flatter reality. With a median of 8 objects per category and 48 categories with ≤3 objects, the out-of-fold category median will score worse, which makes empirical-Bayes pooling matter.

The large mode is a separate, discrete problem. A 1000× miss costs 2·tanh(ln 1000/2) = **199.6%**, and a 200× miss costs 198.0%, so the size of the error barely matters once the mode is wrong. Because sMAPE saturates, hedging between modes is never optimal. In the notes' worked example, with masses at 10 and 10,000 cm³, the optimal prediction jumps between modes at P(large) ≈ 0.5. At P(large) = 0.10 the optimal choice costs 20.0% expected sMAPE. Decoding at exp(E[ln V]) costs 79.7%, and the linear mean costs 192.8%. L2 on ln V lands about 0.69 nats above the base mode, roughly a 66% sMAPE penalty on 90% of a mixed category. L1 on ln V barely moves, but it can never output the large mode. So the only defensible structure is a full predictive distribution followed by an explicit mode decision. The roughly 5-point gap between our 46% and 41% in-sample figures is the current "mode tax" under a category median. It falls on minority-mode objects inside mixed categories, and an image model can remove it only if the mode is visible in the image.

The public leaderboard is noisy. I simulated per-object errors with σ = 0.52 base residuals plus 10% wrong-mode objects. That gives a per-object sMAPE standard deviation of about 55 points, hence a **standard error of about 2.4 points on the 511 public images** and about 1.9 points on the 851 private images. This is my estimate, and it depends on that assumed mix. Public differences under about 5 points are therefore indistinguishable from noise, and model selection must rest on repeated cross-validation ([Bouthillier et al.](https://arxiv.org/abs/2103.03098)).

## A category-conditioned binned head is the primary design

The architecture follows from the arithmetic. Make the category prior explicit and well-pooled. Predict a distribution over ln V instead of a point. Separate the base-mode shape of that distribution from the discrete unit mode. Decode with the sMAPE Bayes rule. Use mesh-derived targets to shape the representation. HL-Gauss gives the head: Gaussian-smoothed, CDF-integrated bin targets with cross-entropy, σ/ς ≈ 0.75, robust to label noise ([HL-Gauss](https://arxiv.org/html/2403.03950)). DORN supplies the log-uniform bin layout ([DORN](https://arxiv.org/abs/1806.02446)). Classification heads help mainly under imbalance ([Pintea et al.](https://arxiv.org/abs/2308.10603)), and Imani & White found no representational advantage in small clean problems ([Imani & White](https://arxiv.org/abs/1806.04613)). So the case for bins here is multimodality plus decoding, not a hoped-for accuracy gain.

### Primary design A: foundation features, category prior, binned base head, mode head

```
rgb_industrial.png (1024², grey backdrop)
   │  resize 518×518 (antialias), keep backdrop, ImageNet norm; TTA = hflip
   ├──► DINOv2-L/14-reg  [LoRA r=16 on last 12 blocks]  → CLS + mean-patch, layers {12,18,24} ─┐
   ├──► SigLIP 2 so400m/16 @384 or NaFlex 512 [frozen]  → pooled embedding                  ─┤
   └──► hand-crafted: silhouette fill, aspect, hull ratio, colour stats, HF spectral energy  ─┤
                                                                                               ▼
                                         concat → LayerNorm → MLP(→1024→768, GELU, dropout 0.2) = h
   ┌───────────────────────────────────────────────────────────────────────────────────────────┤
   │ Category head: 164-way softmax P(c|x)   (replaced by one-hot c if folder category allowed)│
   │ Coarse head: 7-way category_l1 (aux)                                                      │
   │ Base head: K=128 logits over ln V_base bins + log-prior bias log p̃_c(bin)  (HL-Gauss CE)  │
   │ Mode head: logit P(u=1|x,c) + prior offset logit(π̃_c)                     (BCE)          │
   │ Aux heads: ln sorted bbox_mm (3), logit fill, ln surface area, sphericity,                │
   │            ln density = ln(weight_g/V) [MetaFood3D-masked]                 (L1)           │
   └───────────────────────────────────────────────────────────────────────────────────────────┘
Decode: p(z|x) = Σ_c P(c|x)·[(1−π_c(x))·p_base(z|x,c) + π_c(x)·p_base(z−Δ|x,c)]
        → temperature T → t* = argmin_t Σ_k p_k·2·tanh(|t−c_k|/2) → v̂ = exp(t* + δ), clip [0.3, 40000]
```

The base head is trained on ln V_base. For flagged oversized objects, the label is shifted down by the estimated unit factor Δ. Δ = ln 1000 if EDA confirms a clean 10× linear unit error; otherwise use a per-category estimate, because our measured ratios span ~200–1000×. This gives the base head a clean within-category target instead of a two-mode mess. The mode head learns u. Its prior is a Beta-Binomial-shrunk category rate π̃_c = (k_c+α)/(n_c+α+β), with α+β small, around 2, because the rates are strongly bimodal across categories (hazelnut, longan and loquat at 100%; haw_thorn 28/30; most categories 0%). The strength is tuned by out-of-fold log-loss. The log-prior bias p̃_c is the empirical-Bayes-shrunk category histogram of ln V_base. Its location m̃_c = w_c·m_c + (1−w_c)·μ_l1, with w_c = τ²/(τ²+σ_w²/n_c), shrinks toward the coarse-category median, following Efron–Morris and Micci-Barreca target encoding ([Efron & Morris](https://doi.org/10.1080/01621459.1975.10479864); [Micci-Barreca](https://doi.org/10.1145/507533.507538)). Bins are uniform in ln V over [ln 0.2, ln 60000]: 128 bins of width ς ≈ 0.098 nats, finer than the 0.52 within-category spread, with HL-Gauss σ = 0.75ς. The loss is

L = CE_HL(base) + 0.5·BCE(mode) + 0.3·CE(category) + 0.1·CE(coarse) + 0.1·Σ L1(aux),

optionally with Kendall uncertainty weighting in place of the fixed 0.1 auxiliary weights ([Kendall et al.](https://arxiv.org/abs/1705.07115)). After that, 2–3 epochs add the expected-sMAPE term Σ_k p_k·2·tanh(|c_k − ln y|/2) on top of the cross-entropy. That term alone is a poor training loss, because its gradient vanishes like sech² for large errors and would stop the model learning the far mode.

If the organisers allow the folder-path category at inference, the category head becomes an auxiliary and P(c|x) is one-hot. That removes category confusion entirely, and confusion is costly when look-alikes carry different mode rates (walnut 21/27 vs hazelnut 100%). If they do not, mixing over P(c|x) is mandatory, because an argmax category throws away the uncertainty the decoder needs.

The large-mode strategy runs in three tiers. Tier 1 is always on: predict the large mode when the calibrated P(large|x) exceeds a threshold τ* tuned out-of-fold near 0.5. With category information alone, this already sends hazelnut, longan, loquat, haw_thorn, walnut and cherry to the large mode, accepting about 200% on their minority base objects. Tier 2 activates only if the EDA mode-identifiability test passes: the image term in the mode head is trusted and the large mode is predicted per object. Tier 3 is never done: filtering, down-weighting or robust-loss suppression of oversized labels. They are official truth, and any filter guarantees a 199% loss on every such test object. The same reasoning rules out co-teaching or small-loss selection on raw labels ([Co-teaching](https://arxiv.org/abs/1804.06872)).

The mesh supplies privileged auxiliary targets: sorted ln bbox dimensions, fill ratio V/(bbox volume) (median 0.45, std 0.13), surface area, sphericity, convex-hull ratio, and density from weight_g for MetaFood3D. These follow the multi-task pattern of Nutrition5k ([Nutrition5k](https://arxiv.org/abs/2103.03375)) and the privileged-information pattern of PortionNet ([PortionNet](https://arxiv.org/html/2512.22304v1)) and generalized distillation ([Lopez-Paz et al.](https://arxiv.org/abs/1511.03643)). They cannot create a scale cue the image lacks. Their value is to split the shape problem, which the image can see, from the scale problem, which it mostly cannot, and to regularise a representation trained on only 2,042 objects.

### Alternative B: frozen-feature stack with empirical-Bayes and retrieval features

```
image ─► frozen {DINOv2-L-reg, SigLIP2, PE-Core-L, MoGe-2 encoder, (DINOv3-L if licensed)}
         @ {224, 448, 518} px, last-4-layer CLS + mean-patch, hflip-averaged ─► in-fold PCA(256)/backbone
      ─► + EB category prior m̃_c, π̃_c, σ̃_c, coarse prior; + kNN(k=10, cosine, within P(c|x) top-3)
           neighbour ln V_base median, similarity-weighted mean, neighbour mode rate
      ─► heads: ridge on residual (ln V_base − m̃_c); kernel ridge / SVR; LightGBM; mode logistic
      ─► convert each point head to a distribution: Laplace(μ̂, b̂_c), b̂_c from OOF residuals
      ─► mixture with mode probability ─► same sMAPE-argmin decoder
```

This is the CSIRO frozen-DINOv3 recipe adapted to sMAPE: multi-layer, multi-scale features with stacked ridge, which scored public 0.757 / private 0.648 ([Forrest-xlz/CSIRO-kaggle](https://github.com/Forrest-xlz/CSIRO-kaggle)). It also follows the Pawpularity frozen-feature-plus-SVR winner ([summary](https://zenn.dev/kohecchi/articles/efc6d291dec289)). k-NN on frozen DINO features is close to linear-probe accuracy ([DINOv2](https://github.com/facebookresearch/dinov2)) and helps most in low-data settings ([Jia et al.](https://arxiv.org/abs/2112.08459v2)). This is the cheapest strong baseline, the easiest to reproduce, and the instrument for the scale-cue test. It should be built first.

### Alternative C: Size-Matters-style shape × scale factorisation

```
image ─► TRELLIS.2-4B (MIT) or SAM 3D Objects (SAM License) ─► unit-normalised mesh
      ─► voxelise/repair to watertight ─► V_norm = V/L_max³, V/OBB, V/hull, sphericity, aspect
      ─► render 25 views (3 polar × even azimuth, as Size Matters)
image + renders ─► SigLIP2 / CLIP-type encoder ─► MLP ─► ln L_max distribution (supervised by bbox_mm)
ln V = 3·ln L_max + ln V_norm  (or feed V_norm features into design A's base head)
```

Size Matters showed that pairing input-image features with render features of a scale-free reconstruction lowered MAPE from 76.10% (image only) to 35.83% ([Size Matters](https://arxiv.org/html/2601.20051v1)). In our setting the pairing can help only through the shape factor, and the training meshes let us bound that gain before spending GPU time. If shape and scale are independent, removing all fill variance reduces the log-std to √(σ²_lnV − σ²_ln fill). With an overall fill std of 0.13 around 0.45, about 0.29 in log terms, the bound is 0.52 → 0.43 at best: about 6 sMAPE points, and less if within-category fill variance is smaller, which is likely. Cost is about 3–17 s per image on an H100 with ≥24 GB ([TRELLIS.2 card](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/README.md)), so roughly 2–10 GPU-hours for all splits. This branch is worth building only if the EDA bound and design A's fill-head error leave real headroom.

### Alternative D: MoGe-2 geometry branch and offline text priors

MoGe-2 is leakage-clean, MIT-licensed and fast. Its CLS-token log-scale head is a learned metric prior, exactly the quantity whose within-category correlation with true size we need to measure. Its point map yields scale-free thickness and extent ratios and normal statistics. The branch feeds three things into design A's fusion MLP: the predicted log-scale, depth-range/extent ratios, and the fine-tuned MoGe encoder's CLS token. For the 48 categories with ≤3 objects, an offline table of typical volumes produced by an LLM from category names alone can replace the coarse-category parent in the shrinkage. It is frozen into the code repository and verified against training medians. This sends no evaluation images anywhere, which sidesteps the open cloud-API question. Whether "LLM world knowledge" counts as non-public information still needs organiser confirmation. GPT-class priors scored 0.34–0.63 MAPE in food benchmarks ([benchmark](https://arxiv.org/html/2602.13041v1); [Size Matters](https://arxiv.org/html/2601.20051v1)), so they are a prior, not a predictor.

## Data processing and analytics must first prove a scale cue exists

The pipeline should be boring and fully hashed, because organisers will re-run it.

```
raw/{group}/{category}/{sample_id}/{rgb_industrial.png, gt.json, mesh.ply}
 └► manifest.parquet: path, sha256(image), group, category, category_l1, source, labels
 └► labels: ln V; assert |mesh.volume|/1000 == volume.primary_cm3; sorted bbox; fill; area;
            sphericity; hull ratio; density (MF3D); mode flag u; per-object shift factor
 └► folds.csv: 5-fold × 3 seeds, stratified (see validation); saved and versioned
 └► images: RGB float32, 518² antialiased resize (no crop, keep backdrop); grey-threshold mask
            for hand-crafted features only
 └► feature cache: {backbone, revision, resolution, layer, flip} → .npy + hash
 └► models → OOF binned probabilities → calibration (T, δ, τ*) → decoder
 └► writer + validator for preds_volume.zip: every id present, finite, > 0, clipped
```

The augmentation policy follows from auto-framing. Horizontal flips, ±10° rotation with grey fill, ±5% translation, mild colour jitter (brightness/contrast ±0.15, saturation ±0.1, hue ±0.02), light blur and noise are safe. RandomResizedCrop, scale jitter and mixup/cutmix are label-altering or physically meaningless ([FixRes](https://arxiv.org/abs/1906.06423)). Vertical flips are an ablation, because the camera has a canonical elevation and lighting. Use register variants when pooling patch tokens at high resolution ([registers](https://arxiv.org/abs/2309.16588)).

The analytics plan is ordered so its answers gate later work.

| # | Analysis | Method | Decision it drives |
|---|---|---|---|
| 1 | Variance decomposition | Between/within-category ln V variance, overall and by source and category_l1; distribution of per-category σ, not just the median 0.52 | Category-floor estimate per category; heteroscedastic σ̃_c |
| 2 | Out-of-fold category baseline | OOF (not in-sample) category median and EB-shrunk median, scored with sMAPE; singletons reported separately | The honest baseline every model must beat |
| 3 | Oversized-mode anatomy | Flag objects with ln V − m_c,base > ln 30 or implausible bbox (walnut 275 mm); histogram of shift factors (a spike at 1000 means 10× linear unit error); π_c with Beta-Binomial; correlate u with mesh metadata (vertex count, mean edge length in mm, origin offset), sample_id patterns and gt_confidence | Δ for label shifting; whether π varies within category; mode-head features |
| 4 | Scale-cue test A (geometry) | Replicate r≈0.01 between on-screen size and real size on all train images, within category | Confirms silhouette = shape, not scale |
| 5 | Scale-cue test B (learned) | Ridge on frozen features → within-category residual ln V_base − m̃_c, category-stratified CV; **permutation null** (shuffle residuals within category, 200 repeats) | If OOF R² < 0.05 or p > 0.01, stop backbone tuning and invest in priors and decoding |
| 6 | Scale-cue test C (render physics) | Hand-crafted probes vs within-category residual: high-frequency spectral energy per object extent (texture granularity), contact-shadow width and softness on the backdrop, specular-highlight size, Cycles noise level, backdrop gradient. Blender lights, AO distance or shadow softness set in absolute units would leak scale when the camera moves but lights do not | Hidden renderer cues; whether RandomResizedCrop destroys them |
| 7 | Mode identifiability | In the 31 categories with oversized objects, leave-one-out classifier for u on frozen + physics features; AUC with permutation null | Enables tier 2 of the mode strategy |
| 8 | Shape-information bound | Within-category Var(ln fill) / Var(ln V) | Upper bound on Alternative C's gain |
| 9 | Category confusion | 164-way k-NN / probe accuracy; confusion pairs with different priors or π_c | Value of the folder category; need for mixture decoding |
| 10 | Source identifiability | MetaFood3D vs OmniObject3D classifier from the image | The large mode exists only in OmniObject3D |
| 11 | Near-duplicates | DINOv2 CLS cosine above the 99.9th percentile; inspect same-mesh / different-scale pairs ([SSCD](https://arxiv.org/abs/2202.10261)) | Grouping in folds; irreducible noise |
| 12 | Adversarial validation | Train vs public-eval vs test images, LightGBM on embeddings; expect AUC ≈ 0.5 ([Pan et al.](https://arxiv.org/abs/2004.03045)) | Confirms exchangeability of the random split |
| 13 | Rare-category audit | The 48 categories with ≤3 objects: parent choice, text-prior agreement | Shrinkage strength |

## Validation, training and calibration recipes

Cross-validation should mirror the random split. Use 5-fold stratified K-fold on category, with category_l1 as the stratum for categories of fewer than 5 objects, repeated with 3 seeds. Group renders of any duplicated mesh found in analysis #11. Compute every category statistic in-fold, as in target encoding: with ≤3 objects, in-fold statistics leak the label almost completely. CV is still pessimistic for singleton categories. Their held-out object has no same-category training example, whereas every evaluation category appears in training. So report three numbers: OOF sMAPE overall, OOF sMAPE on objects whose category keeps at least one in-fold training member, and the stress-test GroupKFold-by-category score. Treat the second as the leaderboard proxy. Select models by mean OOF across seeds. The public leaderboard serves as a format and sanity check only, given its ~2.4-point standard error. The CSIRO winners' practice of mirroring the hidden-test structure in CV is the precedent ([CSIRO 1st place](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution)).

| Setting | Design A (LoRA fine-tune) | Design B (frozen stack) | Size-Matters-style MLP |
|---|---|---|---|
| Input | 518² (37×37 tokens at patch 14); SigLIP 2 at 384 | 224 / 448 / 518, hflip-averaged | Frozen SigLIP 2 / CLIP features of image + 25 renders |
| Trainable | LoRA r=16, α=32, dropout 0.05 on qkv/proj/MLP of last 12 blocks; or full FT with LLRD 0.75–0.85 after a linear probe (LP-FT) | Heads only | 3-layer MLP |
| Optimiser | AdamW, wd 0.05; lr 1e-3 heads, 2e-4 LoRA; 1-epoch warmup, cosine; 25 epochs; batch 32; bf16; clip 1.0; EMA 0.999 | Ridge α ∈ [0.1, 100]; SVR RBF C grid; LightGBM leaves 15, lr 0.03, ≤2,000 rounds with early stopping, min_data_in_leaf 10, feature_fraction 0.5 | Paper recipe: 300 epochs, batch 64, lr 1e-4 ×0.7 every 10 epochs ([Size Matters](https://arxiv.org/html/2601.20051v1)), with L1 on ln V replacing 1/v-weighted L1 |
| Loss | HL-Gauss CE + mode BCE + category/coarse CE + aux L1; final 2–3 epochs add expected sMAPE | L1 on residual ln V; logistic for mode | L1 on ln-scale |
| Reweighting | Optional sqrt-inverse LDS on ln V, clipped ([LDS](https://github.com/YyzHarry/imbalanced-regression)); RankSim as an ablation | None | None |
| Folds × seeds | 5 × 3 = 15 models | 5 × 3 | 5 × 3 |

These hyperparameters are my starting points, derived from the cited recipes and not tuned on this data. LP-FT protects pretrained features when the head starts random ([LP-FT](https://arxiv.org/abs/2202.10054)). With only 2,042 images, fine-tuning gains will be small and high-variance, so every comparison should use the 3-seed mean.

Ensembling must happen in distribution space. Average the binned probability vectors across folds, seeds and models, with members' point heads converted to distributions using OOF residual widths. Then decode once. Averaging decoded point predictions hedges between modes, which the decision theory shows is harmful. Mixture weights come from hill-climbing on OOF sMAPE, which is the Pawpularity practice ([summary](https://zenn.dev/kohecchi/articles/efc6d291dec289)). Weight soups and WiSE-FT apply only within one fine-tuning family from a shared initialisation ([soups](https://arxiv.org/abs/2203.05482); [WiSE-FT](https://arxiv.org/abs/2109.01903)). Calibration fits exactly three scalars out-of-fold:

- a temperature on the bin logits and on the mode logit, by NLL ([Guo et al.](https://arxiv.org/abs/1706.04599));
- a global log-offset δ by 1-D search on OOF sMAPE;
- the mode threshold τ*.

Lognormal back-transform corrections such as Duan smearing target the mean and are wrong for sMAPE ([Duan](https://doi.org/10.1080/01621459.1983.10478017)).

## Licences and reproducibility prune the model zoo

Winners hand over code that organisers re-run, components must be royalty-free for an EU entity, and evaluation data cannot leave through unvetted APIs. Together these rules remove several of the community's favourite tools before any accuracy comparison.

| Component | Licence (status) | Decision |
|---|---|---|
| DINOv2 (all sizes, incl. reg) | Apache-2.0 (verified) ([repo](https://github.com/facebookresearch/dinov2)) | **Use: primary** |
| SigLIP 2, Perception Encoder | Apache-2.0 (verified) ([SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-naflex), [PE](https://github.com/facebookresearch/perception_models)) | Use |
| MoGe-2 code and weights | MIT, DINOv2 parts Apache-2.0 (team-verified via HF API) | Use |
| TRELLIS.2-4B | MIT (verified) | Optional (Alternative C) |
| SAM 3D Objects | SAM License, royalty-free (verified) | Optional |
| C-RADIOv3/v4 | NVIDIA Open Model License, commercial use allowed (verified) | Optional |
| DINOv3 | Custom gated licence with attribution and indemnity terms (verified) ([licence](https://ai.meta.com/resources/models-and-libraries/dinov3-license)) | Only after legal review and confirmation that organisers can download it; keep a DINOv2 fallback with a known score |
| Depth Pro | Custom Apple licence with "personal" wording | Legal review before use |
| DA V2 Small; DA3 Small/Base/Metric-L/Mono-L; Marigold v1-0 | Apache-2.0 (verified) | Allowed, low value |
| Hunyuan3D 2.x | Territory excludes the EU (verified) | **Excluded** |
| UniDepth, DA V2 B/L/G, DA3 Large/Giant/Nested, Cube R-CNN, MAE, ConvNeXt V2 IN weights, non-C RADIO, original VGGT | Non-commercial (verified) | **Excluded** |
| SF3D / SPAR3D | Stability Community License, US$1M revenue cap (verified) | **Excluded** |
| VolETA code | GPL-3.0 (verified) | Do not vendor into the handover code |
| Size Matters, MFP3D code; Metric3D, ZoeDepth, TRELLIS v1, TripoSR, InstantMesh, Zero123++, OpenLRM, OpenCLIP, EVA-CLIP | Unverified | Check before any use; their trained checkpoints on MetaFood3D/OmniObject3D are leakage regardless |
| HF PEFT | Apache-2.0 (prior knowledge) ([repo](https://github.com/huggingface/peft)) | Use |

For reproducibility, pin everything:

- **Environment:** a `uv.lock` plus a Dockerfile with pinned CUDA, torch, timm, transformers, peft and scikit-learn.
- **Weights:** pin every model by HF revision hash and record its SHA256.
- **Data and folds:** write fold assignments to disk and hash every cached feature.
- **Determinism:** seed Python, NumPy, torch and DataLoader workers; call `torch.use_deterministic_algorithms(True)`; set `cudnn.benchmark=False` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` ([PyTorch notes](https://pytorch.org/docs/stable/notes/randomness.html)).
- **Entry points:** ship one `train.sh` and one `predict.sh`.
- **Inference:** run fully offline, with no cloud calls.

Bit-exact GPU determinism is not guaranteed across hardware, but averaging 15+ models dampens it. A dry run should rebuild the final submission from a clean container on a second machine and compare predictions. The tolerance target is a mean |Δ ln v̂| < 0.01.

## Ablations ranked by expected sMAPE gain

The expected impacts below are my estimates from the decision theory and the cited evidence. They are hypotheses to falsify, not measurements. P0 items are cheap and decide the rest.

| Priority | Ablation | Arms | Expected impact (est.) | Cost |
|---|---|---|---|---|
| P0 | Category prior | none vs raw OOF median vs EB-shrunk; folder category vs predicted vs P(c\|x) mixture | Largest factor: ~94% → ~45–50% OOF | Hours |
| P0 | Large-mode handling | filter labels vs ignore vs category π̃_c vs mode head vs single bimodal head | Filtering costs ≥5 pts; π̃_c recovers majority-large categories; image cue unknown | Hours |
| P0 | Decoding | exp(E[ln V]) vs median vs sMAPE-argmin vs + τ* | 0–1 pt on unimodal objects, several pts on mixed categories | Minutes |
| P0 | Scale-cue gate (EDA #5–7) | Permutation tests | Decides whether P1 backbone work is worth its cost | 1 day |
| P1 | Backbone and resolution | DINOv2-L-reg vs DINOv3-L vs SigLIP 2 vs PE-L vs concat; 224/448/518/768; layer sets | 0–4 pts, conditional on the cue | Days |
| P1 | Head | L1 on ln V vs HL-Gauss vs CORN; K ∈ {64, 128, 256}; σ/ς ∈ {0.5, 0.75, 1.0} | 0–2 pts | Hours |
| P1 | Fine-tuning regime | Frozen vs LP-FT vs LoRA r ∈ {8, 16, 32} vs full FT + LLRD | 0–3 pts, high variance | Days |
| P1 | Auxiliary targets | none vs bbox vs fill vs all; weights 0.05–0.3 | 0–2 pts | Hours |
| P1 | Retrieval features | with vs without k-NN stats | 0–2 pts | Hours |
| P2 | Augmentation | safe set vs + vflip vs + RandomResizedCrop (doubles as a texture-scale-cue probe) | ±1–2 pts | Hours |
| P2 | Imbalance | sqrt-inverse LDS; RankSim | ≤1 pt | Hours |
| P2 | MoGe-2 branch | off vs log-scale only vs full | 0–1 pt unless the cue test finds signal | 1 day |
| P2 | Shape factor (Alternative C) | off vs TRELLIS.2 features | ≤ EDA #8 bound (≤~6 pts, likely less) | 2–4 days |
| P2 | Text priors for rare categories | coarse parent vs LLM prior | ≤1 pt overall, more on ≤3-sample categories | 1 day |
| P3 | Calibration and ensembling | T, δ, τ*; mixture-of-distributions vs point averaging; member count | 0.5–2 pts | Hours |

## Eight weeks to January 1

| Week | Dates | Goals | Exit criteria | Leaderboard use |
|---|---|---|---|---|
| 1 | Nov 4–10 | Manifest, label derivation, folds; OOF category baselines; mode anatomy (EDA #1–3); send organiser questions (folder category, group token, text priors, test-time training, zip format) | OOF baseline table; Δ and π̃_c estimated | One baseline submission to learn the zip format and confirm no item scores 200% as invalid |
| 2 | Nov 11–17 | Feature cache for DINOv2, SigLIP 2, MoGe-2 (and DINOv3 if cleared); design B; scale-cue and mode tests (EDA #4–10) | Go/no-go on backbone investment; design B OOF | 1–2 submissions to check CV–LB agreement |
| 3 | Nov 18–24 | Design A v1: HL-Gauss base head, mode head, EB priors, sMAPE decoder; P0 ablations | Design A beats B on OOF at 3 seeds | Sparing |
| 4 | Nov 25–Dec 1 | P1 ablations: backbone, resolution, head, fine-tune regime, auxiliaries, retrieval | Frozen recipe for A | Sparing |
| 5 | Dec 2–8 | P2 branches gated by EDA: MoGe-2, TRELLIS.2 shape factor, text priors; LDS/RankSim | Keep only branches with ≥1 pt OOF at 3 seeds | Sparing |
| 6 | Dec 9–15 | Full 5×3 retrains; distribution-space ensemble; T/δ/τ* calibration; bootstrap confidence intervals | Final candidate ensembles A (best CV) and B (robust/diverse) | Confirm both |
| 7 | Dec 16–22 | Reproducibility: Docker, determinism, clean-machine rebuild, licence audit, code-handover documentation | Rebuild matches within tolerance | None needed |
| 8 | Dec 23–Jan 1 | Freeze; private-test submissions of the two finalists; holiday buffer | Submitted before deadline with margin | Final |

## Unverified claims and conflicting numbers

| Item | Status / conflict | Handling in this report |
|---|---|---|
| sMAPE "optimal point sits slightly above the median" (backbone notes) | **Contradicted** by the log-space identity 2·tanh(\|Δln\|/2): for log-symmetric predictive distributions the optimum is exactly the median | Use the median / sMAPE-argmin |
| Oversized factor "1000×" (regression notes) vs measured ~200–1000× relative to category | Conflict | Estimate Δ from data; prefer a full-range binned head if the factor is not a clean 1000 |
| Within-category σ = 0.52 | This is the *median* per-category value; the floor varies by category | Use per-category σ̃_c |
| MoGe-2 latency: ~29 ms at 1,200 tokens (paper) vs "60 ms per image" (repo) | Different settings; both plausible | Budget 60 ms |
| MoGe-2 venue and weights licence | The geometry notes call them unconfirmed; the team verified NeurIPS 2025 and MIT | Team verification wins |
| Nutrition5k image-only mass MAE | 40.4 g / 18.8% (Table 3) vs 38.1 g / 29.5% (Table 4) ([paper](https://arxiv.org/abs/2103.03375)) | Insight unaffected |
| Size Matters internal numbers | 59.09 vs 61.24 mL; 34.83% vs 35.83%; 3D-Assisted 123.34 vs 186.45 mL and 104.07% vs 83.26% (79.33% in MFP3D); random split 82.46% may be a different split | Quote 35.83% with caution |
| Size Matters code | One note checked the GitLab URL exists; another did not open it | Treat as existing, contents unverified |
| PortionNet 17.43% | Unreviewed, no code; ~20 images per object raises a possible per-image multi-view evaluation leak (unconfirmed) | Not used as a target |
| Implicit-scale benchmark size | 10 scenes / 24 objects (paper) vs 14 scenes / 34 items (OneVol README) | Context only |
| MDMS = Hillo World, SGPS = OneVol | Inferred from matching numbers and authors | Labelled "≈" |
| SimpleFood45 vs "SimpleFood40" | arXiv vs CVF abstract | Immaterial |
| Dataset sizes | MetaFood3D 743 / 131 (arXiv) vs 637 / 108 (papers) vs 628 here; OmniObject3D 6,000 vs 3,417 (Size Matters) vs 2,776 here | Published per-dataset numbers are rough guides |
| DINOv3 licence | Official custom licence vs an unofficial site claiming CC-BY-NC | Official wins; legal review |
| Marigold v1.1 licence | Apache-2.0 for v1-0 vs a third-party claim of Open RAIL++-M for v1.1 | Use v1-0 only |
| CSIRO 1st-place recipe, Pawpularity winner | From search summaries and secondary write-ups only | Precedent, not evidence |
| Prior-knowledge venues: HL-Gauss ICML 2024, Stewart AISTATS 2023, ConR ICLR 2024, Depth Pro ICLR 2025, UniDepthV2, SAM 3D CVPR 2026, WildMoGe ECCV 2026 | Not confirmed on fetched pages | Marked * |
| Generator training sets (TRELLIS, SAM 3D) possibly containing OmniObject3D scans | Unverified | Check before using them for test inference |
| Leaderboard standard error of ~2.4 pts | My simulation under an assumed error mix | Re-estimate by bootstrapping real OOF errors |

## Conclusion

Absolute size cannot be seen in these renders, so the score is mostly settled before any pixel is processed: by how well the category prior is pooled, by how the metric is decoded, and by how the unit-error mode is bet on. The research shifts the engineering question from "which backbone sees size best?" to "is there any within-category scale cue at all, and is the oversized mode visible?" The scale-cue and mode-identifiability tests answer both within two weeks, using only the training meshes and frozen features. If both come back null, the winning entry is a meticulously calibrated Bayesian lookup table with a neural residual, and compute should shift from fine-tuning to priors, calibration and reproducibility. If the tests find a renderer artefact, such as absolute-unit lighting, shadow softness or texture granularity, it is the only lever that can push below the roughly 40% category floor. Every augmentation and preprocessing choice must then be checked so it does not erase that cue.

The second implication is procedural. With a ~2.4-point public-leaderboard standard error, 48 near-singleton categories, and ~10% of labels sitting about 200× away, model selection by leaderboard is a coin toss. Out-of-fold distributions decoded under the true metric are the only reliable compass. The organiser rulings on the folder-path category and group token, and on offline text priors, are worth more than any architecture choice in this report. They should be requested on day one.

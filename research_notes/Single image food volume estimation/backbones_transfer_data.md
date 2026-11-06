# Foundation backbones, transfer-learning recipes and data/validation practice for small-data single-image food-volume regression

Verification legend (used on every bullet):
- **[V]** = checked in this session (Nov 2026) against the primary page (official GitHub repo, model card, licence page, or search result pointing to the primary source).
- **[K]** = taken from well-known primary literature that I did not re-fetch in this session. The canonical arXiv/repo link is given, but the report writer should treat the specific numbers as "not re-verified".
- **[I]** = my inference (in the Inferences subsections only).

Competition recap (constraints the recommendations below are tuned to): 2,042 train images (1024², Blender, grey backdrop, auto-framed canonical view, no intrinsics), 164 categories (median 8/category), random split for eval/test with the same categories, sMAPE metric, training-only privileged info (mesh, category, coarse category, bbox_mm, weight_g for MetaFood3D), organizers re-run code, 1–4 GPUs of 24–80 GB.

---

## 1. Backbones: which foundation models are best for fine-grained recognition and physical-attribute regression from frozen features, and what are their licences?

### Takeaway
Self-supervised DINO-family models (DINOv2, DINOv3) are the default first choice for frozen-feature regression and are what top teams used in the 2025–26 CSIRO Image2Biomass image-regression Kaggle competition. Pair them with one language-supervised encoder (SigLIP 2 or Perception Encoder, both Apache-2.0) for category/semantic signal. Optionally add RADIO, which distills DINOv3, SigLIP 2 and SAM into one model; the C-RADIO versions are commercially licensed. Licences differ a lot. DINOv2, SigLIP 2 and PE are Apache-2.0. DINOv3 uses a custom Meta licence with gated download. MAE weights and ConvNeXt V2 ImageNet weights are CC-BY-NC. Non-C RADIO models are non-commercial (NSCL). I found no credible open food-specific foundation model.

### Cited Findings

**DINOv2 (Oquab et al., TMLR 2024; arXiv 2304.07193)**
- Code and weights are Apache License 2.0. The exceptions are Cell-DINO (code CC-BY-NC, weights FAIR Noncommercial) and XRay-DINO (weights FAIR Noncommercial). [V] — [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- Sizes: ViT-S/14 (21M), B/14 (86M), L/14 (300M), g/14 (1.1B). Each has a "with registers" (reg4) variant. [V] — [dinov2 repo](https://github.com/facebookresearch/dinov2)
- ImageNet-1k k-NN / linear on frozen features: ViT-S 79.0/81.1, B 82.1/84.5, L 83.5/86.3, g 83.5/86.5. With registers: L 83.8/86.7, g 83.7/87.1. The README says the features "can be directly employed with classifiers as simple as linear layers". [V] — [dinov2 repo](https://github.com/facebookresearch/dinov2)
- The repo README gives no Food-101 or iNaturalist numbers. The DINOv2 paper does report fine-grained linear-probe benchmarks, including Food-101 (paper tables not re-fetched). [V for README absence; K for paper] — [arXiv 2304.07193](https://arxiv.org/abs/2304.07193)
- Registers paper: "Vision Transformers Need Registers", Darcet et al., ICLR 2024 (arXiv 2309.16588). Registers remove high-norm artifact tokens in the patch feature maps, which matters when pooling patch tokens. [V for arXiv link from repo; K for venue] — [arXiv 2309.16588](https://arxiv.org/abs/2309.16588)

**DINOv3 (Siméoni et al., arXiv 2508.10104, Aug 2025)**
- Released ViT backbones (LVD-1689M web data, all patch 16): ViT-S, S+, B, L, H+ (distilled) and ViT-7B. Also ConvNeXt T/S/B/L, plus satellite (SAT-493M) ViT-L/16 and ViT-7B/16. [V] — [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- Licence is the custom "DINOv3 License". Download is gated: you request access through Meta and receive weight URLs by email. [V] — [dinov3 repo](https://github.com/facebookresearch/dinov3); [DINOv3 License page](https://ai.meta.com/resources/models-and-libraries/dinov3-license)
- The licence appears to allow commercial use, with conditions: include the agreement when distributing, show "Built with DINOv3", acknowledge in publications, indemnify Meta, and delete the materials on termination. A GitHub issue notes it is custom and needs legal review. DINOv2, by contrast, was moved to Apache-2.0. [V] — [Meta DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license); [dinov3 issue #31](https://github.com/facebookresearch/dinov3/issues/31)
- Conflicting source: a non-official site (dinov3.pages.dev) claims CC-BY-NC 4.0. It is not a Meta source and should be ignored. [V] — [search result](https://dinov3.pages.dev/privacy-policy)
- timm mirrors DINOv3 weights with a LICENSE.md on the HF repo (e.g. `timm/vit_small_plus_patch16_dinov3.lvd1689m`). This can simplify reproducible download for organizers, subject to the same licence. [V, existence only] — [timm DINOv3 LICENSE](https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m/blob/main/LICENSE.md)
- The README stresses high-quality dense features and strong results "without fine-tuning". The default transform resizes to 256×256. The depth and segmentation examples use 768–1024 px and 896 px. [V] — [dinov3 repo](https://github.com/facebookresearch/dinov3)

**SigLIP / SigLIP 2**
- SigLIP 2: "Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features", arXiv 2502.14786 (Feb 20, 2025). The HF card lists Apache-2.0. A NaFlex (native aspect/resolution) variant exists, e.g. so400m-patch16-naflex (~1B params incl. text tower). [V] — [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
- SigLIP (Zhai et al., ICCV 2023, arXiv 2303.15343). Sigmoid loss. Code in google-research/big_vision (Apache-2.0). [K] — [arXiv 2303.15343](https://arxiv.org/abs/2303.15343); [big_vision](https://github.com/google-research/big_vision)
- In CSIRO Image2Biomass, public baselines combined DINOv3 and SigLIP features. [V, existence of notebook] — [Kaggle notebook "dinov3+siglip"](https://www.kaggle.com/code/llkh0a/cisro-baseline-train-infer-21-12-dinov3-siglip)
- A community Food-101 classifier fine-tuned from `siglip2-base-patch16-224` (Apache-2.0) reports ~0.90 macro-F1. This is self-reported and not a benchmark. [V] — [prithivMLmods/Food-101-93M](https://huggingface.co/prithivMLmods/Food-101-93M/blob/main/README.md)

**CLIP / OpenCLIP / EVA-CLIP**
- CLIP (Radford et al., ICML 2021, arXiv 2103.00020). Its paper reports strong zero-shot and linear-probe Food-101 accuracy. Exact numbers not re-fetched. [K] — [arXiv 2103.00020](https://arxiv.org/abs/2103.00020)
- OpenCLIP (MIT licence; "Reproducible scaling laws for contrastive language-image learning", CVPR 2023, arXiv 2212.07143). It provides LAION/DataComp/DFN-trained ViTs. [K] — [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
- EVA-CLIP (arXiv 2303.15389), code in baaivision/EVA (MIT, per my recollection, not re-checked). [K] — [baaivision/EVA](https://github.com/baaivision/EVA)
- PetFinder Pawpularity (Kaggle 2021, small-data image regression): according to a secondary Japanese summary, the 1st-place team did not fine-tune. They fed frozen pretrained features from several models into SVR and used OpenAI-CLIP-derived metadata, with hill-climbing over concatenated features. The best single model in the 1st and 6th solutions was BEiT. [V for the summary; original Kaggle post not fetched] — [Zenn summary](https://zenn.dev/kohecchi/articles/efc6d291dec289)

**Perception Encoder (Bolya et al., Meta, arXiv 2504.13181, Apr 2025; "The best visual embeddings are not at the output of the network")**
- Code and model weights are Apache-2.0. PLM is separate, under the FAIR Research License. [V] — [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models)
- Variants: PE-Core (CLIP-style) T/16-384, S/16-384, B/16-224, L/14-336, G/14-448. PE-Lang L/14 and G/14 at 448. PE-Spatial G/14-448, plus distilled T/S/B/16-512 and L/14-448. [V] — [perception_models](https://github.com/facebookresearch/perception_models)
- PE-Spatial-G linear-probe ADE20k mIoU is 49.3. The repo claims PE spatial "outperforms DINOv2 on dense prediction tasks". The paper's central claim is that the best features are in intermediate layers, not at the output. [V] — [perception_models](https://github.com/facebookresearch/perception_models)
- Venue: technical report. No peer-reviewed venue is listed on the repo. [V]

**RADIO / AM-RADIO (NVIDIA)**
- AM-RADIO was CVPR 2024 (arXiv 2312.06709) and RADIOv2.5 was CVPR 2025 (arXiv 2412.07679). The C-RADIOv4 tech report is arXiv 2601.17237 (released Jan 27, 2026). C-RADIOv4 (SO400M and H) distills SigLIP2-g-384, DINOv3-7B and SAM3. C-RADIOv3 comes in B/L/H/g. [V] — [NVlabs/RADIO](https://github.com/NVlabs/RADIO)
- Licence: the C-RADIO models (v3, v4) use the NVIDIA Open Model License, which the repo says "enables commercial use cases". RADIO and E-RADIO (non-C) use NSCL, which is non-commercial. [V] — [NVlabs/RADIO](https://github.com/NVlabs/RADIO)

**ConvNeXt V2 (Woo et al., arXiv 2301.00808; CVPR 2023 [K for venue])**
- Code is MIT. ImageNet pre-trained and fine-tuned models are CC-BY-NC. FCMAE self-supervised weights cover Atto (3.7M) to Huge (660M). IN-22k fine-tuned: Base 86.8/87.7 (224/384), Large 87.3/88.2, Huge 88.7/88.9 (384/512). [V] — [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)

**MAE (He et al., arXiv 2111.06377; CVPR 2022 [K for venue])**
- CC-BY-NC 4.0 licence. ImageNet fine-tuned top-1: ViT-B 83.6, L 85.9, H 86.9, H-448 87.8. The README gives no linear-probe numbers. MAE features are known to be weak under linear probing and need fine-tuning [K]. [V for licence and numbers] — [facebookresearch/mae](https://github.com/facebookresearch/mae)

**Food-specific models / food-volume literature**
- A search for a 2025 open-weight food foundation model found only community Food-101 classifiers built on general backbones (SigLIP2, ViT, EfficientNetV2, ResNet50) with self-reported metrics. [V] — [search results, e.g. Food-101-93M](https://huggingface.co/prithivMLmods/Food-101-93M/blob/main/README.md)
- Nutrition5k (Thames et al., CVPR 2021): adding a depth-derived volume signal cut mass MAE from 38.1 g to 29.4 g for an InceptionV3 model. This shows geometric/volume information is a strong auxiliary signal. [V, search snippet] — [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf)
- A 2025 Nutrition5k study (arXiv 2508.03996) found a COYO-pretrained model performed worse than an ImageNet-pretrained one for this regression. Bigger web pretraining is not automatically better for physical-quantity regression. [V, search snippet only] — [arXiv 2508.03996](https://arxiv.org/abs/2508.03996v2)
- PortionNet (arXiv 2512.22304, late 2025) distills 3D point-cloud geometric features into a lightweight image adapter, so inference needs only a single RGB image. This is directly analogous to using the training-only meshes here. Backbone not verified. [V, search snippet] — [arXiv 2512.22304](https://arxiv.org/pdf/2512.22304)
- MetaFood CVPR 2024 challenge (arXiv 2407.09285): 20 food items. The "hard" tier had one image per item and a checkerboard scale reference. Volume was scored by MAPE against scanned meshes, and 16 teams reached the final phase. Single-view volume there relied on an explicit scale reference, which this competition lacks. [V] — [arXiv 2407.09285](https://arxiv.org/abs/2407.09285)

### Inferences
- [I] Backbone shortlist for the frozen-feature stage, in priority order:
  1. DINOv3 ViT-L/16 or H+/16 (or 7B for feature extraction only, if 80 GB is available).
  2. DINOv2-g/14-reg (Apache-2.0, the easiest licence, and a robust fallback).
  3. SigLIP 2 so400m or PE-Core-G/L (Apache-2.0), for category semantics.
  4. C-RADIOv4-H (one model carrying DINOv3 + SigLIP2 + SAM3 knowledge, commercially licensed).

  Drop MAE and ConvNeXt V2 ImageNet weights if winners must be commercially usable (CC-BY-NC). MAE is also poor frozen.
- [I] Reproducibility risk: DINOv3 weights are gated. Check whether the organizers can obtain them, e.g. through the timm/HF mirror plus licence acceptance. Otherwise keep an Apache-2.0 fallback (DINOv2/PE/SigLIP2) whose score is known.
- [I] Physical-attribute regression needs shape/extent and texture-scale cues, which self-supervised dense features preserve better than CLIP-style global embeddings. CLIP-style models add category identity, which in this dataset (same categories in train and test, median 8 images per category) is a major source of variance. Concatenating one of each is the rational default.
- [I] Following the Perception Encoder finding, extract features from several intermediate layers (and CLS + mean-patch pooling). The CSIRO frozen-DINOv3 solution also used multiple layers and scales (Section 2).

### Gaps
- I found no verified open food-domain foundation model (e.g. one trained on Food2K/Food-101 at scale with a permissive licence). If one exists it was not surfaced.
- I did not verify per-backbone numbers on Food-101 linear probes from the DINOv2, DINOv3, SigLIP2 or PE papers. The tables are in the PDFs, which were not fetched.
- EVA-CLIP and OpenCLIP licences were not re-verified in this session (stated from memory as MIT).
- I found no published head-to-head of these backbones specifically for volume or size regression from frozen features.

---

## 2. Transfer recipes for ~2k images: linear probe, full fine-tune, LP-FT, LoRA/PEFT, LLRD, WiSE-FT, soups, feature concatenation, kNN/retrieval, few-shot/prototypes

### Takeaway
With 2,042 images, the strongest and most reproducible baseline is frozen multi-layer, multi-scale features from one or two foundation models with a ridge/SVR/GBM or small MLP head, plus kNN/category-prior features. A frozen DINOv3-7B + Ridge stack scored competitively in CSIRO Image2Biomass, and frozen features + SVR won PetFinder Pawpularity. Fine-tuning can then add on top, but it should be done gently: LP-FT, LoRA/DoRA, or full fine-tuning with layer-wise LR decay. Then average weights (WiSE-FT, or a soup of seeds/folds) or average predictions across folds and seeds.

### Cited Findings
- **CSIRO Image2Biomass (Kaggle, 2025–26)**:
  - The 1st-place writeup is titled as a dual-stream DINO-like model with interval classification and test-time online training. It uses a shared DINOv3-Large backbone over left/right image halves and CV grouped by state and sampling date. The author suspects most top teams used DINOv3. [V via search summary; the Kaggle page did not render for fetch] — [1st Place Solution](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution)
  - A 32nd-place writeup used DINOv3 + DoRA + Tweedie loss. [V, title] — [Kaggle writeup](https://www.kaggle.com/c/csiro-biomass/writeups/dinov3-dora-tweedie-loss-ballanced-r2)
  - A frozen-only solution uses DINOv3 ViT-7B/16 (lvd1689m), "not fine-tuned", with features at 256/512/768 px from several intermediate layers. Per-view Ridge (α=5) is stacked with a Ridge meta-model (α=1) on OOF rows, in both raw and log1p target space. Results are averaged over 2/3/5-fold configurations. It scored public 0.757 / private 0.648 (placement not stated). It warns that the stacked training score is optimistic. [V] — [Forrest-xlz/CSIRO-kaggle](https://github.com/Forrest-xlz/CSIRO-kaggle)
  - A related paper, "Dynamic LoRA Fine-Tuning of DINOv3 for Multi-Component Pasture Biomass Estimation", exists on PMC (not read). [V, existence] — [PMC13517550](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517550/)
- **PetFinder Pawpularity**: frozen features from several pretrained models + SVR, with hill-climbing over feature/model combinations, took 1st place. The best single backbone was BEiT. [V, secondary summary] — [Zenn](https://zenn.dev/kohecchi/articles/efc6d291dec289)
- **LP-FT**: Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution", ICLR 2022 (arXiv 2202.10054). Full fine-tuning with a randomly initialized head distorts pretrained features. Linear-probing first and then fine-tuning (LP-FT) gives better OOD and comparable-or-better ID accuracy. Code at AnanyaKumar/transfer_learning. [K] — [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)
- **WiSE-FT**: Wortsman et al., "Robust fine-tuning of zero-shot models", CVPR 2022 (arXiv 2109.01903). It linearly interpolates the zero-shot/pretrained and fine-tuned weights, θ = (1−α)θ0 + αθ1, which improves robustness at no inference cost. [K] — [arXiv 2109.01903](https://arxiv.org/abs/2109.01903); [mlfoundations/wise-ft](https://github.com/mlfoundations/wise-ft)
- **Model soups**: Wortsman et al., ICML 2022 (arXiv 2203.05482). Averaging weights of models fine-tuned from the same initialization with different hyperparameters ("greedy soup", ordered by held-out accuracy) beats the best single model at no extra inference cost. [K] — [arXiv 2203.05482](https://arxiv.org/abs/2203.05482); [mlfoundations/model-soups](https://github.com/mlfoundations/model-soups)
- **LoRA**: Hu et al., ICLR 2022 (arXiv 2106.09685). **DoRA**: Liu et al., ICML 2024 (arXiv 2402.09353). Both are implemented in HF PEFT (Apache-2.0). **VPT** (visual prompt tuning): Jia et al., ECCV 2022 (arXiv 2203.12119). [K] — [LoRA](https://arxiv.org/abs/2106.09685); [DoRA](https://arxiv.org/abs/2402.09353); [huggingface/peft](https://github.com/huggingface/peft); [VPT](https://arxiv.org/abs/2203.12119)
- **Layer-wise LR decay**: used in BEiT (ICLR 2022, arXiv 2106.08254) and in MAE fine-tuning (FINETUNE.md uses layer_decay ≈0.65 for ViT-B and 0.75 for ViT-L/H, from my recollection; the fetched README did not include the values). [K] — [MAE repo](https://github.com/facebookresearch/mae); [BEiT](https://arxiv.org/abs/2106.08254)
- **kNN on pretrained features**: Jia et al., "Rethinking Nearest Neighbors for Visual Classification" (arXiv 2112.08459). kNN on pretrained features can beat a linear classifier, and interpolating kNN with a linear classifier helps most in low-data and hard settings. Code at KMnP/nn-revisit. The NLP analogue is "Revisiting k-NN for Fine-tuning Pre-trained Language Models" (arXiv 2304.09058, CCL 2023; code zjunlp/Revisit-KNN). [V, search] — [arXiv 2112.08459](https://arxiv.org/abs/2112.08459v2); [arXiv 2304.09058](https://arxiv.org/abs/2304.09058v2)
- DINOv2 and DINOv3 both report k-NN accuracy close to linear-probe accuracy on frozen features: DINOv2-L 83.5 vs 86.3, and the short-schedule DINOv3-L IN-1k model 82.0 vs 83.5. So kNN on frozen DINO features is a strong baseline. [V] — [dinov2](https://github.com/facebookresearch/dinov2); [dinov3](https://github.com/facebookresearch/dinov3)

### Inferences
- [I] Recommended ladder, each rung validated with the same grouped-or-stratified CV (Section 5):
  - **R0.** Category-median and category kNN baseline. Volume is strongly category-determined, and every test category appears in training.
  - **R1.** Frozen features (DINOv3 + SigLIP2/PE, several layers, CLS + mean-patch) followed by PCA/standardize and a head (ridge, kernel ridge/SVR, LightGBM, or a 2-layer MLP), trained on log(volume).
  - **R2.** R1 plus explicit kNN-retrieval features: neighbour log-volumes and similarity-weighted mean.
  - **R3.** LP-FT or LoRA/DoRA (rank 8–32 on attention/MLP, last N blocks) on DINOv3-L/B at 448–518 px, with LLRD 0.7–0.85, few epochs, and EMA.
  - **R4.** WiSE-FT interpolation between the LP head and the FT weights, plus a uniform or greedy soup over seeds.
  - **R5.** Stack the OOF predictions of R1–R4 with ridge or hill-climbing.
- [I] Fine-tuning gains from 2k images are likely small and high-variance. Judge them only on repeated-CV OOF sMAPE, never on the public LB.
- [I] Prototypes / few-shot: for categories with very few examples, use the coarse-category prior and similarity-weighted prototypes as features (shrinkage toward the coarse-category mean). This is a hierarchical-Bayes-style empirical-Bayes estimate rather than a published method.
- [I] Model soups only work across fine-tunes that share an initialization and stay in one loss basin. They do not apply to mixing different backbones. Use prediction-level ensembling for that.

### Gaps
- The full CSIRO 1st-place writeup could not be fetched (the Kaggle page rendered empty), so its exact fine-tuning recipe, augmentations and TTA are unverified.
- I found no published "retrieval-augmented regression" paper verified for vision in this session. The kNN-feature approach above is standard Kaggle practice, not a cited result.
- No direct evidence on LoRA vs full FT vs frozen for physical-quantity regression at n≈2k was found beyond the competition anecdotes.

---

## 3. Multi-task / auxiliary-target learning with privileged information (mesh, bbox_mm, weight_g, category)

### Takeaway
Privileged information is available only at training time: meshes, bounding-box size in mm, weight, and category. Two published routes exploit it. (a) Auxiliary heads, i.e. multi-task learning: predict log bbox dims, mesh-derived descriptors, category/coarse-category and weight jointly with log volume. (b) Distillation from a teacher that sees the privileged information (Lopez-Paz's generalized distillation, i.e. LUPI). Food-domain evidence supports geometry as the key signal: Nutrition5k's volume signal, and PortionNet distilling 3D features into an image adapter.

### Cited Findings
- **LUPI**: Vapnik & Vashist, "A new learning paradigm: Learning using privileged information", Neural Networks 22(5–6), 2009. Privileged features available only at training can speed learning. [K] — [DOI 10.1016/j.neunet.2009.06.042](https://doi.org/10.1016/j.neunet.2009.06.042)
- **Generalized distillation**: Lopez-Paz, Bottou, Schölkopf, Vapnik, "Unifying distillation and privileged information", ICLR 2016 (arXiv 1511.03643). Train a teacher on the privileged features, then train the student on the regular inputs to match the teacher's soft outputs mixed with the labels. [K] — [arXiv 1511.03643](https://arxiv.org/abs/1511.03643)
- **Deep LUPI**: Lambert et al., "Deep Learning Under Privileged Information Using Heteroscedastic Dropout", CVPR 2018 (arXiv 1805.11614). Privileged information controls dropout variance during training. [K] — [arXiv 1805.11614](https://arxiv.org/abs/1805.11614)
- **Knowledge distillation**: Hinton, Vinyals, Dean 2015 (arXiv 1503.02531). [K] — [arXiv 1503.02531](https://arxiv.org/abs/1503.02531)
- **Multi-task loss weighting**: Kendall, Gal, Cipolla, "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018 (arXiv 1705.07115). It uses learned homoscedastic-uncertainty weights per task. [K] — [arXiv 1705.07115](https://arxiv.org/abs/1705.07115)
- **Geometry teacher distillation in food**: PortionNet (arXiv 2512.22304) trains an image adapter to mimic point-cloud features so that inference uses a single image. [V, snippet] — [arXiv 2512.22304](https://arxiv.org/pdf/2512.22304)
- **Volume as auxiliary signal**: Nutrition5k, mass MAE 38.1 → 29.4 g with a volume signal. [V, snippet] — [Nutrition5k CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf)
- The CSIRO 1st place reportedly used interval classification, i.e. a classification-over-bins auxiliary for a regression target. [V, title via search] — [Kaggle 1st place](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution)

### Inferences
- [I] Cheap auxiliary targets derivable from the training mesh and metadata:
  - log bbox_mm (3 dims, sorted)
  - log volume / bbox-volume (a "fill ratio", in (0,1], which removes absolute scale)
  - mesh surface area, compactness/sphericity, PCA extents
  - category (164-way CE), coarse category
  - log weight_g (MetaFood3D only, masked loss)

  Decomposing volume = bbox-volume × fill-ratio separates the scale problem (hard, because of auto-framing) from the shape problem (easier from the image).
- [I] Because images are auto-framed, absolute scale is not directly visible from object image extent. The model must infer it from category priors, texture scale (grain/crumb size), lighting/shadow/perspective cues of the fixed Blender rig, and aspect ratio. An auxiliary bbox_mm head forces the network to encode whatever absolute-scale cues exist. Check this with the EDA in Section 5: plot the correlation of rendered silhouette area vs true bbox.
- [I] A geometry teacher is feasible here. Render or compute mesh descriptors (e.g. a point-cloud encoder on the mesh, or simply the hand-crafted descriptors above) for the teacher, then train the image student with a feature-matching or prediction-matching loss (generalized distillation). At test time only the image is used, which keeps the method legal.
- [I] Use masked losses for MetaFood3D-only weight_g, and down-weight auxiliary losses (e.g. 0.1–0.3), or use Kendall uncertainty weighting, to avoid hurting the main sMAPE.
- [I] Use frozen-feature variants as well: fit the main head on [features, OOF-predicted aux targets] (stacking). This is LUPI-via-stacking and is cheap to validate.

### Gaps
- I found no paper that specifically evaluates mesh-derived auxiliary heads for single-image volume regression on rendered data. The recommendations above are reasoned extrapolations.
- PortionNet's backbone, datasets and quantitative gains were not verified (full text not fetched).

---

## 4. Data processing and augmentation for rendered single-object images

### Takeaway
Any augmentation that changes apparent object size or framing must be treated as label-altering. That includes RandomResizedCrop, zoom/scale jitter, aggressive crops and, possibly, aspect-ratio warps. The safe set is horizontal flip, mild colour/brightness jitter, small rotations/translations that keep the object fully in frame, and blur/noise. Feed ViTs at moderately high resolution, 448–518 px (around 1k–1.3k tokens at patch 14/16). Multi-scale frozen features (256/512/768) worked in CSIRO. TTA should be flips plus a few resolutions.

### Cited Findings
- FixRes (Touvron et al., NeurIPS 2019, arXiv 1906.06423): RandomResizedCrop changes the apparent object size between train and test (the "train-test resolution discrepancy"), and fixing the test resolution or fine-tuning at test resolution helps. This is the mechanism by which crop augmentations alter scale cues. [K] — [arXiv 1906.06423](https://arxiv.org/abs/1906.06423)
- DINOv3's default transform is 256 px. Its dense-task examples use 768–1024 px and 896 px, which shows the backbone is used at high resolution. [V] — [dinov3 repo](https://github.com/facebookresearch/dinov3)
- The CSIRO frozen-DINOv3 solution extracted features at 256/512/768 and ensembled across scales. A reimplementation of the 1st place ran at 512×512 per half-view. [V] — [Forrest-xlz/CSIRO-kaggle](https://github.com/Forrest-xlz/CSIRO-kaggle); [search summary of reimplementation](https://github.com/wajihullahbaig/image2biomass-competition)
- SigLIP 2 NaFlex supports native aspect ratio and variable resolution. PE-Spatial distilled models are released at 512 px. [V] — [SigLIP2 NaFlex card](https://huggingface.co/google/siglip2-so400m-patch16-naflex); [perception_models](https://github.com/facebookresearch/perception_models)
- Registers remove artifact tokens. Prefer -reg variants when using patch-token pooling at high resolution. [K] — [arXiv 2309.16588](https://arxiv.org/abs/2309.16588)
- In CSIRO, a lower-ranked writeup reported that depth-model and per-pixel volume ideas gave no gains. [V, search summary only; low-rank anecdote] — [Kaggle writeup list](https://www.kaggle.com/competitions/csiro-biomass/writeups/how-we-approached-the-problem)

### Inferences
- [I] Because the view is canonical and auto-framed on a grey backdrop:
  - Full image vs crop: the full frame is roughly the object crop already. Do not tight-crop further, since the margin/fill ratio itself may encode shape.
  - If auto-framing normalizes object size, the silhouette fill fraction is a shape, not scale, feature. Compute it from a simple grey-background threshold as an explicit tabular feature.
- [I] Augmentation policy:
  - Safe: hflip; ±10° rotation with grey fill; ±5% translate; colour jitter (brightness/contrast ±0.1–0.2, slight hue/saturation, since food colour is a category cue and should not be destroyed); JPEG/blur/noise.
  - Risky (ablate): vflip, if the camera has a canonical elevation and gravity orientation, since vflip changes lighting/shadow semantics; aspect warps.
  - Avoid: RandomResizedCrop, scale jitter, mixup/cutmix on the regression target (mixing volumes is not linear in pixels).
- [I] Background: keep the grey backdrop. It is constant across train and test (same renderer), so removing it adds pipeline risk without benefit. If anything, add the mask-derived features above.
- [I] Resolution: 1024² source to 448 or 518 (patch 14 → 32²/37² tokens) is a good accuracy/compute point on 24–80 GB GPUs. Evaluate 224/448/768 frozen features and keep the best 1–2 scales.
- [I] Sim-to-sim: train, eval and test come from the same renderer and categories (random split). Domain shift is minimal, so heavy photometric augmentation for robustness is unnecessary. Its main value is regularization at n=2k.
- [I] TTA: hflip, plus 2–3 resolutions for frozen features. Average in log space. Avoid crop-based TTA for the same scale reasons.

### Gaps
- I found no published ablation of scale-destroying augmentations for physical-size regression on rendered images. The guidance is mechanism-based (FixRes) plus practice.
- I could not confirm whether the competition's auto-framing preserves any absolute scale cue (e.g. a fixed camera distance). This must be measured in EDA.

---

## 5. Data analytics and validation design (EDA, CV, adversarial validation, LB overfitting, ensembling, reproducibility)

### Takeaway
The eval/test split is random over the same categories, so CV should mirror that: stratified K-fold by category (and volume quantile), repeated over several seeds. Grouped-by-category CV is a pessimistic stress test to report alongside, not the main selector. Run near-duplicate detection with embeddings before splitting. Use adversarial validation to confirm train/eval exchangeability. Pick final models on repeated-CV OOF sMAPE with the public LB (511 images) only as a sanity check. Ensure deterministic, pinned environments, since the organizers re-run code.

### Cited Findings
- The CSIRO 1st place used a CV scheme grouped by state and sampling date, matching the hidden-test structure. [V, search summary] — [Kaggle 1st place](https://www.kaggle.com/competitions/csiro-biomass/writeups/1st-place-solution)
- The CSIRO frozen solution averaged 2/3/5-fold runs and warned that stacked training scores are optimistic. [V] — [Forrest-xlz/CSIRO-kaggle](https://github.com/Forrest-xlz/CSIRO-kaggle)
- Pawpularity top solutions used hill-climbing over OOF predictions to build ensembles. [V, secondary] — [Zenn](https://zenn.dev/kohecchi/articles/efc6d291dec289)
- Benchmark variance: Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021 (arXiv 2103.03098). Seed, data split and initialization variance is large, so they recommend randomizing multiple sources of variation and repeating. [K] — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- Adversarial validation as a documented method: Pan et al. (Uber), arXiv 2004.03045. Train a classifier to distinguish train from test; AUC ≈ 0.5 means no drift. [K] — [arXiv 2004.03045](https://arxiv.org/abs/2004.03045)
- Near-duplicate / copy detection with embeddings: SSCD (Pizzi et al., CVPR 2022, arXiv 2202.10261). DINO-family features are also commonly used for duplicate retrieval. [K] — [arXiv 2202.10261](https://arxiv.org/abs/2202.10261); [facebookresearch/sscd-copy-detection](https://github.com/facebookresearch/sscd-copy-detection)
- PyTorch reproducibility:
  - Set seeds for Python, NumPy and torch.
  - Call `torch.use_deterministic_algorithms(True)` and set `torch.backends.cudnn.benchmark = False`.
  - Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS.
  - Seed DataLoader workers via `worker_init_fn`/`generator`.
  - Full reproducibility is not guaranteed across releases, platforms or devices.

  [K] — [PyTorch Reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html)

### Inferences
- [I] EDA checklist:
  1. Volume distribution: expect a heavy right tail, so work in log space. Check the span in orders of magnitude.
  2. Per-category mean/std of log-volume, and the fraction of total variance explained by category (between/within). This determines how much a category prior alone achieves; compute the category-median sMAPE baseline.
  3. Within-category volume vs bbox_mm and fill-ratio. Check whether auto-framing removes the image-extent signal (correlate silhouette area with volume within category).
  4. Near-duplicates: cosine similarity of DINOv2/v3 CLS features. Flag pairs above roughly the 99.9th-percentile similarity, look at them, and check whether duplicates share a mesh with a different render or scale. If the same mesh appears at different scales with the same image, the label is unidentifiable from the image; quantify this noise floor.
  5. Mesh-level leakage: put renders of the same mesh in the same fold.
  6. Outliers: |log residual| from a category-median model; inspect mis-scaled meshes.
  7. Modes: multimodal volume within a category (e.g. slice vs whole).
  8. Source dataset (MetaFood3D vs others) as a stratum.
- [I] CV design:
  - Primary: StratifiedKFold (5 folds) on category, or on category × volume bin for categories with ≥5 examples, repeated with 3–5 different seeds. This matches the random eval/test split.
  - Keep groups by mesh ID if meshes are shared (StratifiedGroupKFold).
  - Secondary stress test: GroupKFold by category, which measures behaviour for unseen categories. It is not needed for the leaderboard but is useful for robustness claims.
  - Rare categories (<5): cannot be stratified cleanly. Use StratifiedGroupKFold with coarse-category stratification.
- [I] Metric handling: sMAPE = mean 2|y−ŷ|/(|y|+|ŷ|). For a fixed absolute error, over-prediction is penalized less than under-prediction, so the sMAPE-optimal point prediction sits slightly above the median of the predictive distribution.
  - Train on log-volume with L1/Huber, which approximates relative error.
  - Then fit a single global multiplicative calibration factor on OOF predictions to minimize OOF sMAPE. Optionally fit it per coarse category, only if stable across repeats.
- [I] Adversarial validation:
  - Train a LightGBM or logistic classifier on frozen embeddings plus simple image statistics for train vs eval-image membership.
  - With a random split, AUC should be ≈0.5. A higher AUC indicates a render-setting shift worth investigating.
  - Also run it on OOF folds as a check of fold balance.
- [I] Public LB discipline:
  - 511 public images is a small sample, so the sMAPE standard error is non-trivial. Estimate it by bootstrapping OOF errors on 511-sample subsets.
  - Treat LB differences smaller than about 2× that SE as noise.
  - Choose final submissions by CV, typically one best-CV ensemble and one more diverse or robust ensemble.
- [I] Ensembling:
  - Average fold models × 3–5 seeds in log space.
  - Combine heterogeneous backbones by ridge stacking or hill-climbing on OOF predictions.
  - Apply weight soups/WiSE-FT only within one fine-tuning family.
- [I] Reproducibility for organizer re-runs:
  - Pin Python, CUDA, torch, timm, transformers, peft and scikit-learn in a lock file (e.g. `uv.lock`) and ship a Dockerfile.
  - Pin model weights by HF revision hash and record SHA256s.
  - Cache extracted features with hashes.
  - Fix all seeds and use deterministic flags.
  - Log fold assignments to disk.
  - Provide a single `train.sh`/`predict.sh`.
  - Document gated-weight access for DINOv3 and keep an Apache-2.0 fallback.
  - Expect tiny GPU-level nondeterminism; ensemble averaging dampens it.

### Gaps
- I did not fetch Kaggle grandmaster writeups for Cassava or plant-traits competitions in this session. Only CSIRO (partial) and Pawpularity (secondary summary) are covered.
- I did not verify a peer-reviewed analysis of sMAPE-optimal point estimators. The asymmetry statement is an analytic inference from the formula.
- The actual dataset statistics (variance explained by category, duplicate rate, whether auto-framing preserves scale) are unknown until EDA is run on the competition data.

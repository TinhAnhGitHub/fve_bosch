# CNN and hybrid conv architectures vs ViTs for small-data image regression (food volume, ~2k images)

Note on method: web search returned summaries; the arXiv HTML page for DINOv3 could not be fetched, so DINOv3 per-model benchmark numbers are UNVERIFIED. Inferences are marked as such. No hardware timing was measured.

## 1. Evidence: CNN vs ViT transfer and fine-tuning on small data

### Takeaway
Evidence that CNNs beat ViTs in low-data fine-tuning is real but thin, and mostly compares ImageNet-supervised backbones (ConvNeXt vs Swin), not self-supervised DINO-class ViTs. At large scale, architecture matters little ("ConvNets Match ViTs at Scale"). A modern frozen self-supervised ViT remains the strongest baseline; CNNs are mainly a diversity play.

### Cited Findings
- "ConvNets Match Vision Transformers at Scale" (Smith, Brock, Berrada, De; Google DeepMind; arXiv Oct 2023): NFNets pretrained on JFT-4B, 0.4k-110k TPU-v4 core hours, follow a log-log scaling law; fine-tuned ImageNet top-1 up to 90.4%, matching ViTs at comparable compute. Conclusion: compute and data matter more than architecture. Venue not verified from a primary conference page (arXiv preprint) — [arXiv](https://arxiv.org/abs/2310.16764)
- ConvNeXt V2 (Woo et al., CVPR 2023, pp. 16133-16142): FCMAE sparse-conv masked autoencoder plus Global Response Normalization; plain ConvNeXt+MAE gave 83.7% vs 83.8% supervised for V1-B, with GRN reaching 84.9% (V2-B); sizes from Atto (3.7M, ~76.7% top-1) to Huge (650M, 88.9% public data). Code MIT, but pretrained/fine-tuned ImageNet weights CC-BY-NC per repo README — [CVF paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.pdf), [arXiv](https://arxiv.org/abs/2301.00808), [repo](https://github.com/facebookresearch/ConvNeXt-V2). timm hosts `convnextv2_*.fcmae_ft_in22k_in1k` (and `.fcmae` pretrain-only); the checkpoint licence field was not seen on the model cards (UNVERIFIED) — [timm card](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k)
- "Which Backbone to Use" (arXiv 2406.05612, 2024; venue UNVERIFIED): across natural, medical, galaxy, remote-sensing sets, attention-based backbones did poorly under low-data fine-tuning vs CNNs; ConvNeXt-Tiny best on almost all natural-image sets except CIFAR-10/100; recommends ConvNeXt, EfficientNet, RegNet over Swin in low data. Transformer tested was Swin, not plain ViT — [arXiv](https://arxiv.org/html/2406.05612v1)
- "ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy" (arXiv 2311.09215, ICML 2024 per ICLR-style proceedings listing UNVERIFIED): supervised ConvNeXt transfers comparably to CLIP models, has an advantage on synthetic data, but is more texture-biased than ViT — [arXiv](https://arxiv.org/html/2311.09215v2)
- Oral-disease benchmark (~10k train images, arXiv 2509.23100): ConvNeXt 81.06% vs BEiT 80.00, Swin 79.73, ViT 79.37, DeiT 78.79 — small gaps — [arXiv](https://arxiv.org/html/2509.23100v1)
- Practitioner blog claims ConvNeXt/EfficientNetV2 often beat ViT-B at 224 px without large-scale pretraining (secondary source) — [Label Your Data](https://labelyourdata.com/articles/image-classification-models)

### Inferences
- Our competitor is not a from-scratch or ImageNet-supervised ViT but a frozen DINOv2-L/DINOv3-H+ plus ridge/kNN; the "CNNs win in low data" studies do not transfer directly. The LoRA run already matching the frozen baseline suggests the bottleneck is label/scale ambiguity, not backbone inductive bias.
- Texture-bias of ConvNeXt (2311.09215) is mildly favourable for fine surface cues but unhelpful for silhouette/shape cues.

### Gaps
- No rigorous matched-pretraining, data-fraction study of ConvNeXt vs ViT; no published small-data regression benchmark found.
- Venues for 2406.05612 and 2311.09215 not confirmed from primary pages.

## 2. Self-supervised ConvNeXt backbones (DINOv3 ConvNeXt etc.)

### Takeaway
DINOv3 ships distilled ConvNeXt T/S/B/L (29M/50M/89M/198M) from the ViT-7B teacher, usable frozen or fine-tuned at 512 px in 6 GB; they are the most sensible conv candidates. They share the DINOv3 custom licence. Head-to-head kNN/retrieval numbers vs ViT students were not obtainable.

### Cited Findings
- DINOv3 ConvNeXt Tiny 29M, Small 50M, Base 89M, Large 198M, pretrained on LVD-1689M; ViT students S 21M, S+ 29M, B 86M, L 0.3B, H+ 0.8B — [GitHub](https://github.com/facebookresearch/dinov3), [HF small card](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)
- Paper (arXiv 2508.10104) says ConvNeXt students consistently outperform supervised ImageNet-22K counterparts; distilled-model comparison reportedly in Tab. 14 (numbers not seen) — [arXiv](https://arxiv.org/html/2508.10104v1)
- timm cost at 224 px: ConvNeXt-T 4.5 GMACs, S 8.7, B 15.4 (activations 13.4M / 21.6M / 28.8M). timm >= 1.0.20, transformers >= 4.56.0 support them — [timm small](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m), [base](https://huggingface.co/timm/convnext_base.dinov3_lvd1689m), [tiny](https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m)
- Licence: custom "DINOv3 License" on code and weights (not Apache/MIT); Meta describes it as commercial-permissive but full terms not read — check LICENSE.md in each model repo. DINOv2 is Apache 2.0. Distillation code for ConvNeXt released 2025-11-20 — [LightlyTrain docs](https://docs.lightly.ai/train/stable/pretrain_distill/models/dinov3.html), [HF LICENSE](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/blob/main/LICENSE.md)
- Model card warns fine-tuning may increase biases; frozen use is the intended path with kNN eval script at dinov3/eval/knn.py — [DINOv3 overview](https://www.lightly.ai/blog/dinov3)

### Inferences
- Memory: ConvNeXt-S/B at 512 px is ~5.3x the 224 px MACs (~46 / ~82 GMACs by arithmetic scaling, not measured); with bf16 and gradient checkpointing, ConvNeXt-B full fine-tune at batch 8 plausibly fits 5 GB; ConvNeXt-L is risky. Untested.
- Frozen ConvNeXt features give 4 pyramid stages; multi-stage pooled descriptors are cheap, no fine-tuning needed, and give a different inductive bias to add to the ridge/kNN stack.
- CLIP/SigLIP/OpenCLIP ConvNeXt, MAE-conv: no evidence gathered (gap).

### Gaps
- DINOv3 ConvNeXt vs ViT kNN/retrieval numbers; actual ms/epoch and VRAM on RTX A3000; licence full text.

## 3. Conv heads on ViT patch grids; what MoGe-2 is

### Takeaway
MoGe-2 is a DINOv2 ViT-L encoder with a conv neck and conv heads for dense geometry, but its own ablation shows that global metric scale is better predicted by an MLP on the CLS token than by a conv head. It is a dense-geometry model, not a template for a 2k-image global regressor.

### Cited Findings
- MoGe-2 (Wang et al., NeurIPS 2025): DINOv2 ViT-L (ViT-B in ablations), features from layers 6/12/18/final projected to a common dim; shared ConvStack neck (residual blocks + stride-2 transposed convs, norm layers removed, UV positional encoding) and per-target conv heads (points, normals, mask); metric scale from an MLP head on the CLS token passed through exp — [arXiv](https://arxiv.org/html/2507.02546v1), [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/336572db3e99930814d6b328d4220cb6-Paper-Conference.pdf)
- Ablation: a conv scale head sharing the neck did not improve relative geometry and worsened metric scale; MLP on CLS better (metric depth error 17.7 -> 16.5) — same arXiv source
- Code MIT with DINOv2 part Apache 2.0; weights licence not stated explicitly in repo (issue #98), HF page shows MIT — [issue](https://github.com/microsoft/MoGe/issues/98)
- Registers/CLS decoupling (NeurIPS 2025): in large DINOv2-with-registers, the global representation relies mainly on register tokens, so CLS may not summarise spatial patches; smaller models are better aligned — [paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/01bcbd34b02f3da9700a3ddd0480c156-Paper-Conference.pdf)
- DINOv3 overview suggests small heads on patch tokens for dense tasks and linear probe on CLS/pooled features for global tasks — [overview](https://medium.com/@xiaxiami/dinov3-a-deep-practical-overview-of-metas-new-self-supervised-vision-backbone-005311689dbf) (secondary)
- No direct benchmark found of attention-pooling or conv head vs CLS+mean for global regression on frozen DINO patches.

### Inferences
- MoGe-2 trains on very large mixed metric-depth datasets with dense supervision; with ~2k images and one scalar label per image, a conv decoder has almost no supervision signal; training from scratch is not sensible. At most reuse the frozen MoGe-2 encoder/neck as a feature or depth/normal-map extractor (inference only); but note it is the same DINOv2 family and its metric-scale ability would be anchored on the camera intrinsics assumptions that auto-framing breaks (reasoning, UNVERIFIED for these renders).
- Cheap experiment with positive prior: frozen patch tokens (32x32 at 512 px) -> mean + GeM + attention-pooling probe (a few 100k params) or a 3x3 conv + attention pool, compared with CLS+mean. Expected gain small; large-ViT CLS may lose spatial info per the registers paper.

### Gaps
- No evidence conv necks beat attention pooling for global regression.

## 4. Resolution

### Takeaway
No primary evidence found for volume regression specifically. General EfficientNet-style scaling argument: higher resolution helps fine-grained cues, with diminishing returns.

### Cited Findings
- EfficientNet compound scaling: higher resolution lets ConvNets capture finer patterns; returns diminish; must scale depth/width with it — [EfficientNet (ICML 2019)](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf) (pre-2023 but canonical)
- ConvNeXt/ResNet global-average pooling decouples parameter count from input size, allowing adaptation to higher resolution (secondary blog) — [Medium guide](https://mobasshirbhuiya.medium.com/a-comprehensive-guide-to-choosing-image-shapes-for-computer-vision-models-from-standard-practices-40b481d42da5)

### Inferences
- Since auto-framing normalises object size, extra pixels mostly add surface texture rather than scale; a 1024 px conv pass is likely of low value vs 512 px given the earlier finding that apparent size is uncorrelated with true size. Test 512 vs 768 once at the cheapest model.

### Gaps
- No food-volume or comparable resolution ablation located.

## 5. Silhouette/mask features and ensembling

### Takeaway
Ensembling diverse backbones is well supported; silhouette-based small models found no direct literature in this pass.

### Cited Findings
- CNN and ViT complement each other most in ensembled accuracy gain; lower prediction correlation predicts larger gains (arXiv 2504.09076) — [arXiv](https://arxiv.org/html/2504.09076)
- Heterogeneous small-data ensemble (Swin-T, ViT-B, DeiT-S on HAM10000) +2.54 pp over best single (all transformers) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13518149/)
- CLIP backbone ensembling gains correlate with diversity, tuned with as few as one example per class (arXiv 2405.17139) — [arXiv](https://arxiv.org/abs/2405.17139)

### Inferences
- Handcrafted mask descriptors (area fraction, bbox aspect, Hu moments, elongation, solidity, perimeter^2/area) from the grey-backdrop render are trivial to extract, add near-zero parameters, and feed ridge/GBM within category; they can capture shape factors that CLS features smear. Run them as an extra feature block, not a CNN.
- Gains from adding a conv backbone are plausible only if its residuals are decorrelated from DINO errors; check OOF residual correlation before investing.

### Gaps
- No literature found on silhouette/depth-map CNNs for single-image volume or on 2k-sample regime.

## 6. Recommendation

1. Experiment A (priority, cheap, frozen): DINOv3 ConvNeXt-B (`convnext_base.dinov3_lvd1689m`, 89M) at 512 px, bf16, batch 8-16 inference only; extract stage-3 and stage-4 GAP + GeM features (concatenated), plug into existing ridge + per-category + within-category kNN pipeline. Cost: minutes for 2.2k images. Expected gain: small (maybe 0.3-1 sMAPE points via ensembling; guess, not evidence). Risk: low; licence is DINOv3 custom, as for the ViT-H+ already used.
2. Experiment B (optional): attention-pooling or 3x3 conv + attention-pool probe over frozen DINOv3-L/H+ patch grid (512 px => 32x32), a few hundred thousand params, trained with 5-fold CV on log volume with category embedding. Expected gain: small to modest over CLS+mean; risk: overfitting at 2k samples, so use strong weight decay and compare with ridge on pooled patch statistics first.
3. Cheap add-on: mask/silhouette descriptor block (above).
- Skip: training MoGe-2 or its conv decoder, from-scratch CNNs, ConvNeXt-L fine-tune (memory), ConvNeXt-V2 FCMAE (non-commercial weights, weaker features than DINOv3, no evidence of advantage), 1024 px CNN passes, NFNet/ResNet-RS/RegNet (supervised ImageNet features, no evidence they beat DINO for near-duplicate matching).
- Fine-tuning ConvNeXt-T/S (LoRA-less full fine-tune, 512 px, batch 8, bf16 + checkpointing) is a fallback; only try if A shows the conv features are complementary. Time/epoch not measured here.

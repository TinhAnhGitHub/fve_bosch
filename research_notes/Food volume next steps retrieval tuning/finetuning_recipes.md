# Fine-tuning recipes for large ViTs (DINOv2/v3, SigLIP2 class) on ~2k images, regression, 6 GB GPU

Scope note: roughly 12 search calls; most results were search-engine summaries, not full-text reads. Several key questions found no direct ViT-L small-data evidence. I did not skim the earlier report. Items I could not source are in Gaps or marked UNVERIFIED. Recommendations are inferences, not literature claims.

## LP-FT, LoRA vs full fine-tuning vs adapters (DoRA, LoRA+, rsLoRA, VeRA, IA3), LLRD, which modules, rank

### Takeaway
Evidence supports (a) a cheap linear-probe or head warm-up before updating the backbone (LP-FT), (b) LoRA matching full fine-tuning on small data when applied broadly, with low rank enough on in-domain tasks, and (c) LLRD plus EMA as the standard stabilisers. I found no head-to-head comparison of DoRA, LoRA+, rsLoRA, VeRA or IA3 on ViT-L small-data regression.

### Cited Findings
- LP-FT (linear probe, then full fine-tune), Kumar et al., ICLR 2022 oral, arXiv 2202.10054, code github.com/AnanyaKumar/transfer_learning. Across 10 distribution-shift datasets fine-tuning is about 2% better ID but 7% worse OOD than linear probing. LP-FT is about 1% better ID and 10% better OOD than full fine-tuning, at about the same compute. The mechanism is a random head distorting pretrained features early in fine-tuning. Classification-focused; no regression evidence. — [arXiv](https://arxiv.org/abs/2202.10054)
- "LoRA Without Regret" (Thinking Machines, blog, LLM-focused): LoRA matches full fine-tuning on small-to-medium datasets when applied to all layers, especially MLP layers, and underperforms only when data exceeds adapter capacity. The optimal LoRA lr was about 10x the full fine-tuning lr. Not a ViT source. — [Blog](https://thinkingmachines.ai/blog/lora/)
- SR-LoRA (arXiv 2507.00327), ViT transfer on VTAB: on natural tasks (close to pretraining) LoRA is barely affected by rank and beats full fine-tuning; on specialized tasks with a large domain gap accuracy rises steadily with rank. Stable-rank-based allocation beats uniform LoRA r=32 at equal parameters. In 1-shot, full fine-tuning gets worse going ViT-B to ViT-L, indicating over-parameterised models overfit limited data. Venue not verified. — [arXiv](https://arxiv.org/pdf/2507.00327)
- AdvCLIP-LoRA (arXiv 2505.15130), few-shot CLIP ViT: adapters restricted to particular depths (top, bottom, middle) underperform adapters in all layers; Wq alone is weaker; higher rank helps most at 1-shot. — [arXiv](https://arxiv.org/html/2505.15130v3)
- Original LoRA paper adapts mainly Wq and Wv and left MLP adaptation to future work — [arXiv](https://arxiv.org/html/2106.09685v2)
- DINOv3 LoRA, MIDOG 2025 (arXiv 2508.21041, classification): r=8, alpha 16, dropout 0.05, q and v only (about 1.3M params), AdamW lr 1e-4, wd 0.1, cosine with 10% warmup, grad clip 1.0, 4-fold CV. — [arXiv](https://arxiv.org/html/2508.21041v1)
- DINOv3 LoRA regression, pasture biomass (Sensors 2026, 357 training images, skewed zero-inflated targets): full fine-tuning overfits, frozen backbone cannot adapt. A depth-aware LoRA whose rank and scale grow exponentially with layer depth (early layers barely changed) is their remedy. Single dataset, no mechanistic analysis. This is the closest published analogue to this task. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13517550/)
- LLRD: MAE and BEiT fine-tuning use layer decay 0.65 (ViT-B) and 0.75 (ViT-H, ViT-L-class), with drop path 0.1 to 0.3. MAE repo ViT-H recipe: layer_decay 0.75, drop_path 0.3, 50 epochs. SdAE ViT-L: lr 1e-3 base, decay 0.75, drop path 0.2, 5 warmup epochs. — [MAE FINETUNE.md](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md), [SdAE supp](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900107-supp.pdf)
- "MAE are Secretly Efficient Learners" reports LLRD matters a lot and its best decay value follows an increasing-peaking-decreasing pattern. — [OpenReview](https://openreview.net/pdf/840f7a6a46888bc481640ed2b41b908a1592b207.pdf)
- DoRA (weight-decomposed LoRA, arXiv 2402.09353) is evaluated on LLM commonsense and VL-BART or LLaVA, not on ViT classification or regression. — [arXiv](https://arxiv.org/html/2402.09353v5)
- ICCV 2025 AOFT paper (ViT-B/16, FGVC and VTAB-1k) compares Adapter, LoRA and VPT variants, not DoRA, LoRA+ or rsLoRA. — [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Efficient_Adaptation_of_Pre-trained_Vision_Transformer_underpinned_by_Approximately_Orthogonal_ICCV_2025_paper.pdf)
- Official DINOv3 repo issue: a maintainer advises extracting frozen features and checking downstream performance first. — [GitHub issue](https://github.com/facebookresearch/dinov3/issues/23)

### Inferences
- The current B4.0 is already near the evidence-backed region (all-layer attention LoRA, low rank). Rank is unlikely to be the bottleneck; the sources suggest rank matters mainly with large domain gaps. Renders of food on grey backdrop are probably a moderate gap, so r=16 to 32 is a sensible range; do not expect large gains from r changes.
- Adding MLP (fc1/fc2) LoRA is the best-supported module change (LoRA Without Regret, LLM evidence, UNVERIFIED for ViT), but it raises memory and overfitting risk.
- LP-FT analogue: train the head for 1 to 2 epochs with LoRA frozen (or init the head from ridge regression on frozen features) so the random head does not push gradients into LoRA. Our "head lr 1e-3 simultaneous" setup is the pattern LP-FT warns about. Cheap to test.
- LoRA lr 2e-4 is the standard LLM value; ViT sources use 1e-4 with wd 0.1. With LLRD-style depth scaling (lr or rank growing with depth) is a literature-backed alternative.
- Because held-out was still improving at epoch 10, the schedule is probably too short or lr too low for the adapter, rather than overfitting.

### Gaps
- No ViT-L small-data (about 2k) head-to-head of LoRA vs DoRA vs LoRA+ vs rsLoRA vs VeRA vs IA3. Searches returned LLM or VTAB-1k ViT-B material only. DoRA/rsLoRA/LoRA+ benefit on ViT: UNVERIFIED.
- No source found on adapting only last-N blocks of a ViT; only all-layers-beats-subset (CLIP few-shot).
- LLRD ablation specific to small-data regression: not found.
- Licences of PEFT (Apache-2.0) and timm (Apache-2.0) from memory, UNVERIFIED; DINOv3 uses a custom DINOv3 licence, not checked.

## Regularisation and stabilisers (EMA, SWA, soups, mixup/C-Mixup, drop path, wd, early stopping, seeds, TTA)

### Takeaway
EMA and LLRD are the best-evidenced stabilisers for fine-tuning big pretrained ViTs on small data; C-Mixup is the regression-specific mixup. I found no direct evidence for label smoothing for regression, model soups, or multi-seed averaging in this regime.

### Cited Findings
- C-Mixup, Yao, Wang, Zhang, Zou, Finn, NeurIPS 2022, arXiv 2210.05775: samples mixing partners using a Gaussian kernel on label distance, because vanilla mixup on regression labels can give arbitrarily wrong labels. Across 11 datasets: +6.56% ID, +4.76% task generalisation, +5.82% OOD. Code github.com/huaxiuyao/C-Mixup. Not evaluated on ViT-L fine-tuning with pretrained image backbones in the summary I saw. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html), [GitHub](https://github.com/huaxiuyao/C-Mixup)
- "CLIP Itself is a Strong Fine-tuner" (arXiv 2212.06138): EMA (momentum 0.9998) improved most settings and narrowed the gap when lr was badly chosen; LLRD plus EMA slow representation drift; fine-tuning peaked before epoch 50 then overfit; they dropped Mixup/CutMix. — [arXiv](https://ar5iv.labs.arxiv.org/html/2212.06138)
- MAE: linear probing turns off mixup, cutmix, drop path and colour jitter and uses wd 0 (relevant to a head-only warm-up phase). — [MAE supp](https://openaccess.thecvf.com/content/CVPR2022/supplemental/He_Masked_Autoencoders_Are_CVPR_2022_supplemental.pdf)
- AugReg (Steiner et al., TMLR 2022, arXiv 2106.10270): best stochastic-depth and augmentation settings vary strongly with model capacity and schedule. — [arXiv](https://arxiv.org/pdf/2106.10270)

### Inferences
- EMA over LoRA weights is nearly free on 6 GB (2.4M params) and should help given fold noise; with only about 200 steps per epoch, use a smaller momentum (about 0.99 to 0.999, tune to total steps), not 0.9998.
- C-Mixup on the size residual (pair images with near labels) is plausible, but mixing images of renders changes apparent volume non-linearly, so label mixing of volume is geometrically wrong. Risky; treat as low-priority, UNVERIFIED for this setting.
- Drop path (0.1 to 0.2) on a LoRA'd backbone is an untested but cheap regulariser.
- With fold noise of about 2 sMAPE points, multi-seed or fold-ensemble averaging and flip TTA are probably worth more than any single hyperparameter change; no literature found for exact gain.

### Gaps
- Model soups, SWA, label smoothing for regression, multi-seed ensembling, TTA gains for small-data ViT regression: no sources found in this session.

## Multi-task auxiliary targets, privileged information, loss weighting

### Takeaway
Multi-task heads help in nearby food and nutrition work, but I found no source on mesh-derived privileged targets or on loss weighting for tiny datasets.

### Cited Findings
- Multi-task food recognition plus portion regression on a small dataset (1,744 train / 424 test) with L2 soft parameter sharing; classification features give prior knowledge for portion regression (arXiv 2004.13188). — [arXiv](https://arxiv.org/html/2004.13188)
- Calorie prediction with macronutrient mass auxiliary regressions: multi-task model beat single-task by 9.9% relative error (secondary listing). — [Lacuna](https://lacuna.tiptreesystems.com/work/multi-task-learning-for-calorie-prediction-on-a-novel-large-scale-recipe-dataset/wrk_cf181d742a31025b59f45187892b33db)
- 2026 survey on food portion estimation treats auxiliary inputs and mapping image to depth as a route to portion regression (arXiv 2602.05078). — [arXiv](https://arxiv.org/html/2602.05078v1)

### Inferences
- Auxiliary weight 0.1 is conservative; since B4.0 only matched kNN, an ablation of aux weight (0, 0.1, 0.3) is cheap but fold noise may hide it. Ablate by isolating aux via a single run each only if time allows.
- Uncertainty weighting or GradNorm is unlikely to be worth the complexity with 5 loss terms and noisy folds.

### Gaps
- Kendall uncertainty weighting, GradNorm, LUPI and distillation in small-data ViT regression: not retrieved. Evidence that aux targets help or hurt in this setting: not found.

## Resolution, token count, registers, pooling, DINOv3 specifics

### Takeaway
No source quantified 224 vs 448 vs 512 vs 768 for regression. For pooling, [CLS plus mean-patch] is the standard DINOv2 linear-eval head; attention pooling is suggested only by non-benchmark material.

### Cited Findings
- DINOv2 linear eval uses a linear layer on the class token plus average of patch tokens (our current head matches it). — [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- DINOv2 authors say fine-tuning gives small gains (+2% on ImageNet-1k) and should be a last step. — [DINOv2 paper](https://arxiv.org/html/2304.07193v2)
- "Vision Transformers Need Registers" (ICLR 2024, arXiv 2309.16588): DINOv2 has a few high-norm outlier patch tokens (2.37% with norm >150) carrying global info; register variants exist. May affect patch-mean pooling. — [arXiv](https://arxiv.org/pdf/2309.16588)
- DINOv3 (Meta, arXiv 2508.10104, Aug 2025): Gram anchoring against dense-feature degradation; post-hoc resolution flexibility; outputs CLS, patch and register tokens. RoPE details not confirmed from snippets. — [arXiv](https://arxiv.org/abs/2508.10104), [HF card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- Attention pooling is described as often preferred for complex tasks, without benchmarks (Medium, weak). — [Medium](https://medium.com/@imabhi1216/beyond-cls-advanced-pooling-strategies-for-vision-transformers-8df1785ec81c)

### Inferences
- DINOv3 ViT-L/16 has registers, so mean-patch pooling should be cleaner than on DINOv2 without registers; make sure the mean excludes register tokens.
- The cost lever on 6 GB is tokens: 512/16 = 1024 patches. Dropping to 384 to 448 px would cut time per epoch roughly 2x or more, enabling more epochs or folds; whether accuracy drops is untested (auto-framed renders likely need less resolution than natural images). Test 384 vs 512 once.
- DINOv3 bf16 and RoPE pitfalls: no sourced pitfalls found. UNVERIFIED.

### Gaps
- Resolution scaling for regression, GeM or attention-pooling benchmarks, DINOv3-specific bf16 or RoPE issues: not found.

## Frozen features plus retrieval vs fine-tuning crossover

### Takeaway
Frozen DINOv2-L plus kNN is repeatedly shown to be strong at low data; I found no paper giving a dataset-size crossover for regression, so our "LoRA = kNN at 1.6k" is consistent with, not contradicted by, the literature.

### Cited Findings
- Frozen DINOv2 embeddings with kNN reach state-of-the-art few-shot accuracy and beat meta-learning on four benchmarks (May 2026 paper, secondary Pith review, UNVERIFIED primary). — [Pith](https://pith.science/paper/2605.14145)
- Frozen features can underperform for domains far from pretraining or for localisation-heavy tasks (secondary). — [Emergent Mind](https://www.emergentmind.com/topics/frozen-dinov2-encoder)
- SR-LoRA: full fine-tuning degrades from ViT-B to ViT-L at 1-shot. — [arXiv](https://arxiv.org/pdf/2507.00327)

### Inferences
- With 1.6k samples and a kNN-matching baseline, the highest-return option is probably to combine rather than choose: use fine-tuned embedding plus kNN or ridge on frozen features as an ensemble, and test stacking.

### Gaps
- Crossover curves vs N for regression: not found.

## Recommended B4.1 recipe (6 GB) and ablation order

### Takeaway
Treat as hypotheses from the evidence above; fold noise is 2 points, so only changes with a plausible effect above about 2 sMAPE are worth a full run. Expected gains below are guesses, not sourced.

### Cited Findings
- (Recipe elements are drawn from findings cited in sections above; no source gives an end-to-end recipe for this task.)

### Inferences
Order of ablations (each about 10 to 12 epochs x 6 min, or less at 384 px):
1. Head warm-up (LP-FT-style): 1 to 2 epochs head-only (or ridge-initialised head), then LoRA unfrozen; plus EMA of trainable weights (momentum about 0.995). Low cost; guess +0 to 2 points.
2. Longer schedule: 20 epochs with 1 to 2 warm-up epochs since the score was still improving; early stopping on EMA model. About 2x runtime. Guess +1 to 3.
3. Resolution 384 or 448 vs 512, to buy back time (about 30 to 50% per epoch saved, estimated from token count).
4. Add LoRA to MLP (fc1/fc2) at r=8 to 16, or depth-scaled rank; drop path 0.1.
5. Aux weight ablation (0 vs 0.1 vs 0.3); likely within noise.
6. Ensembling: fold or seed average plus flip TTA, and blend with frozen-feature kNN or ridge. Likely the most reliable gain, no training cost beyond runs.
Lower priority or skip: DoRA, LoRA+, rsLoRA, VeRA, IA3 (no ViT-L evidence found), C-Mixup (geometric label mixing concern), uncertainty weighting or GradNorm.

### Gaps
- All expected-gain figures and runtimes are unsourced estimates. No small-data DINOv3-L regression recipe with ablations exists in what I found beyond the pasture-biomass paper.

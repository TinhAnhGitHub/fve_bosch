# Retrieval / kNN / learned-metric regression for small-data image regression

Scope note: WebFetch was blocked for arxiv.org and github.com in this session, so all facts below come from search-result snippets only (no primary page opened). Venue/year statements come from search snippets that quote proceedings or repo pages; licences, last-commit dates and headline numbers were NOT verified. Anything not backed by a link is in Gaps.

## Q1. Methods that learn a metric/embedding so neighbours predict a continuous target (venue, year, numbers, repo, licence)

### Takeaway
Rank-N-Contrast (NeurIPS 2023), ConR (ICLR 2024), RankSim (ICML 2022, pointer only), TabR (ICLR 2024) and ModernNCA (2024) are the verifiable candidates. None was found to be evaluated on kNN regression over frozen ViT features for small image sets; the closest image-retrieval regressor is a 2025 age-estimation paper. Key numbers and licences remain UNVERIFIED.

### Cited Findings
- Rank-N-Contrast (Zha, Cao, Son, Yang, Katabi): NeurIPS 2023 Spotlight; contrasts samples by their rank in target space so the embedding follows target order; official repo, `RnCLoss` in `loss.py`, AgeDB checkpoints released — [GitHub](https://github.com/kaiwenzha/Rank-N-Contrast), [NeurIPS poster](https://neurips.cc/virtual/2023/poster/71416), [arXiv 2210.01189](https://arxiv.org/abs/2210.01189)
- ConR (Keramati, Meng, Evans): ICLR 2024 (not CVPR); contrastive regularizer modelling global and local label similarity so minority-label features do not collapse into majority neighbours; built on the Yang et al. 2021 DIR and Balanced MSE code bases; official repo BorealisAI/ConR — [GitHub](https://github.com/borealisai/conr), [arXiv 2309.06651](https://arxiv.org/abs/2309.06651)
- RankSim (Gong, Mori, Tung): ICML 2022 (classic, pointer only); makes the ranking of neighbours in label space match the ranking in feature space; closed form, two hyperparameters; evaluated on IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR — [GitHub](https://github.com/BorealisAI/ranksim-imbalanced-regression), [arXiv 2205.15236](https://arxiv.org/abs/2205.15236)
- TabR (Gorishniy et al.): ICLR 2024; an MLP with a kNN-like retrieval module acting as generalised attention over the m nearest candidates; reported best average among tabular DL models and ahead of GBDT on the GBDT-friendly benchmark; repo yandex-research/tabular-dl-tabr; LICENSE text looks like MIT but the name was not shown — [ICLR 2024 page](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html), [GitHub](https://github.com/yandex-research/tabular-dl-tabr), [arXiv 2307.14338](https://arxiv.org/abs/2307.14338)
- ModernNCA ("Revisiting Nearest Neighbor for Tabular Data", 2024): revives Neighbourhood Components Analysis (differentiable kNN) with deep representations; reportedly matches CatBoost across 300 datasets; venue not confirmed in the snippet (OpenReview forum JytL2MrlLT) — [arXiv 2407.03257](https://arxiv.org/html/2407.03257v2), [OpenReview](https://openreview.net/forum?id=JytL2MrlLT)
- IM-Context (arXiv 2405.18202, 2024): in-context imbalanced regression; retrieves neighbours from the original train set and from an inverse-density set; evaluated on AgeDB-DIR, IMDB-WIKI-DIR, STS-B-DIR and six tabular sets; venue not confirmed — [arXiv](https://arxiv.org/pdf/2405.18202)
- Relative Age Estimation (Sandhaus & Keller, arXiv 2502.04852, 2025): initial regressor, retrieve same-age look-alike references by CNN embedding, predict age differences vs references, weighted sum; SOTA on MORPH II and CACD claimed; venue not confirmed — [arXiv](https://arxiv.org/html/2502.04852)
- A 2026 survey "Deconstructing deep imbalanced regression" exists (AI Review) — [Springer](https://link.springer.com/article/10.1007/s10462-026-11570-1)
- The search did NOT surface a "RAR" retrieval-augmented regression paper; the only "RAR" hit (arXiv 2403.13805) is MLLM retrieval-and-ranking for classification — [arXiv](https://arxiv.org/html/2403.13805v1)

### Inferences
- The retrieve-then-predict-difference design (Sandhaus & Keller) is the closest published analogue to "kNN on near twins + size correction": use neighbours' labels plus a learned/ridge delta from the feature difference.
- RnC and ConR are label-aware embedding losses; ModernNCA and TabR are the neighbour-aware heads. All target large or tabular sets; their benefit at 2k samples on top of DINOv2/v3 is untested in the sources found.
- Since ~10% of labels are a 200-1000x mode, rank-based losses (RnC) will be dominated by outliers unless the target is log-transformed and outliers are handled separately; this is reasoning, not a sourced claim.

### Gaps
- Headline numbers (RnC, ConR, TabR, ModernNCA), repo licences and last-commit dates: not verified (github/arxiv fetch blocked).
- Label-distance-aware contrastive variants beyond RnC, and "kNN-augmented DIR" specifically: no matching paper found.
- Venue of ModernNCA, IM-Context and the 2025 age paper: unconfirmed.

## Q2. Neighbour aggregation (weighting, kernel/NW, attention, k, distance-gated fallback; k=1-2 evidence)

### Takeaway
Sourced evidence is thin: TabR frames retrieval as learned attention over m neighbours; KNNOR-Reg uses inverse-distance weighting; the age paper uses a similarity-retrieved set with weighted difference predictions. No source found that directly shows k=1-2 is optimal with near-duplicates.

### Cited Findings
- TabR's retrieval module acts as a generalised attention mechanism over the m nearest neighbours — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html)
- KNNOR-Reg sets synthetic labels as inverse-distance-weighted neighbour averages — [ResearchGate listing](https://www.researchgate.net/publication/381465241_Knnor-Reg_A_Python_Package_for_Oversampling_in_Imbalanced_Regression)
- ImageNet kNN is a few points below linear probe on DINOv2 ViT-g/14 (83.5 vs 86.5) — [DINOv2 paper](https://arxiv.org/html/2304.07193v2)

### Inferences
- With a third of items having a near twin, k=1-2 is plausible because the twin's label dominates; a softmax-over-cosine (temperature) weighting with a gate "if top-1 sim > tau use neighbour, else fall back to ridge" subsumes k selection. Tune tau and temperature by CV; this is a recommendation, not a sourced result.

### Gaps
- No paper found on near-duplicate-aware k selection or distance-gated fallback for regression; no kernel-regression-on-ViT-features evidence.

## Q3. Preventing leakage when the retrieval embedding is fine-tuned on the same train set

### Takeaway
I found no primary source describing cross-fitting or leave-one-out for retrieval heads in a way I could verify. The only partial lead is TabR's own training scheme (candidate set excluding the query), which I could not open to confirm.

### Cited Findings
- Train-test near-duplicate leakage inflates retrieval results and can change system rankings (69% of Robust04 queries had near-duplicates in MS MARCO/ORCAS) — [arXiv 2206.14759](https://arxiv.org/html/2206.14759v2)

### Inferences (not sourced)
- Standard safe practice: nested out-of-fold. For each fold, fine-tune on the other folds, embed held-out objects, and retrieve only from the fold's train bank; report the CV on the held-out. Never retrieve from items the encoder was trained on when scoring those same items. For final test, use the all-train encoder and all-train bank (test twins are not in the training set, so this matches OOF conditions).
- Group folds by near-twin cluster (cosine > threshold) so a twin pair is not split across train/val when the goal is to measure generalisation; but keep the natural twin rate if the test has ~1/3 twins.
- Leave-one-out for a retrieval head: drop self from the candidate set (TabR does this, UNVERIFIED).

### Gaps
- No verified paper on cross-fitting embeddings for kNN regression; TabR's exact leakage handling not confirmed.

## Q4. Retrieval vs fine-tuning at ~2k images; near-duplicate structure in OmniObject3D

### Takeaway
DINOv2 frozen features are within ~2 points of fine-tuned on ImageNet linear eval, supporting frozen+retrieval for small data. OmniObject3D has ~30 objects per category and shape-similar categories; no formal near-duplicate audit found.

### Cited Findings
- DINOv2 linear eval within ~2% of fine-tuned counterparts on ImageNet-1k — [Meta blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/); fine-tuning still reported to add ~2 points in one summary — [Lightly](https://www.lightly.ai/blog/dinov2)
- OmniObject3D: 6,000 scanned objects, 190 categories (~30 per class); clustering of categories shows fruit/vegetable-like shape groups with high inner similarity — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.07525), [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_OmniObject3D_Large-Vocabulary_3D_Object_Dataset_for_Realistic_Perception_Reconstruction_and_CVPR_2023_paper.pdf)

### Gaps
- No kNN/ridge-vs-fine-tune regression benchmark on frozen DINOv2 found; no OmniObject3D duplicate/template audit found.

## Q5. Recommended experiments (6 GB GPU / CPU, cached features), priority order

These are my recommendations, based on the above plus reasoning; expected gains are guesses, not sourced.

1. Soft kNN with gating on cached features (CPU, minutes). Softmax(cos/T) weights over top-m within category, in log-volume space, fallback to ridge when top-1 sim < tau; tune T, tau, m by CV. Expected +0.3 to 1 point; risk low (overfit tau, use nested CV).
2. Retrieve-then-correct (Sandhaus & Keller style): predict log(y_query/y_neighbour) via ridge on feature difference, average over top-m. Expected +0.5 to 1.5; risk moderate, needs scale handling for outlier-mode neighbours.
3. Learned linear/low-rank metric (NCA / ModernNCA-style) on cached CLS concatenation, trained with leave-one-out self-exclusion, log targets, outliers down-weighted. Fits on CPU. Expected +0.5 to 1; risk of overfitting with 2k points; cross-fit with OOF.
4. Small TabR-like head over cached features (m neighbours, self excluded in training) on 6 GB GPU. Expected 0 to 1; higher variance.
5. RnC/ConR fine-tuning of the backbone: only after 1-4; needs OOF encoders (5x cost) to avoid leakage; heavy for 6 GB with ViT-L/H+. Expected uncertain; lowest priority.
6. Treat the oversized mode separately (classifier gate + per-mode kNN) so neighbours from the wrong mode do not contaminate; not a published recipe found here.

### Gaps
- No evidence quantifying gains; all must be validated by grouped, nested CV.

# Label noise, unit-error outliers and mode-aware prediction; long-tailed regression for a saturating metric

Caveat on coverage: ~11 searches, no deep fetches (WebFetch to arxiv was blocked). Most items below are search-snippet level. Nothing found is a direct match for "systematic unit-error labels that stay in the test set". Items marked UNVERIFIED were not confirmed against the primary page.

## 1. Literature on noisy-label / outlier regression (predict-the-noise vs clean-the-noise)

### Takeaway
The 2023-2026 noisy-label regression literature is almost entirely about CLEANING (sample selection, label correction), which is the opposite of the goal here. No paper found models "label is wrong by a known large factor and test labels share the noise". The relevant framing is conditional-mixture modelling plus decision theory (sections 2, 5), not these papers.

### Cited Findings
- ConFrag (Kim, Moon, Moon, Woo, Kim), NeurIPS 2024: sample selection for noisy-label regression. Splits label space into fragments, pairs distant fragments, trains experts per pair, uses neighbourhood agreement among expert features to flag noisy labels. CLEANING, not our goal. Repo github.com/cdjkim/ConFrag (licence not checked, UNVERIFIED). — [GitHub](https://github.com/cdjkim/ConFrag); [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/e68c3a624d19154b28951e8690834607-Paper-Conference.pdf); [arXiv 2502.17771](https://arxiv.org/abs/2502.17771)
- Its reference list cites regression noise baselines C-Mixup, SuperLoss, ITLM (all robust/selection style). — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/e68c3a624d19154b28951e8690834607-Paper-Conference.pdf)
- Interpretable sample selection framework against numerical label noise (Machine Learning, Springer, 2024): argues numerical labels have unbounded candidate values so cannot be corrected precisely, hence filtering. CLEANING. — [Springer](https://link.springer.com/article/10.1007/s10994-024-06637-5)
- Taxonomy of label-noise methods (loss adjustment, transition matrices, label correction, robust losses, sample selection), mostly classification; several not shown to transfer to regression. — [Advances-in-Label-Noise-Learning list](https://github.com/weijiaheng/Advances-in-Label-Noise-Learning)
- A 2026-dated benchmark (Pith summary, UNVERIFIED primary) reports "small-loss failure" and no universal winner for noise-robust training on frozen vision foundation models; classification, but relevant: small-loss selection (Co-teaching family) would drop the oversized mode if it is systematic. — [Pith 2605.22591](https://pith.science/paper/2605.22591)
- "Beyond the Mean: Distribution-Aware Loss Functions for Bimodal Regression" (arXiv 2603.22328, 2026; venue UNVERIFIED): directly about bimodal targets; closest-in-spirit to mode-aware regression. Not read. — [arXiv](https://arxiv.org/html/2603.22328v1)
- Mixture density networks: Bishop's "most probable branch" gives a mode-like prediction for multi-valued mappings; mean-regression averages branches into an implausible middle. — [MDN overview](https://hunterheidenreich.com/notes/machine-learning/generative-models/mixture-density-networks/); [Makansi et al. CVPR 2019, MDN limitations](https://openaccess.thecvf.com/content_CVPR_2019/papers/Makansi_Overcoming_Limitations_of_Mixture_Density_Networks_A_Sampling_and_Fitting_CVPR_2019_paper.pdf)

### Inferences
- Goal-aligned class = "predict the noisy distribution": mixture/switching likelihood (log-space, two components, mode prior from classifier). Filtering/co-teaching/ConFrag all remove exactly the samples that matter.
- Because mode is a discrete latent with a category-level prior, a two-stage model (P(large | x, category) then within-mode regression in log space) is a hand-rolled mixture-of-experts; your current design is already the right family.

### Gaps
- No paper found on unit/scale-error detection in 3D-asset labels. No numbers/licences verified beyond the above.

## 2. Decision theory for choosing the mode under a saturating loss

### Takeaway
Under a bounded, concave-in-|log ratio| loss, a hard mode choice is optimal; hedging in the middle is strictly worse. The threshold is set by relative expected within-mode errors, ~0.5 if symmetric, and the real lever is calibration of P(large), not the threshold.

### Cited Findings
- Cost-sensitive rule: act if calibrated P > c_FP/(c_FP+c_FN); requires calibrated probabilities. (Formula is textbook; survey says calibration is essential for cost-sensitive decisions.) — [Calibration survey, Mach. Learn. 2023](https://link.springer.com/article/10.1007/s10994-023-06336-7)
- Platt works with little data but is misspecified for non-sigmoid distortion; isotonic is flexible, preserves AUC, needs more data, gives steps/ties; beta calibration outperformed logistic calibration on 41 datasets and is better than isotonic on small sets; Venn-Abers is another option. — [Kull et al. 2017 (AISTATS)](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf); [MetricGate comparison](https://metricgate.com/blogs/beta-calibration-vs-platt-vs-isotonic/) (secondary source)
- Use held-out / out-of-fold predictions for calibration; class weighting or resampling distorts probabilities. — [scikit-learn calibration docs](https://scikit-learn.org/stable/modules/calibration.html)
- Under absolute error the optimal point is the conditional median, under squared error the mean, under 0-1-like loss the mode. — [MDN overview](https://hunterheidenreich.com/notes/machine-learning/generative-models/mixture-density-networks/) (secondary)

### Inferences (own derivation, not cited)
- Let wrong-mode cost W≈1.99, within-mode expected cost e_A, e_B (≈0.2-0.4). Choose large iff p*(W) + (1-p)*e_large < (1-p)*W + p*e_small, i.e. threshold p* = (W - e_large)/(2W - e_large - e_small) ≈ 0.5 with a small shift toward the mode with the smaller within-mode error. Threshold tuning beyond 0.5 should give almost nothing; calibration quality matters more.
- Hedge check: for a factor-200 error, predicting the geometric midpoint costs 2tanh(ln200/4)=~1.74 for either truth, versus ≤1.0 expected for hard choice at p=0.5 (199*min(p,1-p)). Because tanh saturates, hedging never wins. Unit-error factor 200-1000 means midpoint cost 1.74-1.83.
- Per-category thresholds add parameters with few samples; prefer one global threshold on a calibrated, category-shrunk probability.
- Conformal sets are of limited use: they give "uncertain" sets but the metric still forces a point choice. Possible use: abstain-to-hedge is never beneficial (see above), so skip.
- Estimating P(large) with few samples per category: Beta-Binomial empirical Bayes: posterior Beta(a+h, b+n-h), shrink weight k/(k+n), fit (a,b) by method of moments or marginal MLE; do not fit the prior from only high-count categories; with k estimated, categories with n<=3 get pulled strongly to the global/coarse-group rate. — [Beta-Binomial shrinkage notes](https://metricgate.com/docs/hierarchical-beta-binomial-baseball/); [Variance Explained](http://varianceexplained.org/r/empirical_bayes_baseball/); [Wikipedia EB](https://en.wikipedia.org/wiki/Empirical_Bayes_method)
- Suggested stack: logistic on features + kNN rate + category-level EB prior fed as features/logit offset; then out-of-fold beta or Platt calibration (isotonic only if >~500 samples).

### Gaps
- No 2023-2026 paper found that directly studies saturating-loss mode selection. Beta calibration vs isotonic evidence is 2017-era plus blog.

## 3. Deep imbalanced regression (DIR) relevance to 164 categories, median 8 objects

### Takeaway
DIR methods target a continuous, long-tailed TARGET density (age, depth) with trained deep features; the evidence is for fine-tuned networks. Little evidence they help on a strong frozen backbone with cached features, and our imbalance is mostly categorical/mode-based, which they do not model. Low priority.

### Cited Findings
- DIR benchmark (LDS/FDS, Yang et al., ICML 2021). — [DIR project page](https://dir.csail.mit.edu/)
- Balanced MSE (Ren et al., CVPR 2022 Oral): Gaussian NLL plus label-prior balancing term; large tail gains on IMDB-WIKI-DIR; BMC variant needs no known prior; code github.com/jiawei-ren/BalancedMSE. — [CVF paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.pdf); [repo](https://github.com/jiawei-ren/BalancedMSE)
- RankSim (Gong et al., ICML 2022): rank-similarity regularization; no result fetched (UNVERIFIED details).
- VIR (Wang & Wang, NeurIPS 2023): probabilistic smoothing with NIG; on IMDB-WIKI-DIR few-shot VIR 9.51 vs RankSim 9.68 MAE (snippet; metric labelling uncertain), small gains. — [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/612a56f193d031687683445cd0001083-Paper-Conference.pdf)
- ConR (Keramati, Meng, Evans, ICLR 2024): contrastive regulariser stopping minority features collapsing into majority neighbours; reported AgeDB-DIR overall MAE 6.81 / few 9.21 (third-party summary); needs augmentations; code github.com/BorealisAI/ConR. — [arXiv 2309.06651](https://arxiv.org/abs/2309.06651); [repo](https://github.com/BorealisAI/ConR)
- 2024-2026 follow-ups named in a review: UVOTE, HCA, Multi-Classification, IM-Context (retrieval-based in-context); and "Exposing Blind Spots in DIR Evaluation" (arXiv 2609.25152, dense-region bias). Not read in depth. — [Springer review](https://link.springer.com/article/10.1007/s10462-026-11570-1); [arXiv 2609.25152](https://arxiv.org/abs/2609.25152)
- No source found testing DIR methods on frozen foundation-model features.

### Inferences
- Gains in these papers are ~0.2-1 MAE points on single-target density; our categories are the tail (48/164 with <=3 objects) and the dominant error is the discrete mode, not a smooth label-density tail.
- If anything is tried: Balanced MSE-style label-prior reweighting on a small log-volume head, or category-level shrinkage, is cheap on cached features. FDS/ConR require end-to-end training and are poor fits.

### Gaps
- No frozen-backbone DIR evidence located.

## 4. Few-shot / rare-category handling

### Takeaway
No directly citable 2023-2026 source found in this pass for hierarchy pooling or text-embedding priors in regression. Recommended approach rests on the Beta-Binomial/hierarchical shrinkage evidence above plus own reasoning.

### Cited Findings
- Hierarchical shrinkage: small groups are pulled hardest toward the grand mean; subgroup priors (e.g. per coarse category) are the standard extension. — [Beta-Binomial notes](https://metricgate.com/docs/hierarchical-beta-binomial-baseball/); [Wikipedia EB](https://en.wikipedia.org/wiki/Empirical_Bayes_method)
- IM-Context (retrieval-based in-context learning for DIR, 2024) is a reported retrieval-style approach; UNVERIFIED details. — [Springer review](https://link.springer.com/article/10.1007/s10462-026-11570-1)

### Inferences
- Pool size and P(large) over a coarse hierarchy (e.g. food group) with partial pooling; for categories with <=3 objects use text-embedding (CLIP/SigLIP text of category name) kNN over categories as the prior, with kNN over image features as likelihood.
- Fit the shrinkage strength k by out-of-fold likelihood, grouped by category (leave-category-out), since test may contain unseen categories.

### Gaps
- No papers verified on text-embedding priors for rare-class regression.

## 5. Validation for a noisy leaderboard (SE ~2.4 points)

### Takeaway
Use repeated, category-grouped stratified CV and a paired bootstrap on per-sample sMAPE differences; accept changes only when the paired CI excludes zero, since a 2.4-point public SE cannot resolve small gains.

### Cited Findings
- Fold partition randomness adds variance; repeating CV with different partitions reduces it. — [Bootstrapping out-of-sample predictions, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6191021/)
- Bootstrap CIs for the CV estimate exist; stability/large-sample assumptions may not hold for small n. — [arXiv 2307.00260](https://arxiv.org/html/2307.00260v2)
- Paired bootstrap protocol ("When +1% Is Not Enough", 2025): BCa CI on deltas + sign-flip permutation; reports unpaired tests often falsely signal 0.6-2.0 point gains; deliberately conservative. — [arXiv 2511.19794](https://arxiv.org/html/2511.19794v1)
- Sources disagree on repeated-CV vs bootstrap bias/variance. — [Applied Predictive Modeling](http://appliedpredictivemodeling.com/blog/2014/11/27/08ks7leh0zof45zpf5vqe56d1sahb0); [Hastie slides](https://hastie.su.domains/MOOC-Slides/cv_boot.pdf)

### Inferences
- Metric is per-sample bounded (0-2), mode errors are ~2.0 vs ~0.3, so the score is dominated by the count of wrong-mode samples: report per-sample paired differences and the number of flipped mode decisions; bootstrap over CATEGORIES (cluster bootstrap), not objects, because objects within a category are correlated.
- Use leave-category-out grouping for rare categories, and plain object-level CV when test categories overlap train; match the test split structure.
- Public LB changes under ~2 SE (~5 points) should be treated as noise unless supported by the CV paired CI.

### Gaps
- Public LB test size/structure unknown, so power is not computable.

## 6. Prioritised recommendation (cached features, CPU)

1. Out-of-fold probability calibration (beta or Platt) of the blended P(large); stack logistic + kNN rate + EB category prior in a single logit model. Expected: modest-to-moderate (mode AUC 0.95 means most gain is in borderline categories); risk low.
2. Beta-Binomial EB shrinkage of category oversize rate with coarse-hierarchy prior; leave-category-out evaluation. Expected: small-moderate on the 48 rare categories; low risk.
3. Keep hard decision, threshold ~0.5 (optionally shifted by within-mode error asymmetry); do not hedge (derivation above). Expected: ~0 beyond calibration; risk none.
4. Within-mode log-volume regression with category shrinkage (ridge/kNN with log-space target, median loss). Expected: small; low risk.
5. Paired cluster-bootstrap + repeated grouped CV as acceptance gate for all above. Cheap, prevents chasing noise.
6. Balanced-MSE-style prior correction or MDN/bimodal head (arXiv 2603.22328): speculative, low expected gain on frozen features, moderate effort. DIR (FDS, ConR, RankSim, VIR): skip.
7. Do not apply co-teaching, ConFrag, or other cleaning methods (they would remove the target mode).

# Distributional / binned regression heads, metric-optimal decoding and ensembling of distributions

Method note: WebFetch to arxiv.org was blocked in this session, so every cited fact below comes from search-result summaries of the primary or mirror pages, not from reading the full paper. Venue/year were taken from the listed pages (ACM DL, Springer, JMLR, ML Anthology, MDPI). Numeric details not visible in snippets are marked UNVERIFIED or listed as gaps. No new (2023-2026) paper was found that tests these heads on a small, multimodal, contaminated-label regression set like ours.

## Q1. Evidence for HL-Gauss / histogram / two-hot / ordinal / quantile / MDN / evidential heads vs plain L1 (2023-2026)

### Takeaway
Evidence that classification-style (histogram) heads beat L2/L1 is real but comes from RL, large tabular sets and foundation-model heads, not from small contaminated-label sets. The best-documented mechanism is optimisation (stable bounded CE gradients, label-noise tolerance), not "modelling the distribution", and the prediction is usually still the bin-weighted mean.

### Cited Findings
- HL-Gauss (Gaussian target integrated over bins, cross-entropy) beat MSE in Atari online/offline, multi-task scaling and robotic manipulation (67% higher peak), reported more robust to noisy targets. ICML 2024, Farebrother et al.; PyTorch package `hl-gauss-pytorch` exists (licence UNVERIFIED) — [arXiv 2403.03950 / ICML 2024 via ACM DL](https://dl.acm.org/doi/10.5555/3692070.3692593), [search summary page](https://arxiv.org/html/2403.03950v1), [PyPI hl-gauss-pytorch](https://pypi.org/project/hl-gauss-pytorch/0.2.8/)
- "Investigating the Histogram Loss in Regression" (Elelimy, Imani, White et al.; arXiv 2402.13425, now JMLR vol. 27): histogram loss converts each label to a bin-probability vector, prediction is the bin-centre-weighted mean; authors conclude benefit is better optimisation, not extra distributional information; on label-noise (CT Scan) L2 degraded most and HL-Gauss least; datasets are fairly large (CT Position, Song Year, Bike Sharing...); rule of thumb ~100 bins, sigma = 2 x bin width; limitation: fixed binning and a known target range — [JMLR 27/24-0260](http://jmlr.org/papers/v27/24-0260.html), [arXiv 2402.13425](https://arxiv.org/abs/2402.13425)
- "Decoding-based Regression" (Song & Bahri, Google DeepMind; TMLR 2025): both distributional heads (histogram "Riemann head" and token-decoding) often beat the pointwise head, especially with little data; decoder head more data-efficient than Riemann head on nearly all tasks; the Riemann head needs more data as the number of bins grows; decoding can beat Gaussian-mixture and Riemann heads — [ML Anthology TMLR 2025](https://mlanthology.org/tmlr/2025/song2025tmlr-decodingbased/), [arXiv 2501.19383](https://arxiv.org/abs/2501.19383)
- TabPFN regression (Hollmann et al., Nature 2025) outputs a bar distribution (histogram) and the official API returns mean, median, mode, quantiles, or full logits for custom Bayes-optimal decoding; best on <=10k rows — [Nature](https://www.nature.com/articles/s41586-024-08328-6), [Prior Labs regression docs](https://docs.priorlabs.ai/capabilities/regression), [bimodal example issue #120](https://github.com/PriorLabs/TabPFN/issues/120). Licence of TabPFN weights UNVERIFIED (check; v2.5+ reportedly non-commercial).
- CORN (Shi, Cao, Raschka; Pattern Analysis and Applications 26, 2023): conditional-probability ordinal head, drops CORAL's weight-sharing, improves over CORAL on MORPH/AFAD (ResNet34) and one tabular set; code `coral-pytorch` — [Springer](https://link.springer.com/article/10.1007/s10044-023-01181-9), [repo](https://github.com/Raschka-research-group/coral-pytorch). Licence UNVERIFIED. Evidence is on age-like ordinal labels, not wide-range log-volume.
- "Beyond the Mean: Distribution-Aware Loss Functions for Bimodal Regression" (arXiv 2603.22328, 2026, not peer-reviewed as far as seen): MDNs are theoretically the gold standard for multimodal regression but prone to optimisation instability; Cramér loss behaves closer to MSE in practice — [arXiv](https://arxiv.org/html/2603.22328v1). Exact numbers UNVERIFIED (fetch blocked).
- Mixture-of-experts distributional regression with robust adaptive first-order optimisation (Rügamer et al., AStA 2024; R package mixdistreg): "robust" refers to stable optimisation, not outlier labels — [Springer](https://link.springer.com/article/10.1007/s10182-023-00486-8)
- Large 2024-25 tabular benchmarks (e.g. Ye et al., 300+ datasets) show no HL-Gauss/histogram-head comparison; I found no paper directly comparing classification-style vs L1 heads on small label-noisy regression — [Ye et al. 2407.00956](https://arxiv.org/html/2407.00956v4)
- Evidential regression: no 2023-2026 primary source retrieved in this session; UNVERIFIED / not covered.
- Quantile-head evidence beyond CQR (below): not retrieved; UNVERIFIED.

### Inferences
- For our problem the histogram head's likely gain is optimisation robustness to the 200-1000x oversized labels (bounded CE gradient vs smooth-L1 on r) and an easy decoding interface, not better within-mode accuracy. Gain over a well-tuned L1 on cached features is probably small (single-digit % relative) — this is my judgment, not a measured number.
- The two-component structure (mode x size correction) can be put directly in a histogram over ln(volume) if bins span both modes (~ln 200 to ln 1000 = 5.3-6.9 nats apart), so mode uncertainty appears as a bimodal histogram and no separate BCE head is needed; but this costs bins and data per category. Alternative: keep the mode classifier and build the mixture explicitly (see Q4).
- Evidence for MDN is mostly "works but unstable"; a 2-component mixture with fixed or category-fixed centres is safer than a free K-component MDN on 2k objects.

### Gaps
- No 2023-2026 small-data (n~2k) comparison with label contamination of HL-Gauss vs L1 vs MDN vs quantile vs evidential; no numeric table found. Licences for CORN/HL-Gauss code not confirmed.

## Q2. Decoding for sMAPE and bounded/saturating losses; calibrated distribution vs hard mode switch

### Takeaway
Under any loss the Bayes point forecast is the minimiser of expected loss under the predictive distribution; sMAPE is not elicitable for a standard functional (not mean/median/quantile), so one must numerically minimise expected sMAPE over a grid. Because 2*tanh(|d|/2) saturates, this "expected-loss argmin" automatically becomes mode-seeking (windowed-mode) when the distribution is multimodal and well separated, and collapses to a hedge/median-like point only when mass is close. I found no paper that tests soft expected-loss decoding against a hard p>0.5 mode switch; that comparison is an inference.

### Cited Findings
- Gneiting, "Making and Evaluating Point Forecasts": the optimal point forecast is the Bayes rule for the specified scoring function (mean for squared error, median for absolute error, weighted-median / ratio-of-expectations forms for percentage/relative errors); for APE a Bayes rule may not exist if the predictive distribution has an infinite fractional moment of order -1 — [arXiv 0912.0902](https://arxiv.org/pdf/0912.0902)
- Kolassa, "The Symmetric Mean Absolute Percentage Error: Unnecessary or Dangerous" (Forecasting 8(2):24, 2026): given a predictive distribution, the sMAPE-optimal forecast is found by computing expected sMAPE as a function of the forecast and minimising numerically; sMAPE does not elicit any standard functional; for count data integers are optimal, for intermittent data zero — [MDPI](https://www.mdpi.com/2571-9394/8/2/24), [Lancaster record](https://research.lancaster-university.uk/en/publications/the-symmetric-mean-absolute-percentage-error-unnecessary-or-dange/)
- Kolassa 2020, IJF "Why the 'best' point forecast depends on the error or accuracy measure" (gamma-simulation demonstrations) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301359)
- Posterior mean is Bayes for squared loss, mode for 0-1 loss (vanishing window); for multimodal distributions the right point depends on the loss (secondary source, MetricGate blog; weak) — [MetricGate](https://metricgate.com/blogs/posterior-mean-vs-posterior-mode/)
- TabPFN's API exposes mode, mean, median and "Bayes-optimal point under custom loss" from a bar distribution; this is the closest practical precedent for decoding a histogram head by expected custom loss — [Prior Labs docs](https://docs.priorlabs.ai/capabilities/regression)
- "Generative Distribution Prediction" (PMC) learns the full conditional and predicts by loss-adapted risk minimisation incl. modal regression — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13529898/) (year/venue UNVERIFIED)

### Inferences
- Concrete decoder: with bin centres c_j, probabilities p_j (histogram, or mixture discretised on a fine grid), choose y* = argmin_y sum_j p_j * 2*tanh(|y-c_j|/2). It is 1-D, costs O(grid^2) per object, trivial on CPU. With two modes at gap g ~ 5-7 nats: loss to wrong mode ~ 2*tanh(g/2) ~ 1.99, so y* sits on the heavier mode unless p is close to 0.5, then it still picks a mode, because the interior point pays ~1.9 to both. So expected-loss argmin ~ hard switch near p=0.5 for well-separated modes; the difference is in the threshold, not the shape.
- The real benefit of soft decoding is (a) the threshold is set by the expected loss using the within-mode spread (a very diffuse minor mode contributes less "saved loss" than a tight one, so effective switching threshold is not exactly 0.5), (b) inside a mode it picks the sMAPE-optimal point (a windowed mode, slightly below the median shifted towards the densest region) instead of the L1/median. Expected gain is probably modest (maybe 0.3-1.5 sMAPE points) — UNVERIFIED estimate; test on OOF.
- Prediction: the optimal threshold on p(oversized) can be tuned directly by OOF sMAPE; if the classifier is miscalibrated, a calibrated p (temperature/Platt on OOF logits) matters more than the decoder.
- Per-category cluster constants imply the within-mode predictive distribution is multi-peaked too (clusters); treat them as mixture components with weights from the kNN/cluster posterior, then decode with the same argmin.

### Gaps
- No paper found directly comparing expected-loss decoding vs hard mode choice; no formal result for sMAPE-optimal forecast under bimodal laws (Kolassa 2026 covers count/intermittent cases; I could not read the full text for continuous multimodal analysis).

## Q3. Calibration and ensembling of distributions

### Takeaway
Pool distributions rather than point predictions, pick the pool by sharpness needs: linear pool (mixture) for multimodal/mode uncertainty (keeps both modes, widens), quantile averaging (Vincentization) only for unimodal members. Calibrate with few parameters (temperature on the histogram logits or mode logit) on OOF predictions, and use conformal only for interval coverage, not for sMAPE point decoding.

### Cited Findings
- Schulz & Lerch, "Aggregating distribution forecasts from deep ensembles" (Machine Learning, Springer 2025; arXiv 2204.02291): both linear pool and Vincentization improve over members across 12 datasets and 7 ensembling strategies; LP is overdispersive; Vincentization preserves family/sharpness; Lakshminarayanan et al. use LP, Rasp & Lerch average parameters — [Springer](https://link.springer.com/article/10.1007/s10994-025-06946-3), [arXiv](https://arxiv.org/abs/2204.02291)
- Gneiting & Ranjan, "Combining Predictive Distributions" (EJS 2013): linear pooling tends to increase dispersion, helpful when components are underdispersed; linear pool not flexibly dispersive; beta-transformed linear pool is, fit by maximising log score — [EJS PDF](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/Combining-predictive-distributions/10.1214/13-EJS823.pdf), [arXiv 1106.1638](https://arxiv.org/abs/1106.1638)
- Rahaman & Thiery (NeurIPS 2021): linear-pool ensembles are less confident than members; benefit comes from correcting overconfidence; they recommend pooling then temperature scaling (classification) — [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2021/file/a70dc40477bc2adceef4d2c90f47eb82-Paper.pdf)
- Temperature scaling is the one-parameter default for neural nets; isotonic recalibration risks overfitting on small calibration sets (Guo et al. 2017; regression discussion in MDPI Sensors 2022 proposes scalar scaling as good as complex recalibration) — [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf), [MDPI 2022](https://www.mdpi.com/1424-8220/22/15/5540)
- Conformal: CQR (Romano, Patterson, Candès, NeurIPS 2019) wraps any quantile regressor, finite-sample marginal coverage under exchangeability — [arXiv 1905.03222](https://arxiv.org/abs/1905.03222). Conformal Histogram Regression (Sesia & Romano, NeurIPS 2021) builds intervals from a histogram predictive distribution (nested intervals; non-convex sets possible) — [NeurIPS PDF](https://proceedings.neurips.cc/paper/2021/file/31b3b31a1c2f8a370206f111127c0dbd-Paper.pdf). Distributional conformal prediction (PNAS 2021) handles non-unimodal outcomes — [PNAS](https://www.pnas.org/doi/10.1073/pnas.2107794118). A density-score (HDR) conformal set is a union of intervals — [arXiv 2511.18141](https://arxiv.org/pdf/2511.18141) (applied to Dirichlet regression; not a direct multimodal-regression source).
- Cheap ensembles: SWAG needs one training run; 5-member SGD ensemble NLL is competitive with one SWAG at ~5x more compute; ~30 samples needed for stable accuracy; single-mode posterior; MultiSWAG recovers deep-ensemble gap — [Maddox et al. 1902.02476](https://arxiv.org/abs/1902.02476), [repo swa_gaussian](https://github.com/wjmaddox/swa_gaussian). Repulsive Last-Layer Ensembles (Steger et al., arXiv 2412.15758): shared frozen features with multiple heads, close to deep ensembles at lower cost — [Pith](https://pith.science/paper/2412.15758). Variational Bayesian last layers (Harrison et al. 2024; venue UNVERIFIED).
- TabPFN ensembling accumulates logits across estimators then decodes — [TabPFN regressor.py](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py)

### Inferences
- On cached features, a deep ensemble of heads (different seeds/folds/bin offsets) costs seconds each on CPU, so the 6 GB GPU constraint is moot; SWAG on the LoRA backbone is the only memory-sensitive option and adds little over head ensembles (last-layer diversity is limited by shared features).
- Averaging point predictions in ln-space destroys the multimodality that sMAPE decoding needs; averaging probabilities (linear pool over bins/mixtures) keeps it. Use LP of member histograms, then temperature-scale the pooled logits (log-pool) or the pooled distribution on OOF.
- Stacking OOF predictions: fit only a few parameters (pool weights, temperature, switch threshold) on out-of-fold predictions grouped by object/category to avoid leakage; more flexible stackers risk overfit at n=2,042 (judgment).
- Conformal does not improve sMAPE; it is useful only to check calibration of p(mode)/intervals. Skip unless a coverage claim is needed.

### Gaps
- No evidence on temperature scaling specifically for HL-Gauss histogram heads in regression found; no quantitative comparison of LP vs Vincentization under bimodal members (Schulz & Lerch numbers not read). Kuleshov et al. 2018 recalibration paper not retrieved.

## Q4. Mixed-mode / contaminated-label regression with a discrete latent

### Takeaway
No retrieved 2023-2026 paper addresses a unit/scale-mode latent. The closest are mixture-of-experts distributional regression and robust MoE with heavy-tailed experts; the design itself follows from the problem and the current best pipeline already is such a switching model.

### Cited Findings
- Mixture of experts distributional regression (Rügamer et al., AStA 2024) — [Springer](https://link.springer.com/article/10.1007/s10182-023-00486-8)
- Robust expert mixtures with normal mean-variance mixture errors (Computational Statistics 2026): classical MoE is outlier-sensitive; identifiable up to label switching — [Springer](https://link.springer.com/article/10.1007/s00180-026-01771-1)
- Robust Classification via Regression (Englesson & Azizpour, ICLR 2024): classification label-noise method, not mode switching — [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/1cb5b3d64bdf3c6642c8d9a8fbecd019-Paper-Conference.pdf)
- Noisy MoE semi-supervised regression (arXiv 2410.09039) — [arXiv](https://arxiv.org/pdf/2410.09039); details UNVERIFIED.

### Inferences
- Model p(ln V | x, cat) = (1-pi) * P_base(ln V - ln c_cat) + pi * P_over(ln V - ln c_cat - delta), with pi = classifier output (calibrated, category-prior shrunk, e.g. Beta-Binomial), P_base/P_over histograms or Student-t/Laplace on r. Learn delta empirically per mode (oversized factor 200-1000x = ln 5.3-6.9, probably spread) rather than a single shift. Train r-heads separately per mode (the base head must not see oversized labels at all, which removes the contamination for the within-mode regressor).
- Because labels in the over-mode are official ground truth, a robust loss on the base head is not needed once modes are separated; keep modes explicit instead of letting a smooth-L1 absorb them.

### Gaps
- No paper on unit-ambiguity (scale-convention) latent regression found.

## Q5. Priority-ordered recommendations (cached features, CPU / 6 GB GPU)

1. Expected-sMAPE decoder on the existing outputs (hours, ~0 risk). Build a mixture from existing parts: mode prob p (OOF-calibrated by temperature/Platt), per-mode within-category predictive spread (OOF residual quantiles of r, category-shrunk), cluster constants as components; decode y* = argmin expected 2tanh(|d|/2) on a grid. Compare to hard p>0.5 on identical OOF folds; also tune a threshold on p. Expected gain: 0.3-1.5 sMAPE points (UNVERIFIED guess); risk low, main risk is miscalibrated p.
2. Calibrate p(oversized) properly: OOF logit -> temperature or Platt, with a category-level Beta prior (shrink category frequency) and kNN neighbour votes as features; evaluate by Brier/log loss and by decoded sMAPE. Expected gain: the largest single lever, because each wrong-mode costs ~199 points per object (at 10% oversized, one extra 1% misclassified objects = ~2 sMAPE points). Risk low.
3. Histogram (HL-Gauss, ~100 bins, sigma = 2 x bin width, per JMLR paper rule of thumb) head on r for the within-mode regressor over cached features, trained per mode; ensemble 5-10 seeds/folds by linear pool, then decode with item 1. Expected gain: small to moderate over smooth-L1 (evidence is on larger sets; label-noise robustness claimed); risk moderate (bin count vs 2k samples, per Song & Bahri more bins need more data; use fewer bins, e.g. 32-64, on a restricted r range).
4. Pool, not average: if several heads/kNN/cluster predictors exist, convert each to a distribution (histogram on a shared grid) and pool by linear pool with weights and temperature fit on OOF (group-by-object folds). Keep Vincentization only for unimodal within-mode pooling. Expected gain: 0.5-1.5 points, risk low-moderate (stacker overfit; keep <=5 parameters).
5. Optional: TabPFN-regression on cached features plus category/kNN features as an extra pool member (outputs a bar distribution natively, supports custom Bayes decoding). Worth a try if the licence allows competition use (UNVERIFIED) and n<=10k fits; risk: licence and CPU runtime.
6. Skip or defer: free-K MDN (instability reported), evidential heads (no evidence retrieved), CORN/CORAL (ordinal evidence on age-like labels; bins on ln-volume can be done with HL-Gauss at lower risk), SWAG on the LoRA backbone (adds little beyond head ensembles), conformal (no sMAPE benefit).

### Gaps
- All expected-gain numbers above are my estimates, not measured; the only way to settle them is OOF experiments with grouped folds.

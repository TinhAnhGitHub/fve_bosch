# Learning formulation for a heavy-tailed, bimodal, partly noisy scalar target scored by sMAPE (single-image food volume)

Verification tags used throughout:
- **[F]**: read this session from the primary source (arXiv page, arXiv HTML full text, official repo, or full PDF text).
- **[C]**: computed or derived this session (Python check script; derivation shown inline).
- **[K]**: from prior knowledge of the paper. The canonical URL is given, but the page was **not** re-fetched this session. Check venue, numbers and defaults before quoting them as fact.

Competition facts used below come from the task brief and are not independently verified: 2,042 train objects, 164 categories, log-volume std 1.89 overall and ~0.52 within category, and ~10% of labels in a second mode ~1000x larger (mesh unit errors). Those labels remain ground truth.

---

## Q1. Regression as classification, ordinal binning and distributional regression: when do they beat L1/L2 on log targets, especially for multimodal targets?

### Takeaway
Use a **categorical (binned) head over log-volume, trained with HL-Gauss-style Gaussian-smoothed cross-entropy**. It can represent the two modes, which L1/L2 and unimodal heads (Gaussian, Laplace, evidential/Student-t) cannot. It is also robust to noisy targets and works as a drop-in replacement. Do not decode it with the expected value (the HL-Gauss/AdaBins default). Decode with the sMAPE Bayes rule (Q4). Evidence that classification beats regression on accuracy alone is mixed for small, clean supervised problems. The decisive argument here is multimodality plus the decision-theoretic decoding, not raw accuracy.

### Cited Findings
- **HL-Gauss / "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"** (Farebrother, Orbay, Vuong, ..., Kumar, Agarwal; arXiv 2403.03950, 2024; published at ICML 2024 [K]). The fetched page did not state a venue. **[F]**
  - **Key idea:** the scalar target is treated as the mean of a Gaussian N(y, σ²). The Gaussian is integrated over each bin with CDF differences, which gives the categorical target p_i = Φ((b_{i+1}−y)/σ) − Φ((b_i−y)/σ), normalised over the bins. The loss is cross-entropy, and the prediction is the expectation over bin centres. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
  - **Hyperparameters:** bin width ς = (v_max − v_min)/m. The paper recommends tuning the ratio σ/ς rather than σ, uses σ/ς = 0.75 by default (mass spread over ~6 neighbouring bins), and ablates m ∈ {21, 51, 101, 201}. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
  - **Evidence:** ~30% IQM gain with SoftMoE on Atari, 1.8–2.1x in multi-task Atari, ~40% on Wordle, ~70% smaller gap to Stockfish for chess without search, and 67% higher peak in robotic manipulation. HL-Gauss beat Two-Hot and C51. The authors conclude that the cross-entropy loss matters more than modelling the return distribution. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
  - **Mechanisms:** HL-Gauss degrades more gracefully than MSE as target noise grows, and its advantage disappears with deterministic dynamics. Cross-entropy-trained features probe better, and the networks keep plasticity under non-stationarity. Label smoothing explains only part of the gain. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
  - **Caveat for supervised regression:** the paper runs no standard supervised regression benchmark. It cites Imani & White (2018), whose small-scale supervised regression study found **no representational benefit** from the histogram loss. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
  - **Code:** reference JAX and PyTorch implementations are in Appendix A of the paper. No repository URL was found. **[F]** [arXiv HTML](https://arxiv.org/html/2403.03950)
- **Imani & White, "Improving Regression Performance with Distributional Losses"**, ICML 2018. This is the origin of the histogram loss with Gaussian-smoothed targets (HL-Gauss). **[K]** [arXiv 1806.04613](https://arxiv.org/abs/1806.04613)
- **"Regression as Classification: Influence of Task Formulation on Neural Network Features"** (Stewart, Bach, Berthet, Vert; arXiv 2211.05641, v2 March 2023; AISTATS 2023, PMLR 206 [K]). The fetched page listed no venue. **[F]**
  - **Key idea:** two-layer ReLU networks are characterised by measures over feature space. For 1-D inputs, the optimal feature-measure support under square loss "can differ greatly" from the support under classification. The implicit bias of gradient methods "could partly explain" why cross-entropy often beats square loss, and the square loss may face optimisation difficulties. No code link was found. **[F]** [arXiv 2211.05641](https://arxiv.org/abs/2211.05641)
- **Pintea, Lin, Dijkstra, van Gemert, "A step towards understanding why classification helps regression"**, ICCV 2023. The paper reports that adding a classification loss to a regression loss helps mainly under **imbalanced** training data. **[K]** [arXiv 2308.10603](https://arxiv.org/abs/2308.10603)
- **Zhang et al., "Improving Deep Regression with Ordinal Entropy"**, ICLR 2023. The paper argues that classification yields higher-entropy feature spaces than regression and adds an ordinal-entropy regulariser to regression. **[K]** [arXiv 2301.08915](https://arxiv.org/abs/2301.08915); code [github.com/needylove/OrdinalEntropy](https://github.com/needylove/OrdinalEntropy) **[K]**
- **DORN** (Fu, Gong, Wang, Batmanghelich, Tao), CVPR 2018. Monocular depth is cast as **ordinal regression** over a **spacing-increasing discretisation (SID)**: bins are uniform in log-depth. An ordinal loss (K−1 binary "y > t_k" tasks) gives large gains over L2 depth regression. Uniform-in-log bins are exactly the structure that fits log-volume. **[K]** [arXiv 1806.02446](https://arxiv.org/abs/1806.02446); code [github.com/hufu6371/DORN](https://github.com/hufu6371/DORN) **[K]**
- **AdaBins** (Bhat, Alhashim, Wonka), CVPR 2021. Bin centres are predicted adaptively per image, and the output is the linear combination Σ_i p_i c_i (a soft expectation). It is trained with a scale-invariant log loss plus a Chamfer loss on bin centres. **[K]** [arXiv 2011.14141](https://arxiv.org/abs/2011.14141); code [github.com/shariqfarooq123/AdaBins](https://github.com/shariqfarooq123/AdaBins) **[K]**
- **BinsFormer** (Li et al.). This extends adaptive bins with Transformer bin queries and a hybrid classification–regression output. Venue not verified (IEEE TIP version [K]). [arXiv 2204.00987](https://arxiv.org/abs/2204.00987); code [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox) **[K]**
- **CORAL** (Cao, Mirjalili, Raschka), Pattern Recognition Letters 2020. K−1 binary "y > r_k" logits share weights with different biases, which guarantees rank-consistent cumulative probabilities. **[K]** [arXiv 1901.07884](https://arxiv.org/abs/1901.07884); code [coral-cnn](https://github.com/Raschka-research-group/coral-cnn) **[K]**
- **CORN** (Shi, Cao, Raschka), Pattern Analysis and Applications 2023. Conditional binary tasks P(y > r_k | y > r_{k−1}) are chained by the product rule. This gives rank consistency without CORAL's weight-sharing constraint. **[K]** [arXiv 2111.08851](https://arxiv.org/abs/2111.08851); code in the `coral-pytorch` package **[K]**
- **Label distribution learning.** Geng, "Label Distribution Learning", IEEE TKDE 2016 **[K]** ([arXiv 1408.6027](https://arxiv.org/abs/1408.6027)). DLDL (Gao et al., IEEE TIP 2017) targets a Gaussian label distribution with KL loss for age and head-pose estimation **[K]** ([arXiv 1611.01731](https://arxiv.org/abs/1611.01731)). HL-Gauss is essentially DLDL with CDF-integrated bins.
- **Mixture density networks** (Bishop, 1994, Aston technical report). These output a K-component Gaussian mixture p(z|x) = Σ π_k N(μ_k, σ_k²), trained by NLL. They natively represent multimodal targets but are known to be numerically fragile (mode collapse, σ → 0). **[K]** [Aston eprint 373](https://publications.aston.ac.uk/id/eprint/373/)
- **Deep Evidential Regression** (Amini, Schwarting, Soleimany, Rus), NeurIPS 2020. A Normal-Inverse-Gamma prior gives a **unimodal Student-t** predictive. It cannot represent a two-mode target. Later work questions its uncertainty semantics (Meinert, Gawlikowski, Lavin, "The Unreasonable Effectiveness of Deep Evidential Regression", AAAI 2023). **[K]** [arXiv 1910.02600](https://arxiv.org/abs/1910.02600); [arXiv 2205.10060](https://arxiv.org/abs/2205.10060)

### Inferences
- **Why classification should win here.** The within-category target in log space is approximately 0.9·N(μ, 0.52²) + 0.1·N(μ + ln 1000 ≈ μ + 6.91, 0.52²). These are two modes ~13σ apart. L2 in log space regresses to the conditional mean, which is 0.69 nats above the base mode: a ~66% sMAPE penalty on the 90% component (2·tanh(0.345) ≈ 0.665 **[C]**). L1 in log space regresses to the median, which barely moves: the median of the mixture is the 0.556 quantile of the base component, +0.14σ ≈ +0.07 nats (**[C]**, from the normal quantile). Neither can ever output the large mode. A categorical head over log bins can put mass on both modes. The sMAPE decision rule (Q4) then chooses which mode to bet on.
- **Recommended head.** Use ~100–150 bins uniform in ln(volume) over about [ln 0.2, ln 60000] (≈ 12.5 nats, so ~0.1 nat bins, finer than the 0.52 within-category spread). Use HL-Gauss targets with σ/ς ≈ 0.75 (the paper's default) and cross-entropy. This is DORN's SID idea (log-uniform bins) combined with HL-Gauss's smoothing. Adaptive bins (AdaBins/BinsFormer) are designed for per-pixel dense maps and add little for a single scalar with 2k samples.
- **Ordinal (CORAL/CORN) versus categorical.** Ordinal heads model cumulative probabilities P(z > t_k) and enforce a monotone CDF. This remains valid for bimodal distributions, because the CDF just has two steep regions. Both formulations yield a full distribution usable for Bayes decoding. HL-Gauss is simpler and has the stronger recent evidence. CORN is a reasonable ablation.
- **Explicit two-component alternative** (structured MDN). Predict a base log-volume distribution plus π(x) = P(unit-error mode | x), with the large-mode location tied to the base mode plus ln 1000. This uses the known physics of the error, needs no free mixture components, and avoids MDN instability. See Q3.
- Classification also helps with the extreme dynamic range: it removes the scale sensitivity of L2 on an 11-nat range.

### Gaps
- No peer-reviewed study was found that compares binned/distributional heads with L1/L2 specifically on **multimodal** targets scored by a **bounded relative-error metric**. The case above is a derivation, not an empirical citation.
- The Stewart et al. AISTATS 2023 venue and the ICML 2024 venue of HL-Gauss come from prior knowledge. The fetched arXiv pages did not state them.
- No official repository for HL-Gauss was located; only the paper-appendix code.

---

## Q2. Imbalanced / long-tailed regression: which methods fit small data with few samples per category?

### Takeaway
The DIR family (LDS/FDS, Balanced MSE, RankSim, ConR, HCA) targets imbalance in the **label value** on large benchmarks (tens of thousands to hundreds of thousands of images). Here the imbalance is mainly across **categories** (48 categories with ≤3 objects) and the label space is reasonably covered in log scale. The cheap, low-risk options are LDS-style inverse-density reweighting in log-volume (mild, e.g. sqrt-inverse) and possibly RankSim as a regulariser. FDS, Balanced MSE (BMC) and ConR need large batches or feature statistics that are noisy at n ≈ 2k. Category-level partial pooling (Q5) is likely more valuable than DIR machinery.

### Cited Findings
- **Deep Imbalanced Regression (LDS/FDS)** (Yang, Zha, Chen, Wang, Katabi), ICML 2021 Long Oral. DIR is defined as learning from imbalanced continuous targets, including target values with little or no data. The method smooths both the label and the feature distributions, on the premise that nearby targets should influence each other. **[F]** [arXiv 2102.09554](https://arxiv.org/abs/2102.09554)
  - **LDS in the code:** bin labels, count per bin, convolve the counts with a kernel window (`get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)`, `scipy.ndimage.convolve1d`), and weight the loss by `1/eff_num`. **[F]** [github.com/YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression)
  - **LDS formula:** the effective density is p̃(y′) = ∫ k(y, y′) p(y) dy, with loss weights ∝ 1/p̃(y). A sqrt-inverse option exists in the code. **[K]**
  - **FDS:** calibrates per-bin feature means and covariances, smoothed across neighbouring bins, before the final linear regressor. It is updated once per epoch from running statistics (`start_update`, `start_smooth`). **[F]** [repo](https://github.com/YyzHarry/imbalanced-regression)
  - **Benchmarks:** IMDB-WIKI-DIR, AgeDB-DIR, NYUD2-DIR, STS-B-DIR, SHHS-DIR. **[F]** [repo](https://github.com/YyzHarry/imbalanced-regression)
- **Balanced MSE** (Ren, Zhang, Yu, Liu), CVPR 2022. Standard MSE is shown to be suboptimal under an imbalanced training label distribution. Balanced MSE corrects for that distribution, including in a variant that needs no prior knowledge of it. **[F]** [arXiv 2203.16427](https://arxiv.org/abs/2203.16427); code [github.com/jiawei-ren/BalancedMSE](https://github.com/jiawei-ren/BalancedMSE) **[F]**
  - **Mechanism and variants:** the likelihood is corrected as p_bal(y|x) ∝ p_train(y|x)/p_train(y). The variants are GAI (GMM label prior), BMC (batch Monte-Carlo: a cross-entropy over the in-batch labels with a learnable noise σ) and BNI (kernel density prior). **[K]**
- **RankSim** (Gong, Mori, Tung), ICML 2022. The regulariser encodes "samples that are closer in label space should also be closer in feature space" by matching each sample's neighbour ranking in feature space to its ranking in label space. It captures both near and distant relations, and is complementary to re-weighting, two-stage training and distribution smoothing. Evaluated on IMDB-WIKI-DIR, AgeDB-DIR and STS-B-DIR. **[F]** [arXiv 2205.15236](https://arxiv.org/abs/2205.15236); code [github.com/BorealisAI/ranksim-imbalanced-regression](https://github.com/BorealisAI/ranksim-imbalanced-regression) **[K]** (the fetched page linked no repo)
- **ConR** (Keramati, Meng, Evans). A contrastive regulariser models global and local label similarity in feature space. It penalises wrong proximities in proportion to label similarity, so minority features do not collapse into majority neighbours. It is "orthogonal to existing approaches" and boosts state-of-the-art methods on four large DIR benchmarks. **[F]** [arXiv 2309.06651](https://arxiv.org/abs/2309.06651); code [github.com/BorealisAI/ConR](https://github.com/BorealisAI/ConR) **[F]**
  - The fetched page listed no venue; it was published at ICLR 2024 **[K]**.
- **HCA, "Deep Imbalanced Regression via Hierarchical Classification Adjustment"** (Xiong & Yao), CVPR 2024. A fine classifier over quantised targets is adjusted by an ensemble of progressively coarser classifiers. Range-preserving distillation merges them into one classifier. The authors frame quantisation as a trade-off between classification accuracy (coarse, balanced bins) and quantisation error (fine bins). Evaluated on age, crowd counting and depth. Code: [github.com/xhp-hust-2018-2011/HCA](https://github.com/xhp-hust-2018-2011/HCA). — [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Deep_Imbalanced_Regression_via_Hierarchical_Classification_Adjustment_CVPR_2024_paper.html) (search-result summary, **[F]**-partial)
  - A third-party summary gave IMDB-WIKI-DIR bMAE numbers that were internally inconsistent, so they are omitted.
- **Variational Imbalanced Regression** (Wang & Wang), NeurIPS 2023. A probabilistic approach that borrows statistics from similar labels in a VAE-style model. **[K]** [arXiv 2306.06599](https://arxiv.org/abs/2306.06599)

### Inferences
- **Which methods fit n ≈ 2k:**
  - **LDS reweighting** works: it is cheap and stable, and it applies directly to the HL-Gauss cross-entropy as per-sample weights. Use sqrt-inverse or a clipped inverse to avoid exploding weights on the tail extremes (0.38 cm³, 33k cm³).
  - **RankSim** works: it is batch-local and has no density estimation.
  - **FDS** is risky: per-bin feature statistics from a handful of samples per bin are noisy.
  - **Balanced MSE-BMC** is designed for MSE heads and is redundant once the head is a classifier.
  - **ConR** needs augmentations and large batches.
  - **HCA's coarse-to-fine** classifier idea matches this data well, with coarse bins (e.g. 1 nat) and fine bins (0.1 nat). It is cheap to approximate with an auxiliary coarse-bin cross-entropy head.
- **Is reweighting even the right goal?** sMAPE weights every sample equally, so balancing toward rare *values* trades away accuracy on common values. Unless the test distribution differs from train, reweighting should be mild and tuned on out-of-fold sMAPE.
- The real long tail is **per-category sample count**. Address it with category-aware priors and partial pooling (Q5) and with strong pretrained backbones, not with DIR losses.

### Gaps
- No DIR paper evaluates at the ~2k-sample, 164-category scale or with sMAPE. Applicability is inferred.
- The Balanced MSE variant details and the RankSim and ConR venue and repo for RankSim come from prior knowledge. The fetched abstracts did not include them.

---

## Q3. Regression under label noise and outliers, when the "noise" is the scoring ground truth

### Takeaway
Do not filter the ~10% "1000x" labels or downweight them away: the test set presumably contains the same phenomenon, and missing it costs ~198% per sample. Instead, **model it as a latent discrete mode**: z = ln V = z_base + 6.91·u, with u ~ Bernoulli(π(x)). Learn π(x) explicitly through a binary head, or implicitly through a two-mode categorical head. Robust losses and co-teaching are appropriate only for the **base-mode** regressor, never for the final scored output.

### Cited Findings
- **sMAPE as a bounded loss in log space.** For positive values with ratio r = ŷ/y, sMAPE(ŷ, y) = 2|r−1|/(r+1) = 2·tanh(|ln ŷ − ln y|/2). This is checked numerically for r ∈ {1e-3 … 1e3}. So sMAPE is a symmetric, bounded (max 2), redescending loss in log space. Its influence function ψ(e) = sech²(e/2) → 0 for large log errors. It behaves like L1 for small log errors and saturates like a 0–1 loss for large ones. **[C]** (derivation; identity checked in `smape_check.py`)
- **Barron, "A General and Adaptive Robust Loss Function"**, CVPR 2019. A single family nests L2, pseudo-Huber/Charbonnier, Cauchy, Geman-McClure and Welsch. The robustness parameter can be learned. **[K]** [arXiv 1701.03077](https://arxiv.org/abs/1701.03077); code [github.com/jonbarron/robust_loss_pytorch](https://github.com/jonbarron/robust_loss_pytorch) **[K]**
- **Co-teaching** (Han et al.), NeurIPS 2018. Two networks exchange small-loss samples to filter noisy labels in classification. **[K]** [arXiv 1804.06872](https://arxiv.org/abs/1804.06872)
- **Kendall & Gal**, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017. Heteroscedastic aleatoric regression predicts σ(x) and uses the loss |e|/σ + log σ (Laplace) or e²/2σ² + ½ log σ² (Gaussian). This lets the model attenuate the loss on samples it cannot fit. **[K]** [arXiv 1703.04977](https://arxiv.org/abs/1703.04977)
- HL-Gauss degrades more gracefully than MSE as target noise grows. **[F]** [arXiv HTML 2403.03950](https://arxiv.org/html/2403.03950)

### Inferences
- **Why L1/L2 and robust losses alone fail.** Robust losses treat the large mode as an outlier and converge to the base mode. That is in fact the right decision when π(x) < 0.5 (Q4), but it **forecloses** predicting the large mode when the image or category signals it. Expected loss in the "always base mode" regime is ≈ π·1.996 on top of the base-mode error. With π = 0.1 this adds ~20 points of sMAPE. **[C]** (closed form: 2·tanh(6.91/2) = 1.996)
- **Recommended structure** (a two-head "noise-aware" model). Training the base head only on samples whose label is consistent with the base mode is the "noise filtering" step, applied only to the base head. Both heads remain scored jointly through the decision rule.
  1. Base head: an HL-Gauss categorical over ln V (or L1 in ln V), trained with the 1000x samples shifted down by ln 1000. Only do this if they can be identified reliably, for example by comparing against the category median: they sit ~6.9 nats away, ≈13 within-category σ. Otherwise train it on all samples, letting the categorical head absorb both modes.
  2. Mode head: a binary classifier for P(u = 1 | x), trained on the identified flags.
  3. Decoding: compute the expected sMAPE of each candidate (Q4). With far-separated modes this reduces to choosing the large mode iff P(u = 1 | x) > ~0.5 (**[C]**: the argmin switches between p_big = 0.45 and 0.55).
- Whether π(x) is predictable from the image determines whether this structure yields any gain. Possible predictors are object category, OmniObject3D source/batch artefacts, and rendering cues. With 1000x in volume, the cue is a 10x linear-unit error (mm vs cm), and rendered images may be scale-normalised. If π(x) is not predictable, the optimal policy is "always base mode", and the 1000x samples are an irreducible ~20% floor (for π ≈ 0.1). Measure P(u = 1 | category) on train. If some categories have π_c > 0.5, the decision flips for them.
- A single categorical head trained with plain cross-entropy on raw labels already learns p(mode | x) implicitly. The decoding rule is what matters. The explicit binary head is mainly useful because u is a clean, low-noise binary target with better sample efficiency.
- Co-teaching or small-loss selection on raw labels would systematically discard the large mode. Use it only for the base head, or not at all.

### Gaps
- No peer-reviewed work was found on "structured label noise that is part of the evaluation ground truth" (unit-error modes) in regression. The modelling recommendation is a derivation.
- No co-teaching-for-regression paper was verified this session.
- Whether OmniObject3D unit errors cluster by category or source is not verified here. It requires inspecting the training labels.

---

## Q4. Loss functions and decision theory for relative-error metrics: the optimal point estimate under sMAPE / MAPE / log error, and how to decode a binned distribution

### Takeaway
Train a distribution, then **decode with the Bayes rule of the metric**:
- **sMAPE** depends only on |Δ ln| through 2·tanh(|Δ ln|/2). For a unimodal, symmetric log-distribution its Bayes rule is the **median** (= exp μ for a lognormal). There is no exp(σ²/2) correction.
- For **multimodal** distributions it is a **mode-seeking weighted median**, best found by grid search over bin centres.
- Expected-value decoding (the HL-Gauss/AdaBins default, or E[ln V]) is badly wrong for bimodal predictions: 80% versus 20% expected sMAPE in the worked example.
- **MAPE** has a different optimum, the β = −1 median, which equals exp(μ − σ²) for a lognormal.

### Cited Findings
- **Gneiting, "Making and Evaluating Point Forecasts"**, JASA 106 (2011); arXiv 0912.0902 (v2 2010). **[F]** [arXiv 0912.0902](https://arxiv.org/abs/0912.0902) (full PDF text extracted)
  - **Bayes rule:** "the forecaster can issue an optimal point forecast, namely, the Bayes rule, which minimizes the expected loss under the forecaster's predictive distribution." **[F]**
  - **Mismatched scoring misleads:** averaging a score that is not matched to the forecasting task "can lead to grossly misguided inferences". In the paper's simulation, a deliberately biased "pessimist" beats the true-model "statistician" under absolute error and APE, and an "optimist" wins under relative error. **[F]**
  - **β-median:** the family S_β(x, y) = |1 − (y/x)^β| (β ≠ 0) nests APE |(x−y)/y| (β = −1) and relative error |(x−y)/x| (β = 1). If F has density f on (0, ∞) with a finite fractional moment of order β, the Bayes rule is the median of the density ∝ y^β f(y), the **β-median** med^(β)(F). The ordinary median is the limit β → 0. **[F]**
  - **Log-scale scores:** for a log-scale accuracy score, the Bayes rule is a functional applied to the predictive distribution **of the logarithm**, not of the quantity itself. **[F]**
  - The text does **not** treat sMAPE. **[F]**
- **sMAPE asymmetry.** Goodwin & Lawton (1999, IJF) showed that sMAPE is asymmetric for forecasts at equal absolute distance (110 vs 90 against actual 100). Hyndman later corrected the direction stated in Hyndman & Koehler (2006): the heavier penalty falls on forecasts that are too *small*. Several sMAPE definitions coincide when all data and forecasts are non-negative. — [Hyndman blog, "Errors on percentage errors" (via r-bloggers)](https://www.r-bloggers.com/2014/04/errors-on-percentage-errors/)
- **Hyndman & Koehler, "Another look at measures of forecast accuracy"**, IJF 22 (2006) 679–688. Many measures used in the M-competitions are degenerate in common situations, and the authors propose MASE as the standard. — [RePEc record](https://econpapers.repec.org/paper/mshebswps/2005-13.htm)
- **M4 competition** (Makridakis, Spiliotis, Assimakopoulos, IJF 2020). Evaluation combined sMAPE and MASE into the Overall Weighted Average (OWA). **[K]** [doi:10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014)
- **Kolassa, "Why the 'best' point forecast depends on the error or accuracy measure"**, IJF 36 (2020). For skewed predictive distributions, the optimal point forecast differs by metric (MSE → mean, MAE → median, MAPE → a lower value). **[K]** [doi:10.1016/j.ijforecast.2019.02.017](https://doi.org/10.1016/j.ijforecast.2019.02.017)

**Derivations (all [C]):**
- **Ratio form:** with r = ŷ/y > 0, 2|ŷ−y|/(ŷ+y) = 2(r−1)/(r+1) for r ≥ 1. This equals 2·tanh((ln r)/2), and by symmetry 2·tanh(|ln r|/2) for all r. In log space sMAPE is therefore **symmetric**. The Goodwin–Lawton asymmetry concerns equal *absolute* deviations, not equal *ratios*.
- **First-order condition:** for candidate x and a discrete predictive distribution {(y_k, p_k)}, d/dx [2|x−y|/(x+y)] = +4y/(x+y)² if x > y and −4y/(x+y)² if x < y. The optimum therefore satisfies
  Σ_{y_k < x} p_k · y_k/(x+y_k)² = Σ_{y_k > x} p_k · y_k/(x+y_k)².
  This is a **weighted median with weights w_k(x) = y_k/(x+y_k)²**. In log space, with t = ln x and z_k = ln y_k, the weight is ½·sech²((t − z_k)/2). Bins far from x receive vanishing weight, which makes the rule mode-seeking.
- **Unimodal case:** if ln Y is symmetric about μ, the expected loss E[2·tanh(|t − ln Y|/2)] is minimised at t = μ, so x* = median = e^μ. A simulated lognormal(0, 0.52²) gave a sMAPE-optimal x = 1.000 (median 1).
- **MAPE case, for contrast:** for a lognormal(μ, σ²), y^β f(y) ∝ lognormal(μ + βσ², σ²). So:
  - the MAPE optimum (β = −1) is exp(μ − σ²);
  - the relative-error optimum (β = +1) is exp(μ + σ²);
  - the sMAPE optimum is exp(μ).
  With σ = 0.52, MAPE-style decoding would shrink predictions by e^{−0.27} ≈ 0.76, which is the wrong correction for sMAPE.
- **Bimodal worked example:** masses at 10 and 10,000 (1000x), with p_big the probability of the large mode.

  | p_big | sMAPE-optimal ŷ | Expected sMAPE at optimum | exp(E[ln y]) decode | Linear-mean decode |
  |---|---|---|---|---|
  | 0.10 | 10 | 20.0% | 79.7% | 192.8% |
  | 0.30 | 10 | 59.9% | 167.7% | 171.3% |
  | 0.45 | 10 | 89.8% | 186.7% | 143.6% |
  | 0.55 | 10,000 | 89.8% | 186.7% | 121.6% |

  The optimum jumps between the modes at p_big ≈ 0.5. Hedging between modes is never optimal, because sMAPE saturates.
- **Irreducible floor from the within-category spread:** the expected sMAPE of the median predictor for lognormal residuals is 23.6% (σ = 0.3), 39.7% (σ = 0.52), 70.2% (σ = 1.0) and 108% (σ = 1.89). A category-only predictor therefore floors at ~40% on the base mode. The image must explain within-category variance to go lower.

### Inferences
- **Decoding recipe for a categorical head** (bin centres c_k in ln space, probabilities p_k, optionally temperature-calibrated):
  1. Evaluate R(t_j) = Σ_k p_k · 2·tanh(|t_j − c_k|/2) for all candidate t_j = bin centres, refined to a 0.01-nat grid if desired. This is O(K²) and trivial at K ≈ 150.
  2. Output exp(argmin_j R(t_j)).
  - For a two-head model, apply the same rule to the mixture (1−π)·p_base + π·shift(p_base, ln 1000).
- **Fine-tuning option:** train directly on the expected sMAPE, L = Σ_k p_k · 2·tanh(|c_k − ln y|/2), the "expected cost" loss, or on sMAPE of the decoded point. Gradients vanish for large errors (sech²), so it is a poor loss from scratch and could suppress learning of the far mode. Use cross-entropy (HL-Gauss) for representation learning and sMAPE only for decoding, calibration, or a short final fine-tune.
- **For a point-regression baseline**, L1 on ln V is the right surrogate: its Bayes rule is the median of ln V, matching sMAPE's unimodal Bayes rule. L2 on ln V targets the mean and is pulled by the large mode, so it is wrong. L1 on raw V is dominated by the large tail.
- The usual M4-style use of sMAPE involves no special training trick; competitors decode medians or point forecasts. No forecasting-literature source on decoding binned distributions under sMAPE was found, so the recipe above is a derivation.

### Gaps
- No published closed-form Bayes rule for sMAPE was found. The weighted-median fixed-point condition above is derived here and checked numerically, not cited.
- The M4 OWA details and the Kolassa 2020 IJF content come from prior knowledge, not re-fetched.

---

## Q5. Hierarchical / category-conditioned regression and empirical-Bayes shrinkage for categories with few samples

### Takeaway
The category explains most of the log variance (overall std 1.89 versus 0.52 within category), so a **category prior** is the strongest single feature. With a median of 8 samples per category and 48 categories with ≤3, raw category means are noisy and should be **partially pooled**: James–Stein / empirical-Bayes shrinkage of category log-medians toward a parent (super-category or global) mean. The network then predicts a residual, or receives the shrunk prior as an input. Because the test category is predicted from the image, use the category *posterior* to mix priors.

### Cited Findings
- **Efron & Morris, "Data Analysis Using Stein's Estimator and its Generalizations"**, JASA 70 (1975). Shrinking many group means toward a pooled mean dominates the per-group means in total squared error when there are ≥3 groups (the James–Stein effect). The classic baseball batting-average example is in this paper. **[K]** [doi:10.1080/01621459.1975.10479864](https://doi.org/10.1080/01621459.1975.10479864)
- **Micci-Barreca, "A Preprocessing Scheme for High-Cardinality Categorical Attributes"**, SIGKDD Explorations 3(1) (2001). Smoothed target encoding blends the category mean with the prior mean using a weight λ(n) that increases with category count. This is the standard tabular-ML form of empirical-Bayes pooling. **[K]** [doi:10.1145/507533.507538](https://doi.org/10.1145/507533.507538)
- **Gelman et al., *Bayesian Data Analysis*, 3rd ed. (2013), hierarchical normal model.** Partial pooling of group means has the posterior-mean form below. **[K]** [BDA3 website](http://www.stat.columbia.edu/~gelman/book/)

  θ̂_c = (n_c/σ² · ȳ_c + 1/τ² · μ) / (n_c/σ² + 1/τ²)

### Inferences
- **Concrete shrinkage** in ln V, using robust statistics because of the 1000x mode:
  1. Per category, compute the base-mode location m_c as the median of ln V after removing or shifting flagged 1000x samples, with sample size n_c.
  2. Estimate the within-category variance σ_w² ≈ 0.52² (pooled) and the between-category variance τ² by method of moments, (Var of m_c) − mean(σ_w²/n_c).
  3. Shrink: m̃_c = w_c·m_c + (1−w_c)·μ_parent, with w_c = τ²/(τ² + σ_w²/n_c).
  - With τ ≈ 1.8 and σ_w = 0.52, w ≈ 0.92 even at n = 1. Shrinkage on the *location* is mild, because categories are well separated relative to the within-category noise.
  - Shrinkage matters more for per-category **π_c** (the large-mode rate) and **σ_c**. Use a Beta–Binomial prior on π_c, for example Beta(α, β) fit to the global ~10% rate, so a category with 1 of 2 samples in the large mode is not assigned π_c = 0.5.
- **How to use it with the network:**
  1. Use the shrunk prior as an input feature or a residual target: predict ln V − m̃_ĉ, where the category comes from an image classifier (or zero-shot CLIP/VLM labelling at test time).
  2. Or add the prior as a log-prior to the HL-Gauss logits.
  3. When the category prediction is uncertain, mix: p(z|x) = Σ_c P(c|x) · p(z | x, c).
  - The Bayes decoding (Q4) then handles multimodality across both categories and unit-error modes.
- **Cross-validation hygiene:** compute category statistics out-of-fold, as in target encoding, to avoid leakage. With ≤3 samples, in-fold category statistics leak the label almost entirely.
- **Hierarchy depth:** if a natural food super-category exists (fruit, bread, ...), a two-level hierarchy (category → super-category → global) gives better parents for rare categories. This mirrors HCA's coarse-to-fine idea (Q2).

### Gaps
- No peer-reviewed deep-learning paper evaluating empirical-Bayes category shrinkage for single-image volume or mass regression was found in this session.
- τ² has not been estimated from the actual labels. The w ≈ 0.92 figure assumes τ ≈ the overall std, which overstates τ slightly.
- Efron–Morris, Micci-Barreca and BDA3 are cited from prior knowledge, not re-fetched.

---

## Q6. Calibration and post-hoc bias correction for log-space predictions

### Takeaway
For sMAPE, **do not** apply lognormal back-transform bias corrections (Duan smearing, exp(σ²/2)), because sMAPE's unimodal Bayes rule is the median. Two calibration steps do help:
1. Temperature-scale the categorical head on out-of-fold data. Mode probabilities drive the large-mode decision, so their calibration matters.
2. Fit a single global (or per-super-category) log-offset δ that minimises out-of-fold sMAPE, which corrects systematic shrinkage toward the global mean.

### Cited Findings
- **Duan, "Smearing Estimate: A Nonparametric Retransformation Method"**, JASA 78 (1983). To estimate E[Y] after regressing ln Y, multiply exp(prediction) by the mean of exp(residuals). This targets the **mean**, which is the wrong functional for sMAPE. **[K]** [doi:10.1080/01621459.1983.10478017](https://doi.org/10.1080/01621459.1983.10478017)
- **Gneiting (2011):** the Bayes rule must match the scoring function, and for log-type scores it is a functional of the log predictive distribution. **[F]** [arXiv 0912.0902](https://arxiv.org/abs/0912.0902)
- **Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks"**, ICML 2017. Temperature scaling, a single scalar T on the logits fit by validation NLL, is a simple and effective post-hoc calibration for softmax classifiers. **[K]** [arXiv 1706.04599](https://arxiv.org/abs/1706.04599)
- **Kuleshov, Fenner, Ermon, "Accurate Uncertainties for Deep Learning Using Calibrated Regression"**, ICML 2018. Isotonic recalibration of the predicted CDF yields calibrated quantiles for regression. **[K]** [arXiv 1807.00263](https://arxiv.org/abs/1807.00263)
- **Lognormal derivation:** sMAPE-optimal = exp(μ), MAPE-optimal = exp(μ − σ²), relative-error-optimal = exp(μ + σ²). **[C]** (Q4)

### Inferences
- **Recommended post-hoc pipeline** (all fit on out-of-fold predictions, K-fold grouped by object, possibly stratified by category):
  1. **Temperature scaling** of the bin logits, and separately of the mode-head logit if one is used, fit by NLL. Miscalibrated mode probabilities move the p ≈ 0.5 switching point directly (Q4 table).
  2. **Decode** with the expected-sMAPE argmin (Q4).
  3. **Global log-offset δ:** ŷ ← ŷ·e^δ, with δ chosen by 1-D search to minimise out-of-fold sMAPE. Optionally fit per super-category, or as a linear map t ← a + b·t to undo regression-to-the-mean shrinkage on the 11-nat range. Keep b near 1 and regularise it, given 2k samples.
  4. Optionally apply a **decision threshold** on P(large mode): replace 0.5 with a tuned τ*, because the probabilities are imperfect.
  5. Fit only one or two scalars per calibration step; more risks overfitting the 2k-sample out-of-fold set.
- Isotonic CDF recalibration (Kuleshov) is useful if quantiles are needed, but brings little for a median-type decision beyond temperature scaling plus δ.
- **Ensembling:** average the *probability vectors* across folds/models, then decode. Averaging decoded point predictions in ln space hedges between modes, which Q4 shows is harmful. Averaging in linear space is worse.

### Gaps
- No source specific to post-hoc calibration under sMAPE was found. The δ-offset and threshold tuning are standard practice inferred from the decision theory, not cited results.
- Duan 1983, Guo 2017 and Kuleshov 2018 are cited from prior knowledge, not re-fetched this session.

# Mesh as training-time-only supervision for single-image volume (research notes, 2026-11-05)

Method: WebSearch + WebFetch of arXiv abstracts/HTML. Items marked UNVERIFIED were not confirmed from a primary page. No paper found that exactly matches "mesh-derived auxiliary targets for single-image volume regression"; this is a real gap. Evidence below is mostly adjacent.

## Q1. Privileged / auxiliary 3D supervision for single-image physical-property regression

| Paper | Venue/year | Finding | Repo / licence | Small-data? |
|---|---|---|---|---|
| Lian, "Single-Image Food Mass Estimators Do Not Transfer Their Gram Scale Across Datasets", arXiv 2610.29025 | preprint, Oct 2026 | DINOv2-base fine-tuned on Nutrition5k (2,253 dishes); training-only auxiliary depth-derived "height-sum" head (discarded at test; paper calls depth "privileged information"). Ablation (Table IX): val MAE 24.8 g with vs 25.7 g without; seed control 0.6-0.9 g gain in-domain; no consistent out-of-domain gain; frozen backbone 35.2 vs 35.5 g. Proxy correlates only 0.33 with log mass. | github.com/yongshenglian/foodmass-transfer; licence UNVERIFIED | Yes (2k samples, same scale as ours). Best direct evidence: gain is real but small (~3%). |
| Physically Guided Visual Mass Estimation, arXiv 2601.20303 | IJCAI 2026 main | Volume-/density-latent heads under mass-only supervision, monocular depth for geometry, VLM for material. Beats image2mass/ABO-500 baselines (numbers not read). No explicit volume labels, so the privileged-label variant is untested. | code UNVERIFIED | Moderate |
| image2mass (Standley et al.) | CoRL 2017 | Early image-to-mass with a shape-estimation module; scraped product data. (Details from search snippet only.) | UNVERIFIED | - |
| Image2Mass++ (ScienceDirect 2025) | journal | Reports Andrade et al. added Pix2Vox++ 3D reconstruction, "only marginal" gains (secondary report, UNVERIFIED). | - | - |
| MFP3D, arXiv 2411.10492 | 2024 | MetaFood3D; MAE 62.6 ml, MAPE 41.4%. Uses 3D reconstructor at inference, so not privileged. | UNVERIFIED | - |
| Size Matters, arXiv 2601.20051 (Vinod et al., Purdue) | 2026 | Learn a scale factor from foundation-model features to rescale a single-view reconstruction; ~30% lower volume MAE (abstract). Uses MetaFood3D + OmniObject3D(?) (search summary; UNVERIFIED). Needs reconstruction at inference. | gitlab.com/viper-purdue/size-matters; licence UNVERIFIED | Yes; relevant because predicting scale/volume ratio relative to shape mirrors our "fill" head |
| MetaFood3D, arXiv 2409.01966 | 2024 | 743 scanned food meshes, 131 categories, volume in Blender, nutrition. | Data CC BY-NC 4.0, request form | Possible extra training data (food only; check domain) |
| Generic LUPI: Vapnik & Vashist 2009; Lopez-Paz et al. "Unifying distillation and privileged information" (ICLR 2016) | - | Foundations; from memory/search mention, not fetched. | - | - |
| "Distill What RGB Can Recover", arXiv 2608.00110 | 2026 | Privileged 3D evidence teacher -> RGB-only student, down-weights samples RGB can't recover. VLM/3D-QA, not regression. Idea (sample down-weighting) is transferable. | UNVERIFIED | - |

Take-away: the only quantified training-only 3D-aux result near our setting yields ~1 g / 3% in-domain with a noisy proxy. Expect modest gains.

## Q2. Which mesh-derived targets
No paper found that compares mesh-derived target types in a low-data regime (UNVERIFIED / not found). Reasoned options (no citation):
- Cheap, exact, scale-bearing but pose-dependent: sorted bbox dims (already used), fill, surface-area/volume^(2/3) compactness (isoperimetric ratio), convex-hull volume / volume (convexity), PCA eigenvalue ratios of vertices, inertia-tensor moments normalised by volume^(5/3), thickness stats.
- Render-dependent: silhouette area fraction, depth-map statistics. Only valid if you re-render with the same auto-framing; our test render pipeline is auto-framed, so silhouette fraction is "scale-free" and learnable.
- Scale-free shape descriptors (aspect ratios, fill, convexity, compactness) are better aligned with what a single image can show than absolute dims. Absolute dims are ambiguous from the image and mostly category prior.
- Caution: Lian's result shows a weakly correlated proxy gives tiny gains; targets should be tightly tied to volume (fill, compactness, convexity).

## Q3. Re-rendering, shape embeddings, retrieval
- ULIP (CVPR 2023, arXiv 2212.05171) / ULIP-2 (CVPR 2024, arXiv 2305.08275): github.com/salesforce/ULIP; code licence UNVERIFIED; data triplets ODC-By for Objaverse.
- OpenShape (NeurIPS 2023, arXiv 2305.10764): github.com/Colin97/OpenShape_code; licence UNVERIFIED; 46.8% zero-shot Objaverse-LVIS; supports image->3D shape retrieval; default config for one A100 (inference of pretrained encoder fits 6 GB).
- Uni3D (ICLR 2024 spotlight, arXiv 2310.06773): github.com/baaivision/Uni3D; licence UNVERIFIED; 6M-1B params; Tiny/Small usable on 6 GB.
- Point-MAE (ECCV 2022): github.com/Pang-Yatian/Point-MAE; MIT per third-party aggregator (UNVERIFIED); old PyTorch pins.
- FiT3D (ECCV 2024, arXiv 2407.20229): github.com/ywyue/FiT3D; 3D-aware fine-tuning of DINOv2 via multi-view feature lifting; gains shown for segmentation/depth only; UNVERIFIED for volume. Scene-level, not object-level.
- These encoders are scale-normalised (point clouds are normalised to unit sphere in OpenShape/ULIP-style pipelines; from memory, UNVERIFIED), so their embeddings capture shape, not size. Useful as a shape teacher, not size teacher.
- Re-rendering: no paper found specific to volume regression. Practically, our train images are already Blender renders, so synthetic-to-real gap is nil; extra viewpoints from the mesh are the real win (unlimited same-distribution augmentation).

## Q4. Mesh descriptors for unit-error / oversized labels
Searched; no literature found on detecting scale-label errors in 3D asset datasets (Objaverse's own metric-scale metadata is crowdsourced category-level and not per-instance exact, per HF discussion; OmniObject3D has no noise analysis in results). A practitioner blog notes unit mismatches are typically factors of 10/100/1000 (non-peer-reviewed). Recommendation is therefore engineering, not literature: check gt volume vs mesh.volume (trimesh) and bbox_mm vs mesh extents; ratio near 10^3 or 10^k flags unit error. Caveat: if mesh and gt.json share the same wrong units (likely since both come from the same asset), the check finds nothing; better is mesh-vs-category-median (z-score of ln volume within category) combined with image-based prediction disagreement.

## Integration into our pipeline

1. Mesh sanity audit (zero training cost, do first): compute trimesh volume, bbox, hull volume, surface area per object; compare with gt.json. Gives label-noise flags, exact fill, and the descriptors. Also reveals whether gt volume is mesh volume (then exact targets for free).
2. Extra aux heads on existing DINOv3 setup, scale-free targets: ln fill (have), ln convexity (V/V_hull), ln compactness (S^1.5/V), PCA eigenvalue ratios, silhouette-area fraction rendered from the mesh under the same auto-framing. Small MLP heads on frozen/LoRA features; negligible memory. Weight ~0.1-0.3, drop heads that don't help CV. Validate with category-grouped K-fold (hold out whole objects, ideally near-duplicate-aware), report sMAPE mean +/- std over seeds; expect gain within noise (Lian: ~3%).
3. Mesh re-rendering augmentation: render K extra views per mesh in Blender with the same auto-framing/lighting, sampled poses; train with the same volume label. Likely the highest-impact for 2k objects; GPU not needed beyond rendering (Eevee/Cycles low samples). Validate that the added views don't leak across CV folds (group by object).
4. Shape-embedding distillation (OpenShape/Uni3D-Tiny): encode meshes (sampled 10k points) offline once, store embeddings; add cosine/contrastive loss from image features to the frozen shape embedding; also use for kNN over train meshes by shape. Caveat: shape embeddings are scale-normalised so help with fill/aspect only; expect small gain; also check licences before use.
5. Optional: volume-consistency loss, predicted dims product x fill = volume (physically-structured head; similar spirit to 2601.20303 and Size Matters).

## Ranked top 3
1. Mesh audit + scale-free mesh descriptors as aux targets (fill, convexity, compactness, PCA ratios, silhouette fraction) - cheapest, closest to evidence (2610.29025), also yields oversized-label flags.
2. Mesh re-rendering as multi-view augmentation (same pipeline; unlimited data; group-by-object CV).
3. Shape-embedding teacher (Uni3D-T/OpenShape) distillation + shape-based retrieval - higher effort, uncertain gain because embeddings discard scale.

Honest caveat: no published result shows mesh supervision substantially improving single-image volume; the best adjacent ablation is ~3% MAE. Treat expected gains as 0-5% sMAPE, and the main reliable benefit as better label QA.

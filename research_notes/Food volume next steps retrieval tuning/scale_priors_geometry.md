# Scale priors from foundation models and geometry models for auto-framed single-image food volume

Scope note: this pass used about 8 tool calls. WebFetch was blocked for github.com, so licences and VRAM figures come from search snippets and are flagged. I did not skim the earlier report. Many key questions are only partly answered (see Gaps).

## Q1. VLM/LLM size estimation, size priors, and whether text-only category priors can compete with data medians

### Takeaway
I found no paper that benchmarks text-only LLM category size priors against data-derived category medians, and none that extracts absolute size from a scale-free auto-framed image. The nearest evidence says LLMs are weak on size comparison and VLMs only gain size knowledge after being trained on spatial data.

### Cited Findings
- POSQA (EMNLP Findings 2023) is a size-comparison QA dataset of 92 entities with absolute sizes. Even the largest LLMs do poorly zero-shot, and they lean on the prompt context rather than their weights. It has code at cambridgeltl/POSQA. — [POSQA](https://arxiv.org/html/2310.13394)
- SpatialVLM (arXiv 2401.12168) states that VLMs lack quantitative spatial knowledge. After training on synthesized spatial data, it gives reliable metric estimates and gains common-sense knowledge of object sizes. — [SpatialVLM](https://arxiv.org/pdf/2401.12168)
- SD-VLM (arXiv 2509.17664, venue UNVERIFIED) reports a 56.31% success rate on its reference-object estimation task, against 32.28% for GPT-4o. The task needs a reference object, which our renders lack. — [SD-VLM](https://www.alphaxiv.org/abs/2509.17664)
- PhysBench (arXiv 2501.16411) tests 75 VLMs on physical properties, size among them, and finds large gaps in physical understanding. — [PhysBench](https://arxiv.org/html/2501.16411v2)

### Inferences
- Under our setup the category is known from the folder, so a VLM would only be adding a category-level prior. A within-category size signal (e.g. a small versus large walnut) is not something these papers cover.
- A text-only prior is cheap and can be computed offline once per category. I'd expect it to be noisy for 164 categories and to be at best a weak feature next to data-derived medians. This is an inference with no direct evidence, and I did not find any evidence that it beats medians.
- Run a one-off LLM query for typical size/volume of each category. Use the answer as a feature only if it correlates with the train category medians. Where it does not, the data-derived median wins.

### Gaps
- No paper found comparing LLM-elicited size priors with Visual Genome or Objects365 size statistics.
- No CLIP size-knowledge probing paper found. Not searched beyond one query.
- The venues of SD-VLM and PhysBench were not verified.

## Q2. Texture, material and surface detail as an absolute-scale cue in renders

### Takeaway
I found nothing published on scale from texture, depth of field, shadow softness or grain for synthetic renders. This is an unanswered question, not a negative result.

### Cited Findings
- None found. The searches in this pass did not target this question directly, so the absence is not strong evidence either way.

### Inferences
- Auto-framing makes the object fill the frame, so texture granularity relative to the object (for example seed or shell texture density) is the only plausible leak. DINO's within-category signal of residual R^2 about 0.3-0.4, mostly through look-alike neighbours, may already encode part of it. This is speculation.
- A cheap check is to compute high-frequency energy and texture spatial frequency statistics and test them against residual size within category.

### Gaps
- Needs direct searches for "scale from texture", "texture scale estimation", and render-pipeline metadata (DoF, shadow). Not done.

## Q3. Metric-depth and point-map models as feature sources

### Takeaway
Metric depth models can give only focal-length-dependent output, and no evidence was found of anyone using their encoder features or predicted focal length to regress object size under auto-framing. The conversion from depth to metres needs the focal length in pixels, which is unknown here.

### Cited Findings
- Depth Anything 3 metric: metres equal focal (pixels) times net output divided by 300, so it requires intrinsics. DA3METRIC-LARGE is Apache 2.0, while DA3NESTED-GIANT-LARGE is CC BY-NC 4.0 (non-commercial). Checkpoints with the -1.1 suffix are recommended, and older ones are deprecated. — [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE), [DA3NESTED-GIANT-LARGE](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE)
- DA3 VRAM: no official table. Third-party guides say Base needs 4 GB or more and Large 8 GB or more. These are unofficial. — [DA3 repo](https://github.com/bytedance-seed/depth-anything-3)
- MoGe-2 predicts a point map in metric scale by adding a decoupled global scale factor. It uses a DINOv2 ViT-L backbone at about 60 ms per image (A100/RTX 3090, FP16), so it should fit on a 6 GB GPU at reduced resolution. This VRAM fit is an inference. MoGe-3 exists, with ViT-L and ViT-g checkpoints. Licence text was not retrieved. — [MoGe-2 paper](https://arxiv.org/abs/2507.02546), [MoGe repo](https://github.com/microsoft/moge)

### Inferences
- With auto-framing, the model's scale guess comes from appearance, not geometry, so the earlier report's "scale collapse" finding implies metric heads would mostly encode category. Their encoder features or predicted FOV might add a little, but this is untested.
- The cheap, falsifiable test is to extract the MoGe-2 or DA3 encoder embeddings (or predicted scale) and check residual R^2 against the DINO features already in use.

### Gaps
- Licence and VRAM for MoGe-2 (the repo says "See LICENSE"), Depth Pro, UniDepth v2, Metric3D v2 and VGGT were not verified. Mark all UNVERIFIED.
- No paper found using these models' features or predicted focal length as a size regressor input.

## Q4. Single-image-to-3D as a shape-factor source

### Takeaway
Published volume or fill-ratio accuracy for these models was not found. Hardware and licensing are mixed, and only TRELLIS.2 and Hunyuan3D 2.1 are cheap to describe. TRELLIS.2 does not fit a 6 GB GPU.

### Cited Findings
- TRELLIS.2: MIT licence for code and model, with some dependencies (nvdiffrast, nvdiffrec) under separate terms. The official requirement is an NVIDIA GPU with at least 24 GB. Sources conflict on a lower floor of 16 GB at 512 resolution. — [TRELLIS.2 repo](https://github.com/microsoft/trellis.2), [CodeSOTA](https://www.codesota.com/news/trellis-2-3d-generation)
- SAM 3D Objects: a maintainer says the smallest reliable GPU is a V100 (about 32 GB), and a 24 GB card can work if mesh decoding is skipped. Licence not confirmed. — [issue 6](https://github.com/facebookresearch/sam-3d-objects/issues/6)
- Hunyuan3D 2.1: README lists 10 GB for shape, 21 GB for texture and 29 GB for both. A community fork with offloading reports 4-6 GB for shape only, which is unofficial. Licence is the Tencent Hunyuan Community Licence, which per a third-party summary excludes the EU, UK and South Korea and bars using outputs to improve other AI models. The summary is not the licence text. — [Hunyuan3D-2.1](https://github.com/tencent-hunyuan/hunyuan3d-2.1), [respawnstack summary](https://respawnstack.com/tools/hunyuan3d)
- InstantMesh: the Hugging Face card lists Apache 2.0, trained on Objaverse (about 270k filtered objects). OmniObject3D appears only as an evaluation benchmark, with no evidence of training use. OmniObject3D has 6,000 scans across 190 categories under CC BY 4.0. — [InstantMesh paper](https://arxiv.org/html/2404.07191v1), [OmniObject3D](https://omniobject3d.github.io/)

### Inferences
- Any shape-factor from these models is normalised by the auto-framing, so it can give fill ratio or relative shape, not absolute size. Which makes it a candidate for the volume-to-bounding-box ratio only.
- Training-set contamination risk by OmniObject3D: Objaverse-trained models may share objects with it. I found no evidence either way for TRELLIS.2, SAM 3D and Hunyuan3D (UNVERIFIED).
- Runtime on a 6 GB GPU is likely impractical for TRELLIS.2 and SAM 3D. Only InstantMesh or offloaded Hunyuan3D shape-only might fit, and only unofficially.

### Gaps
- No published volume or fill-ratio accuracy found for any of the four models.
- Training data lists for TRELLIS.2, SAM 3D Objects and Hunyuan3D were not retrieved.

## Q5. Recommendation

### Takeaway
Little here is likely to pay off. Cheap tests first, expensive 3D generation skipped.

### Inferences
- Try (cheap, a few hours): (a) LLM-elicited per-category size priors, kept only if they correlate with train medians, which is zero per-image cost; (b) MoGe-2 ViT-L (reduced token count) encoder features or predicted scale as extra regressor inputs, checked by residual R^2 gain over DINO; (c) simple texture-frequency statistics. Expected gain is small, probably under a few points of within-category R^2. This is a guess, not evidence.
- Skip: TRELLIS.2 and SAM 3D Objects (24-32 GB), the Hunyuan3D texture stage, and the DA3 Nested Giant (non-commercial). The reference-object VLM methods need a reference that renders lack.
- Verify licences from the LICENSE files before use, since search-derived licence claims here are secondary.

### Gaps
- No evidence of any 2024-2026 method recovering absolute size from a scale-free auto-framed image.

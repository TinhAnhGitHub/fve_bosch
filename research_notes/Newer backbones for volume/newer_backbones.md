# Newer backbones for single-image volume regression (researched 2026-11-06)
Verification level: facts below come from HF/GitHub/arXiv pages or search summaries fetched today. Anything marked UNVERIFIED was not confirmed. VRAM figures are my estimates, not measured.

## Verified candidates
| Candidate | Repo id | Params / res | Licence | Notes |
|---|---|---|---|---|
| C-RADIOv4-H (Jan 2026, arXiv 2601.17237) | nvidia/C-RADIOv4-H | 653M, patch16, 256-2048 px step 16 | NVIDIA Open Model License; card says commercial/non-commercial OK; no gate noted | Distils SigLIP2-g-384 + DINOv3-7B + SAM3. `summary, spatial = model(px)` via trust_remote_code. Card: ADE20k 55.2, kNN IN 86.68 (vs v3-H 52.75 / 86.23). No DINOv3 comparison on the card. |
| C-RADIOv4-SO400M | nvidia/C-RADIOv4-SO400M | 431M (card) | same | Smaller, same teachers. |
| C-RADIOv3-L/H/g | nvidia/C-RADIOv3-{L,H,g} (timm/vit_*_radio_v3.c_radio remaps) | 320M/653M/1.2B | same | Teachers: DFN CLIP-H, SigLIP2-SO400M, DINOv2-g-reg, SAM-H. Superseded by v4. |
| PE-Spatial (Meta, arXiv 2504.13181) | facebook/PE-Spatial-{T16-512,S16-512,B16-512,L14-448,G14-448} | G ~1.9B (PE-core G is 1.88B; spatial sizes not listed) | Apache-2.0, no gate seen | Core intermediate-layer features self-distilled + SAM2.1 masks. G14: ADE linear 49.3. Loads through repo's `pe.VisionTransformer.from_config`, `forward_features(img, strip_cls_token=True)`, `layer_idx=` option. Repo code is not timm. |
| PE-Core | facebook/PE-Core-{B16-224,L14-336,G14-448} | 0.09B / 0.32B / 1.88B | Apache-2.0 | CLIP-style. Contrastive features carry language/size priors (SigLIP2 already failed for us, so low prior). |
| PE-Lang | facebook/PE-Lang-L14-448 (and G, Tiling) | 448 px | Apache-2.0 unless noted (UNVERIFIED per card) | MLLM-aligned. |
| TIPSv2 (CVPR'26, arXiv 2604.12012) | google/tipsv2-b14 | 86M vision, patch14 | Apache-2.0 | Only B seen. Focus is patch-text alignment. Weak case here. |
| DINOv3 other sizes | facebook/dinov3-vith16plus-pretrain-lvd1689m (840M), -vit7b16-pretrain-lvd1689m | 840M / 6.7B | DINOv3 License (gated, custom); LICENSE.md not read for competition-use terms | ViT-7B: ~14 GB bf16 weights; frozen only on T4 would need 8-bit/offload, UNVERIFIED feasibility. We already use H+. |
| Web-DINO (arXiv 2504.01017, ICCV'25) | facebook/webssl-dino7b-full8b-518 (also 224; smaller sizes exist) | 7B | licence UNVERIFIED | Web-scale SSL without language. 7B too big for T4; smaller webssl sizes unverified. |
| Depth Anything 3 | depth-anything/DA3METRIC-LARGE, DA3-LARGE-1.1 | ~0.35-0.4B | Apache-2.0 (Large, Metric-Large seen; Giant/Base/Small UNVERIFIED) | Metric depth = focal*net/300; needs focal length, which we lack. |
| MoGe-2 (arXiv 2507.02546) | Ruicheng/moge-2-vitl | 326M | MIT (HF page) | Outputs metric point map; scale from single image is a learned guess for scene images. |
| Pi3 / Pi3X | yyfz233/Pi3, yyfz233/Pi3X | UNVERIFIED | Weights non-commercial research/education only | Multi-view geometry. REJECT. |

## Related literature found
- "Size Matters" arXiv 2601.20051: single image -> One-2-3-45 mesh -> frozen CLIP ViT-L/14@336 on the image and 75 renders -> MLP predicts scale factor. OmniObject3D MAE 70.5 mL, MAPE ~86%. Category-mean baseline had MAPE 87.5%, so the learned model barely beat the category prior on MAPE. Supports the idea that the size cue comes from category/semantics. Shape priors from reconstruction were not shown to help MAPE.
- arXiv 2602.13041 (MetaFood 2025 implicit-scale challenge): VLM baselines were competitive on volume but geometry reconstruction was more accurate.
- No paper found that reports a new architecture beating DINOv2/v3 on object size/volume regression. No 2026 general backbone found that clearly beats DINOv3 on frozen features (search summary only; DINOv4/SigLIP3/PE2 not found, so UNVERIFIED as nonexistent).
- Small-data FGVC study arXiv 2605.15599 (frozen ViT-B/16, emerald grading): SigLIP2/supervised encoders gave better linear separability than DINOv3. Niche domain, one dataset.
- Diffusion features: no paper found comparing them to DINO for image-level regression. One result (arXiv 2610.11751) says diffusion features are reconstruction-oriented, not semantic. REJECT.
- Qwen3-VL vision tower: SigLIP2-based (L/16 ~300M for 2B/4B, SO400M for larger), Apache-2.0 per third-party site (not checked on HF). Vision-tower features are SigLIP2 derivatives, which already failed. Using the LLM for a size prior is a different, heavier experiment (UNVERIFIED value; VLMs are reported weak at metric estimates).
- MambaVision / ConvNeXt-V2 / InternImage / RepViT / FastViT: I found no credible small-data transfer evidence in this pass. Our own ConvNeXt (DINOv3) test failed on the board. REJECT, unverified.

## Ranked recommendation
1. C-RADIOv4-H, frozen, nvidia/C-RADIOv4-H. Distils DINOv3-7B, SigLIP2 and SAM3 jointly. SAM3 mask/shape-aware dense features are the part that is new relative to our DINOv2/DINOv3 mix. Features: `summary` (CLS-like), plus mean-pooled and maybe GeM-pooled spatial tokens, at 512 px (multiple of 16). VRAM: ~3-4 GB fp16 inference at 512, fits A3000. Evaluate: screen as a replacement and as an addition in the frozen branch (per-category cluster + ridge + logistic + kNN) and the blend. Adopt only if >=0.5 CV on seeds 0-2 and 3-5. Guess: +0.3 to +1 sMAPE; most likely it is highly correlated with DINOv3-H+ since DINOv3-7B is a teacher.
2. PE-Spatial-L14-448 (or B16-512 for speed), facebook/PE-Spatial-L14-448. Different pretraining lineage (contrastive core + SAM2.1 distillation), Apache-2.0, ungated. Use `forward_features(strip_cls_token=True)`, pool tokens (no CLS in spatial output, so use mean/GeM over tokens, and maybe last 2 layers via `layer_idx`). VRAM: ~2 GB fp16 for L. Evaluate the same way. Guess: +0 to +0.7; diversity is the argument, not strength. Try G14 (~1.9B, ~4-5 GB fp16) only if L shows signal.
3. PE-Core-L14-336 (facebook/PE-Core-L14-336), frozen CLIP-style tower, mostly to test whether language-derived size knowledge differs from SigLIP2's, which failed. Low prior. VRAM ~1.5 GB. Guess: +0 to +0.4. Run only after 1 and 2 because it is cheap.
4. DINOv3 ViT-H+ test-time/feature variants (no new weights): multi-layer concat of the last 4 blocks and GeM pooling from the frozen DINOv3 branch. Not a new architecture, but cheaper and more likely to carry signal than adding families that are correlated. Guess: +0 to +0.5.

## Rejected
- DINOv3 ViT-7B / Web-DINO 7B: too big for T4 fine-tuning; for frozen use, ViT-7B is a DINOv3 sibling already distilled into H+ and C-RADIOv4.
- MoGe-2, DA3, Pi3, VGGT, DUSt3R/MASt3R, Depth Pro, UniDepth V2: need or assume intrinsics/scene context. Auto-framed renders carry no scale, so metric output would be a category-prior guess at best. Pi3 weights are non-commercial. (Relative-shape cues might help fill/aspect, but our bbox/fill aux heads already target that. UNVERIFIED, not tested.)
- SigLIP2 variants, Qwen3-VL/InternVL towers, TIPSv2: same contrastive/language lineage as SigLIP2, which failed. InternVL3 not looked up (UNVERIFIED).
- Diffusion/DiT features: no evidence for image-level regression; heavy.
- ConvNeXt/MambaVision/RepViT/FastViT/InternImage: no evidence found; ConvNeXt already failed on the board.
- C-RADIOv3 / AM-RADIO: superseded by v4.

## Caveats
Gate/licence status read from search summaries for several repos; check each card before downloading. Competition-use terms for DINOv3 and NVIDIA Open Model License were not read in full. No weights were downloaded or run.

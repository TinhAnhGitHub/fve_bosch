# F3DR plan: one general model (from 2026-11-10)

> **Update 2026-11-11:** phases 0–1 are done (results in `f3dr_submissions/F3DR_full_report.md` sections 9.7–9.9). No single-model change beat seed noise on the leaderboard; the best is the RH-Mix5 mix at 43.85. The plan is now **generator shape + a learned size model** (report section 17). Phases 2–4 below are superseded by it.

Goal: beat v22 (46.35 leaderboard; competitor 46.88) with ONE model, no ensembles.
Baseline single model: pl_t_s0_ema (v21, 55.74 leaderboard, local strict 34.3 / 34.5 on real-handed shapes, median 19.0).
v24 (+ positional embedding) scored 55.35: keep `--pos-emb` on.

## Rules
- Train only on Google Colab and Kaggle. Never on the local machine (user, 2026-11-10).
- Judge each change on 2 seeds, mean AND median of local strict (RH=1), against pl_t_s0_ema. A gap under ~2 is noise.
- Pull every result (checkpoint, score log) as soon as it finishes; Colab sessions can vanish.
- Any data beyond the cached tokens (raw images, meshes, model weights) needs the user's explicit OK before it goes to Kaggle/Colab.

## Infrastructure (ready)
- Kaggle private datasets: `anhanhanhanh/f3dr-cache` (train.npz, train_dinov3l.npz, train_mirror.npz), `anhanhanhanh/f3dr-code` (atlas.py, geom.py, score2.py, run_exps.py, pl_t_s0_ema.pt).
- Kaggle kernels `f3dr-k1`, `f3dr-k2` (T4 x2, 3 runs each). Kernel folders in the scratchpad `kernels/`; `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`; results via `kaggle kernels output`.
- Colab: `colab new -s c1 --gpu L4`; caches via `kaggle datasets download anhanhanhanh/f3dr-cache -f <file>` (one file at a time), token passed with `colab exec --env`.
- `baselines/colab/run_exps.py WORK DATA CODE name "flags" ...` runs experiments in parallel over the GPUs, scores each with RH=1, writes out/summary.txt.
- After changing atlas.py: `kaggle datasets version -p <code dir>`, wait for "ready", then push kernels.

## Phase 0: running now (2026-11-10 ~14:15)
| Where | Runs |
|---|---|
| Colab c1 (L4) | n_pos_s2, s_rh_s2, s_posrh_s1 |
| Kaggle f3dr-k1 | s_rh_s0, s_rh_s1, n_pos_s1 |
| Kaggle f3dr-k2 | n_cat, n_rep, s_posrh_s0 |
Decide: positional embedding (3 seeds), right-handed frame (3 seeds), both together, category dropout, repulsion.
Results so far (leaderboard): positional embedding alone 55.35 / 60.12 -> dropped. Right-handed 54.77, pos+right-handed 52.89 (best single model) -> `--right-handed` is now the default in new recipes. Local mean ranked these wrongly; confirm with uploads over 2-3 seeds.

## Phase 1: small fixes on the existing caches (no new uploads)
1. **Size head**: unit-size shape + log-scale offset from the category mean (Gaussian NLL); output E[s] = exp(mu + sigma^2/2); kNN size from DINOv3 class tokens as a feature; global multiplier fitted on holdout; short end-to-end mm fine-tune. Targets: size (44% of error), inward crowding (RMS radius 0.77x).
2. **Sinkhorn/EMD term** next to Chamfer (8192 vs 8192, geomloss); tune the weight on holdout CD-L2. Targets: missing > stray.
3. **Loss variants**: HyperCD, InfoCD, plain L1-CD. Avoid DCD.
4. **Token-side augmentation**: feature noise, FroFA-style brightness/contrast on tokens, SWA on top of EMA, more weight decay.
5. 384 px with sheet attention (`dinov3l384` entry is back; cache needs uploading).

## Phase 2: needs raw images (user OK required for the upload, or build caches locally only with permission)
6. **Image augmentation**: `atlas.py augcache --encoder dinov3l --aug-sets 8`, train with `--aug-sets 8` (code written and checked).
7. **Re-render the training meshes** (16-32 views, lighting/focal jitter, same framing): strongest fix for overfitting (train 8 vs holdout 35).
8. **MoGe-2 geometry tokens** (MIT) as a second token stream.
9. **High-resolution texture crops** for the size head.
10. **DINOv3 ViT-H+** tokens (weights are in the local HF cache).

## Phase 3: architecture
11. **Coarse-to-fine decoder**: 512 query seeds -> Snowflake upsampling to 8192, CD at every level, 2-4M params.
12. **Pixel-aligned point features** in the camera frame (first prove the grader aligns rotation globally).
13. **Category template + residual** (k-medoid templates).

## Phase 4: pretrained image-to-3D
14. **TRELLIS v1 zero-shot** on the 204 holdout images (A100, 1-2 h): shape only, scale to our predicted size, ICP onto our cloud from 24 rotations; score with oracle size and with predicted size. Adopt only if predicted size beats our model.
15. If the oracle-size shapes win clearly: **LoRA on TRELLIS(.2) sparse-structure flow** (Atlansert/Trellis.2-lora, MIT), ~3k steps, 4-8 A100 h.
Licences: TRELLIS MIT; skip RMBG-2.0 and nvdiffrast (non-commercial). Not usable: SF3D/SPAR3D (enterprise licence), Hunyuan3D (EU excluded).

## Final step
Retrain the chosen recipe on all 2,042 objects, build the zip (exactly 8192 points), archive checkpoint + atlas.py + commit in f3dr_submissions/.
Also still open: `predict` for the private test split, DINOv3 licence check, whether category-from-path is allowed.

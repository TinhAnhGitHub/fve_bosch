# SAM 3D shapes for the food photos

Runs Meta's **SAM 3D Objects** through **Fast-SAM3D** (same model, about 2.7x faster) on every photo and saves each
object's 3D shape as 8192 points. On the 92 holdout objects run so far, SAM 3D's shape at the true size had about half
our model's error (about 23 vs 49 mm^2) and beat it on 72 of 92 objects. Its own size is useless on these renders
(every object comes out about 0.5 m), so the size has to come from a separate model trained on these outputs.

## Requirements

- Linux or WSL2 with an NVIDIA GPU and a driver that runs CUDA 12.1 (the prebuilt pytorch3d / kaolin / flash-attn wheels are Linux-only)
- About 20 GB of disk for the environment and weights, and 16 GB+ of normal memory for `--low-vram`
- A Hugging Face account with **approved access** to `facebook/sam-3d-objects`, and a read token
- GPU memory: about **17.6 GB** peak on an L4 in normal mode. On a **12 GB** card use `--low-vram`, which keeps each model
  in normal memory until its step runs. That mode has **not been tested on 12 GB**: each photo prints its peak GPU memory, so
  the first photo tells you whether it fits.

## Steps

```bash
cd fve_f3dr/baselines/sam3d
bash install.sh                                   # about 10-15 min
HF_TOKEN=hf_xxx bash download_weights.sh          # about 7 GB
bash run_all.sh ../../data/f3d --low-vram         # drop --low-vram on a 24 GB card
```

`run_all.sh` does three passes and writes into `fve_f3dr/baselines/out/atlas/cloud/`:

| Pass | Photos | File | Time at 7.5 s/photo (L4, normal mode) |
|---|---|---|---|
| Holdout (`holdout_ids.txt`) | 204 | `sam3d_hold.npz` | about 25 min |
| All training photos | 2,042 | `sam3d_train.npz` | about 4.3 h |
| Public test (eval) | 511 | `sam3d_eval.npz` | about 1 h |

`--low-vram` will be slower than this, because models move between normal and GPU memory for every photo.
Results save every 10 photos and a rerun resumes where it stopped.

## Output (each .npz)

| Key | Shape | Meaning |
|---|---|---|
| `ids` | (n,) | sample ids |
| `pts_mesh` | (n, 8192, 3) | surface points of SAM 3D's mesh, in its own frame and made-up scale |
| `pts_gs` | (n, 8192, 3) | Gaussian centres, object frame |
| `scale`, `rotation`, `translation` | (n, 3), (n, 4), (n, 3) | SAM 3D's own placement; its size is not usable here |

Failed photos are all NaN.

## Files

- `install.sh`, `download_weights.sh`, `run_all.sh`, `gen_sam3d.py`: the pipeline
- `fix_flow_matching_model.py`, `fix_dino.py`: two fixes to the Fast-SAM3D repo as published (an import of a missing solver,
  and a DINOv2 path pointing at its authors' disk); `install.sh` copies them in
- `holdout_ids.txt`: the 204 holdout sample ids (seed-0 10% split used by `atlas.py`)

## Licences

SAM 3D Objects: Meta SAM License (gated; commercial use allowed with restrictions). Fast-SAM3D code: see its repo.
MoGe: MIT. BiRefNet (masks): MIT. The rendering libraries (nvdiffrast etc.) are not installed: `gen_sam3d.py` stubs them,
because shape generation does not need them.

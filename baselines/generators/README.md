# Pretrained image-to-3D generators (shape only)

Scripts used to test pretrained single-image 3D generators as a **shape** source. Each produces 8192 surface points
per photo in the generator's own frame and arbitrary scale; the size has to come from a separate model.

| Folder | Model | Licence | Status (holdout, squared Chamfer in mm^2 with the TRUE size) |
|---|---|---|---|
| `../sam3d/` | SAM 3D Objects via Fast-SAM3D | Meta SAM License (gated) | 92 objects: ~23 vs our model ~49; wins on 72/92. Best so far |
| `triposg/` | TripoSG (VAST-AI) | MIT | 204 objects: 29.6 vs our 38.9 (same objects: SAM 3D 13.8 vs TripoSG 35.2 on 20) |
| `idol/` | IDOL (speridlabs) | Apache-2.0 (FLUX.2 VAE taken from FLUX.2-klein-4B, Apache-2.0) | running on Kaggle, not scored yet |
| `trellis2/` | TRELLIS.2-4B | MIT (skip RMBG-2.0 and nvdiffrast: non-commercial) | never got a shape out (attention backend / install issues) |

Not used: Hunyuan3D 2.x (licence excludes the EU, UK and South Korea), SF3D/SPAR3D (enterprise licence above $1M revenue).

With our model's predicted size every generator scores far worse than our model (TripoSG ~91, SAM 3D ~97),
so the next step is a separate size model: frozen DINOv3 image features + the generator's size-free shape
descriptors -> object size and its uncertainty.

Scoring: `../tools/trellis_score.py` (oracle size / predicted size / bounded scale+ICP, against real-handed holdout shapes).

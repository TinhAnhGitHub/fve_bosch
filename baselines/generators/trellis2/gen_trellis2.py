"""Generate TRELLIS.2 shapes (shape stage only, 512 resolution) for folders of photos, on the Colab A100.

usage (inside the TRELLIS venv): python gen.py OUT.npz DIR [DIR ...]
Saves ids (n,) and pts (n, 8192, 3): surface samples of the generated mesh in TRELLIS' own frame and scale.
Licence notes: the default RMBG-2.0 background remover is non-commercial, so ZhengPeng7/BiRefNet (MIT) is used;
DINOv3 loads from /content/dino (copied from the local Hugging Face cache); no texture stage, no nvdiffrast.
"""
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "xformers"  # flash-attn is not installed
sys.path.insert(0, "/content/TRELLIS.2")
from unittest.mock import MagicMock  # nvdiffrast (NVIDIA non-commercial licence) is only needed for rendering; not installed

for _m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec", "nvdiffrec_render"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import torch
import trimesh
from PIL import Image

import trellis2.modules.image_feature_extractor as fe
import trellis2.pipelines.rembg as rembg_mod

# 1. licence-clean background remover: same class, MIT weights
_BiRefNet = rembg_mod.BiRefNet
def _mit_birefnet(model_name=None, **kw):
    m = _BiRefNet(model_name="ZhengPeng7/BiRefNet")
    m.model.float()  # the checkpoint loads in half precision; the input is float32
    return m


rembg_mod.BiRefNet = _mit_birefnet
# 1b. TRELLIS expects DINOv3ViTModel.layer; transformers 5 keeps the layers at model.model.layer
if not hasattr(fe.DINOv3ViTModel, "layer"):
    fe.DINOv3ViTModel.layer = property(lambda self: self.model.layer)
# 2. DINOv3 from the uploaded local copy (the Hugging Face repo is gated)
_orig = fe.DINOv3ViTModel.from_pretrained
fe.DINOv3ViTModel.from_pretrained = classmethod(lambda cls, name, *a, **k: _orig("/content/dino", *a, **k))

from trellis2.pipelines import Trellis2ImageTo3DPipeline

out_path = Path(sys.argv[1])
photos = sorted(p for d in sys.argv[2:] for p in Path(d).glob("*.png"))
print(len(photos), "photos", flush=True)
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.cuda()

ids, pts, secs = [], [], []
done = {}
if out_path.exists():  # resume
    z = np.load(out_path, allow_pickle=False)
    done = dict(zip(z["ids"].tolist(), z["pts"]))
t0 = time.time()
for k, p in enumerate(photos):
    sid = p.stem
    if sid in done:
        ids.append(sid); pts.append(done[sid]); continue
    t1 = time.time()
    try:
        with torch.no_grad():
            image = pipe.preprocess_image(Image.open(p).convert("RGB"))
            torch.manual_seed(0)
            cond = pipe.get_cond([image], 512)
            coords = pipe.sample_sparse_structure(cond, 32, 1, {})
            slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords, {})
            meshes, _ = pipe.decode_shape_slat(slat, 512)
        m = meshes[0]
        mesh = trimesh.Trimesh(m.vertices.float().cpu().numpy(), m.faces.cpu().numpy(), process=False)
        cloud, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0)
        cloud = np.asarray(cloud, np.float32)
    except Exception as e:  # keep going; a NaN cloud counts as a failure when scoring
        print("FAIL", sid, repr(e)[:200], flush=True)
        if not any(np.isnan(x).any() for x in pts):
            import traceback; traceback.print_exc()
        cloud = np.full((8192, 3), np.nan, np.float32)
    ids.append(sid); pts.append(cloud); secs.append(time.time() - t1)
    torch.cuda.empty_cache()
    if (k + 1) % 25 == 0 or k + 1 == len(photos):
        np.savez(out_path, ids=np.array(ids), pts=np.stack(pts))
        print(f"{k + 1}/{len(photos)}  {np.mean(secs):.1f} s/image  elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
print("GEN DONE", flush=True)

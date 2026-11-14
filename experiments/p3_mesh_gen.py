"""P3: single image -> TripoSR mesh -> scale-free shape descriptors (one CSV row per sample, resumable).
Run inside a TripoSR checkout on a GPU VM:  python p3_mesh_gen.py <data root with train/ val/> <rows.csv> <out.csv>
Descriptors (mesh units cancel out): sorted oriented-bbox aspect ratios, fill = V / bbox volume, V / convex hull,
sphericity, plus ln e1 (object extent in the auto-framed image's units, ~constant) and mesh health flags."""
import csv, sys, time
from pathlib import Path
import numpy as np, rembg, torch, trimesh
from PIL import Image
sys.path.insert(0, ".")
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground

root, rows_csv, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
COLS = ["sample_id", "ln_e1", "a2", "a3", "fill", "hull", "spher", "watertight", "n_comp", "sec"]
done = {r["sample_id"] for r in csv.DictReader(open(out))} if out.exists() else set()
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
# TSR.from_pretrained, but the checkpoint uses transformers-4.35 ViT key names; current transformers renamed them
REN = [("encoder.layer.", "layers."), ("attention.attention.query", "attention.q_proj"), ("attention.attention.key", "attention.k_proj"),
       ("attention.attention.value", "attention.v_proj"), ("attention.output.dense", "attention.o_proj"),
       ("intermediate.dense", "mlp.fc1"), ("output.dense", "mlp.fc2")]
cfg = OmegaConf.load(hf_hub_download("stabilityai/TripoSR", "config.yaml")); OmegaConf.resolve(cfg)
model = TSR(cfg)
ckpt = torch.load(hf_hub_download("stabilityai/TripoSR", "model.ckpt"), map_location="cpu")
for k in [k for k in ckpt if k.startswith("image_tokenizer.model.encoder.layer.")]:
    n = k
    for a, b in REN: n = n.replace(a, b)
    ckpt[n] = ckpt.pop(k)
model.load_state_dict(ckpt)
model = model.to("cuda")
model.renderer.set_chunk_size(8192)
sess = rembg.new_session(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])  # CPU rembg is 13 s/image


def descriptors(v: np.ndarray, f: np.ndarray) -> list:
    parts = trimesh.Trimesh(v, f, process=True).split(only_watertight=False)
    m = max(parts, key=lambda p: p.area)  # drop floaters
    trimesh.repair.fill_holes(m)
    e = np.sort(m.bounding_box_oriented.primitive.extents)[::-1]
    V = abs(m.volume)
    return [np.log(e[0]), np.log(e[1] / e[0]), np.log(e[2] / e[0]), np.log(V / e.prod()), np.log(V / m.convex_hull.volume),
            np.log(36 * np.pi * V ** 2 / m.area ** 3) / 3, int(m.is_watertight), len(parts)]


with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if not done: w.writerow(COLS)
    for r in csv.DictReader(open(rows_csv)):
        if r["sample_id"] in done: continue
        t0 = time.time()
        img = resize_foreground(remove_background(Image.open(root / r["split"] / r["group"] / r["category"] / r["sample_id"] / "rgb_industrial.png"), sess), 0.85)
        a = np.asarray(img).astype(np.float32) / 255
        img = Image.fromarray(((a[..., :3] * a[..., 3:] + (1 - a[..., 3:]) * 0.5) * 255).astype(np.uint8))
        with torch.no_grad():
            mesh = model.extract_mesh(model([img], device="cuda"), False, resolution=128)[0]  # 256: same descriptors +-0.01, 4x slower (CPU MC)
        try:
            d = descriptors(mesh.vertices, mesh.faces)
        except Exception as ex:  # empty / degenerate mesh -> NaN row, imputed downstream
            print(r["sample_id"], "FAILED", ex, flush=True); d = [np.nan] * 6 + [0, 0]
        w.writerow([r["sample_id"], *np.round(d, 5), round(time.time() - t0, 2)]); fh.flush()
print("MESH_DONE", flush=True)

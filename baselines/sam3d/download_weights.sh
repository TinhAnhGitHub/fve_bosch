#!/bin/bash
# Download the SAM 3D Objects checkpoints (gated: your Hugging Face account needs approved access to
# facebook/sam-3d-objects) and the MoGe depth model (public), then point the pipeline config at MoGe.
# usage: HF_TOKEN=hf_... bash download_weights.sh      (the token is read from the environment, never stored)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "$HF_TOKEN" ] || { echo "set HF_TOKEN to a read token of an account with access to facebook/sam-3d-objects"; exit 1; }
cd "$HERE/Fast-SAM3D"
"$HERE/.venv-sam3d/bin/python" - <<'PY'
import glob, os, shutil
from huggingface_hub import snapshot_download
d = snapshot_download("facebook/sam-3d-objects", local_dir="checkpoints/hf-download", token=os.environ["HF_TOKEN"], max_workers=4)
for f in glob.glob(f"{d}/checkpoints/*"):
    t = "checkpoints/hf/" + os.path.basename(f)
    if not (os.path.exists(t) and t.endswith(".yaml")):  # keep the repo's yaml files: they include the accelerated variants
        shutil.move(f, t)
snapshot_download("Ruicheng/moge-vitl", local_dir="checkpoints/moge-vitl")
shutil.rmtree("checkpoints/hf-download", ignore_errors=True)
print("weights:", sorted(os.listdir("checkpoints/hf")))
PY
sed -i "s#pretrained_model_name_or_path: .*moge-vitl/model.pt.*#pretrained_model_name_or_path: $HERE/Fast-SAM3D/checkpoints/moge-vitl/model.pt#" checkpoints/hf/pipeline.yaml
grep -n "moge-vitl" checkpoints/hf/pipeline.yaml
echo "WEIGHTS DONE. Next: bash run_all.sh"

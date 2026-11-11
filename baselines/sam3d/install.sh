#!/bin/bash
# Install SAM 3D (Fast-SAM3D, training-free 2.7x speed-up of Meta's SAM 3D Objects) into ./Fast-SAM3D with its own
# Python 3.10 environment in ./.venv-sam3d. Linux or WSL2 with an NVIDIA driver (CUDA 12.1 compatible) is required:
# the prebuilt pytorch3d / kaolin / flash-attn wheels below exist for Linux only.
# usage: bash install.sh            (run from this folder; about 10-15 minutes)
set -ex
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
[ -d Fast-SAM3D ] || git clone -q https://github.com/wlfeng0509/Fast-SAM3D.git
command -v uv > /dev/null || pip install -q uv
unset UV_SYSTEM_PYTHON
export UV_PYTHON_DOWNLOADS=automatic
uv python install 3.10
rm -rf .venv-sam3d
uv venv .venv-sam3d --python 3.10 --python-preference only-managed
PY="$HERE/.venv-sam3d/bin/python"
pipi() { uv pip install --python "$PY" -q "$@"; }

pipi torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pipi "numpy<2" hydra-core==1.3.2 omegaconf loguru opencv-python-headless pillow trimesh plyfile optree easydict scipy imageio \
  scikit-image timm==0.9.16 safetensors seaborn matplotlib plotly open3d==0.18.0 lightning==2.3.3 roma xatlas fvcore iopath einops \
  transformers==4.46.3 huggingface_hub==0.26.5 pymeshfix point-cloud-utils==0.29.5 spconv-cu121==2.3.8 rtree pyrender ftfy jsonlines \
  astor python-igraph pyvista kornia "setuptools<70"
pipi --no-deps xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu121
pipi pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
pipi "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pipi kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html
pipi "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" "git+https://github.com/microsoft/MoGe.git"
(cd Fast-SAM3D && ("$PY" patching/hydra || true) && uv pip install --python "$PY" -q -e . --no-deps)

# Later installs upgrade these; pin them back last (kaolin and flash-attn need torch 2.4.1, kaolin needs numpy 1,
# transformers 4.46 needs huggingface_hub < 1).
pipi --reinstall torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pipi "numpy<2" "huggingface_hub==0.26.5"

# Two fixes to the Fast-SAM3D repo as published: an import of a solver its solver.py has commented out,
# and a DINOv2 path pointing at its authors' disk.
cp fix_flow_matching_model.py Fast-SAM3D/sam3d_objects/model/backbone/generator/flow_matching/model.py
cp fix_dino.py Fast-SAM3D/sam3d_objects/model/backbone/dit/embedder/dino.py

"$PY" -c "import torch, pytorch3d, flash_attn, kaolin, spconv, xformers, moge, kornia, astor; print('imports ok', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "INSTALL DONE. Next: HF_TOKEN=hf_... bash download_weights.sh"

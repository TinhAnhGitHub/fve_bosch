#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
cd /content
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb >/dev/null
apt-get update -qq
apt-get install -y -qq cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-nvrtc-dev-12-6 libcusparse-dev-12-6 libcublas-dev-12-6 libcusolver-dev-12-6 libcurand-dev-12-6 cuda-cccl-12-6 cuda-profiler-api-12-6 >/dev/null 2>&1
ls /usr/local/cuda-12.6/bin/nvcc
pip install -q uv
uv venv /content/venv --python 3.13 -q
. /content/venv/bin/activate
uv pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install -q imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers pandas zstandard kornia timm numpy pillow setuptools wheel scipy
uv pip install -q "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
git clone -q --recursive https://github.com/microsoft/TRELLIS.2.git /content/TRELLIS.2
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=10
for pkg in CuMesh FlexGEMM; do git clone -q --recursive https://github.com/JeffreyXiang/$pkg.git /content/ext_$pkg; uv pip install /content/ext_$pkg --no-build-isolation 2>&1 | tail -3; done
uv pip install /content/TRELLIS.2/o-voxel --no-build-isolation 2>&1 | tail -3
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
echo INSTALL DONE

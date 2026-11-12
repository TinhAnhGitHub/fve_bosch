#!/usr/bin/env bash
# L4 VM setup for TripoSG public-test inference (same pins as the A100 run; diso built for sm_89).
# Log /content/setup.log, ends with SETUP DONE or SETUP FAILED. Needs code.tgz and eval.tar in /content.
set -euo pipefail
trap 'echo "SETUP FAILED at line $LINENO"' ERR
step() { echo; echo "=== $(date -u +%H:%M:%S) $* ==="; }
cd /content
step unpack
mkdir -p /content/f3dr /content/f3d
tar -xzf /content/code.tgz -C /content/f3dr
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step python env
command -v uv >/dev/null || pip install -q uv
[ -x /content/venv/bin/python ] || uv venv -q -p 3.12 /content/venv
PY=/content/venv/bin/python
$PY -V
uvp="uv pip install -q --python $PY"
$uvp torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
$uvp -r /content/f3dr/baselines/generators/triposg/finetune/requirements.txt
$uvp torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
[ -d /content/TripoSG ] || git clone -q https://github.com/VAST-AI-Research/TripoSG.git /content/TripoSG
git -C /content/TripoSG checkout -q fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
$PY -c "import torch, torch_cluster, peft, lightning; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'fp16 ok')"

step diso build against CUDA 12.6
. /etc/os-release; echo "OS $VERSION_ID"
apt-get update -qq >/dev/null 2>&1 || true
if ! apt-cache show cuda-nvcc-12-6 >/dev/null 2>&1; then
  v=$(echo $VERSION_ID | tr -d .)
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$v/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb
  dpkg -i /tmp/ck.deb >/dev/null
  apt-get update -qq >/dev/null 2>&1 || true
fi
apt-get install -y -qq --no-install-recommends cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-cccl-12-6 cuda-nvrtc-dev-12-6 \
  cuda-nvtx-12-6 cuda-profiler-api-12-6 libcublas-dev-12-6 libcusparse-dev-12-6 libcusolver-dev-12-6 libcurand-dev-12-6 \
  > /tmp/apt.log 2>&1 || { tail -5 /tmp/apt.log; echo "apt failed"; false; }
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8
nvcc --version | tail -1
$uvp setuptools wheel ninja
uv pip install --python $PY --no-build-isolation diso 2>&1 | tail -3
cd /tmp && $PY -c "import torch, diso; print('diso ok')"

step contracts
cd /content/f3dr/baselines/generators/triposg/finetune
$PY test_contracts.py --upstream /content/TripoSG -v 2>&1 | tail -8

step done
echo "SETUP DONE"

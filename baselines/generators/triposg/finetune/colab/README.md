# Colab L4 run of the public test (2026-11-12/13)

How the 511 public-test photos were run with the LoRA (epoch 4) and the size head on a Colab L4 (24 GB).
Both scripts run **on the VM**, so the paths are VM paths (`/content/...`). Nothing here uploads anything by itself.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `pub_setup.sh` | Unpacks `/content/code.tgz` (`baselines/geom.py` and `finetune/`), builds a Python 3.12 venv with torch 2.5.1+cu124, the requirements, `torch-cluster` and TripoSG at the pinned commit, installs CUDA 12.6 from NVIDIA's apt repo and builds `diso` for sm_89, then runs `test_contracts.py`. The log ends with `SETUP DONE` or `SETUP FAILED`. Needs the photos in `/content/f3d/eval/`. |
| 2 | `pub_run2.sh` | Cuts the sorted photo list into 12 shards (`rows[s::12]`) and runs two workers on the one GPU (worker 0: shards 2 4 8 10, worker 1: shards 3 5 9 11, started 60 s apart). Each finished shard is `/content/pub2/lora_eval_n12_s<K>.npz`. The log ends with `PUB2 DONE` or `PUB2 FAILED`. |

Measured: 340 photos in 2 h 31 min (about 2.2 photos per minute for both workers together, about 10 GB of GPU memory).

`pub_run2.sh` is the second attempt. The first VM was terminated after 2.5 h (we sent no stop; the cause is unknown) and the photos it
had not yet saved were lost. Its
6-way shards 0 and 1 (`rows[0::6]`, `rows[1::6]`) had been downloaded, and they equal the residues {0, 6} and {1, 7} of the 12-way
split, so shards 2 3 4 5 8 9 10 11 are exactly the rest. Merge all ten files with `finetune/merge.py <files> --output <npz> --expect 511`.

What went wrong before, and what to do:

- `predict.py --shard` writes its npz only when a shard ends. Download each finished shard at once with `colab download` and compare its
  sha256 with the VM's copy.
- Uploads of 100 MB parts through `colab upload` failed; 40 MB parts worked. Error text can contain the runtime-proxy token, so mask it
  before printing.
- Ubuntu 24.04 has no CUDA 12.4 packages; 12.6 works. The image's default CUDA is 13, which torch 2.5.1 will not build extensions against.

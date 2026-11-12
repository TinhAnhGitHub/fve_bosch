# Train food-specific TripoSG adapters or the full transformer

This is an experimental implementation, not an official TripoSG training recipe.
It supports two independent phases: shape adaptation (`lora` or `full`) and an image-to-size MLP (`size`).
Full means every **transformer** parameter. DINOv2 and the VAE remain frozen in both recipes.

Read [the architecture explanation](../../../../docs/TRIPOSG_ARCHITECTURE.md) or open
[the standalone interactive HTML](../../../../docs/triposg_architecture.html).

## Prepare the environment

Use an authorized Colab or Kaggle machine for encoding and model training. The
project plan prohibits local model training. These scripts do not upload anything.
Get explicit approval before transferring raw images, meshes, or new weights to a cloud service.

Use Python 3.10–3.12. From the repository root:

```bash
git clone https://github.com/VAST-AI-Research/TripoSG.git /content/TripoSG
git -C /content/TripoSG checkout fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r baselines/generators/triposg/finetune/requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

The PyG torch-2.5.0 wheel directory supports the torch 2.5.* series. If the
platform has another PyTorch/CUDA pair, install matching wheels instead of mixing binaries.
Do not install upstream `requirements.txt` on top: its NumPy pin targets an older Python stack.
`torch-cluster` is needed only for VAE encoding. TripoSG imports `diso` for mesh
extraction; install it on the CUDA machine before prediction:

```bash
pip install diso
```

On the current Colab and Kaggle images (CUDA 13 by default, Ubuntu 24.04) this fails: torch 2.5.1 is built for
CUDA 12.4 and refuses to build extensions against another major version. Build `diso` against a CUDA 12 toolkit
installed from NVIDIA's apt repo (12-6 works on Ubuntu 24.04; `cuda-nvcc-12-4` is listed there without its
libraries, so check that every package has a candidate), with `--no-build-isolation` and `TORCH_CUDA_ARCH_LIST`
set for the GPU (8.9 for an L4, 7.5 for a T4). `colab/pub_setup.sh` does this on a Colab L4 and
`cuda12_home()` in `kaggle/run.py` does it on Kaggle.

Set `HF_HOME` and cache/output directories to a disk with sufficient free space.
Encoding has a separate memory requirement from training, especially at the
upstream example's 204,800 surface points. The starting recipe uses 32,768 points.
Validate VAE reconstruction before trusting that reduced sample count.

## Build a split manifest

This operation reads local files and hashes, with no GPU or uploads:

```bash
python baselines/generators/triposg/finetune/prepare.py manifest \
  --data ../data/f3d-20261111T150205Z-1-002/f3d \
  --output baselines/out/triposg/manifest.json
```

The layout is `train/group/category/id/{rgb_industrial.png,mesh.ply}`. Missing
pairs fail explicitly. Exact duplicate mesh files stay in the same split. If you
know which scans depict the same physical object, supply `--groups groups.json`,
a JSON map from sample ID to physical-object ID. File hashes alone cannot identify
different scans of the same item. This split differs from the old Atlas holdout:
re-score all comparison models on its IDs, or construct a separate fixed comparison manifest.

## Encode a cache on the cloud machine

Verify coordinate units against `gt.json:bbox_mm` first. `--mesh-unit-mm 1` means
mesh coordinates are already millimetres; use `1000` for metres. Do not correct
large meshes based only on appearance: check the challenge's current unit protocol.

```bash
python baselines/generators/triposg/finetune/prepare.py encode \
  --upstream /content/TripoSG --data /content/f3d \
  --manifest /content/manifest.json --output /content/triposg-cache \
  --mesh-unit-mm 1 --pose pca --tokens 2048 --surface-points 32768
```

`--pose pca` removes arbitrary scan rotation using a right-handed frame. PCA can
be unstable on symmetric shapes. `--pose mesh` preserves file orientation and is
appropriate only when image/mesh poses are meaningful or deliberately retained.
Neither option estimates camera extrinsics. Canonical targets differ from the
pretrained model's image-aligned pose convention; evaluate this domain shift.

Use an empty output folder. `metadata.json` is written only when all objects have
both image features and mesh posteriors. On failure, inspect the error and use a
new output directory; partial caches are not trainable.

The raw RGB input keeps the background. For stronger foreground conditioning,
prepare RGBA images with externally generated object masks before manifesting.
The same crop/padding policy is applied at prediction. No background-removal
weights are implicitly downloaded. No appearance augmentation is applied to a
single cached view. Additional views require regenerating and extending the cache.

## Measure memory, then train shape

Start with LoRA on a 24 GB L4/A10-class GPU. This is the recommended initial
configuration, not a measured optimum. The config name indicates a target GPU,
not a promise that a particular driver/runtime fits.

```bash
python baselines/generators/triposg/finetune/train.py \
  --upstream /content/TripoSG --cache /content/triposg-cache \
  --config baselines/generators/triposg/finetune/configs/lora_24gb.json \
  --output /content/runs/lora-memory --memory-test

python baselines/generators/triposg/finetune/train.py \
  --upstream /content/TripoSG --cache /content/triposg-cache \
  --config baselines/generators/triposg/finetune/configs/lora_24gb.json \
  --output /content/runs/lora
```

The memory test performs one accumulated optimizer update, including Adam state
allocation, and prints peak allocated/reserved GiB. It does not save checkpoints.
Use a fresh output directory for the real run. Size and shape trainers are single
GPU; two Kaggle T4s do not combine into one 32 GB GPU.

For a T4, use `configs/lora_t4_16gb.json`: fp16, rank 8, batch 1. If it does not
fit, use a larger GPU first. Reducing rank cannot remove transformer activation
memory. Do not silently reduce the number of latent tokens without rebuilding
and validating the VAE cache.

To train all transformer weights, replace the config with
`configs/full_a100_80gb.json` and use another output directory. This keeps fp32
master weights and uses bf16 autocast. Conventional Adam state plus gradients
requires substantially more memory. No FSDP, DeepSpeed, or optimizer sharding is implemented.

## Train the size head independently

```bash
python baselines/generators/triposg/finetune/train.py \
  --upstream /content/TripoSG --cache /content/triposg-cache \
  --config baselines/generators/triposg/finetune/configs/size.json \
  --output /content/runs/size
```

The size phase never loads TripoSG, DINOv2, or the VAE. It reads cached image
features and physical radius labels. Log-radius MSE is a stable starting objective,
not an exact optimization of the challenge's squared metric Chamfer. Inspect
linear-radius MSE and final metric CD as well. A later uncertainty head can address
multimodal sizes; this implementation predicts a single log-radius value.

For a controlled loss ablation, use `configs/size_radius_mse.json` in another
output folder. It keeps the head and all other settings unchanged, and minimizes
linear-radius MSE divided by a fixed `40²` unit factor. Compare final metric CD
and radius errors by size range, not the numerical values of the two losses.
The historical generator ~91 CD result used an earlier size model and does not
measure this new MLP. Neither size-loss recipe has been trained yet.

PyTorch Lightning's `Trainer` owns the training loop, automatic backward passes,
mixed precision, accumulation, clipping, and scheduler steps. `FoodAdaptation`
implements only `training_step`, `validation_step`, `configure_optimizers`, and
artifact metadata hooks. `ModelCheckpoint` keeps the best validation checkpoint
and `last.ckpt`. `EarlyStopping` stops after the configured validation patience.
`CSVLogger` records metrics under `logs/`.

Each phase writes `run.json`, Lightning `.ckpt` files, and an exported `best.pt`.
The export contains inference weights and metadata. LoRA exports contain only
adapter weights. Lightning resume checkpoints also include the frozen base model,
optimizer, scheduler, precision-plugin state, and training progress; budget disk
space accordingly. Resume with the SAME config/cache:

```bash
python baselines/generators/triposg/finetune/train.py \
  --upstream /content/TripoSG --cache /content/triposg-cache \
  --config baselines/generators/triposg/finetune/configs/lora_24gb.json \
  --output /content/runs/lora --resume /content/runs/lora/last.ckpt
```

Only load checkpoints you trust. They include Python optimizer/RNG metadata.
Archive the model artifacts and logs promptly when cloud sessions finish.

## Generate and compare predictions

```bash
python baselines/generators/triposg/finetune/predict.py \
  --upstream /content/TripoSG --data /content/f3d \
  --shape /content/runs/lora/best.pt --size /content/runs/size/best.pt \
  --split val --steps 50 --guidance 7 --output /content/lora-val.npz

python baselines/generators/triposg/finetune/evaluate.py \
  --predictions /content/lora-val.npz --cache /content/triposg-cache \
  --data /content/f3d --output /content/lora-val-scores.npz
```

Omit `--shape` for the unchanged pretrained-generator baseline with the SAME size
head. Change it to the full checkpoint for the full-tuning arm. Use identical IDs,
steps, seed, masks, and scoring settings. Prediction also writes a challenge-layout
ZIP containing exactly 8,192 points per object. `--split eval` uses eval images
only, with no ground-truth geometry.

The local evaluator uses proper PCA rotations and 24 rigid ICP starts. It reports
metric CD-L2, oracle-radius CD-L2, and radius MSE. It never fits scale during ICP.
It is NOT verified against the official grader. Oracle-size CD is diagnostic only.
Latent validation loss selects checkpoints but does not guarantee better geometry;
compare generated geometry before promoting a checkpoint.

## Recommended experiment order

| Arm | Shape | Size | Purpose |
|---|---|---|---|
| A | Frozen TripoSG | Learned MLP | Baseline for adaptation |
| B | LoRA rank 16, LR 5e-5 | Same MLP | Recommended first adaptation |
| C | Full transformer, LR 5e-6 | Same MLP | Test if extra trainable capacity helps |

Keep 2,048 tokens, effective batch 16, gradient norm 1, and 10% condition dropout.
Start with uniform sigma sampling. Compare logit-normal only as a controlled ablation.
LoRA targets Q, K, V, and output projections in both self- and cross-attention.
Try rank 8 or LR 2e-5 if validation worsens early. Do not choose a winner from
training loss. Repeat the promising arm with seed 43 on the same split. Expand to
grouped cross-validation before making claims from a small holdout.

There is no measured best configuration or training VRAM result yet. The 24 GB
LoRA and 80 GB full presets are starting targets. Record measured memory and
generated-cloud quality before spending the full training budget.

## Verify implementation without training the model

```bash
python baselines/generators/triposg/finetune/test_contracts.py --upstream /content/TripoSG -v
```

These CPU tests use tiny randomly initialized upstream modules. They test Lightning
fit/checkpoint/resume, LoRA
gradient isolation, save/reload, full-transformer gradients, VAE XYZ+normal input,
flow direction, proper geometry alignment, and split leakage. FPS selection is
stubbed only for the tiny VAE test; real encoding still requires torch-cluster.

## Future directions

Keep **flow matching as the main training objective** while testing alternative
adapters. Adapter architecture determines which parts of TripoSG can change;
the loss determines what those changes learn. Improve either independently.
The current setup is untrained: the ideas below are experiments, not fixes for
demonstrated failures. The alternative adapter designs and geometry auxiliary
loss below are not implemented by the current recipes.

### What the flow-matching loss learns

The generator learns to turn random noise into a shape, guided by the photo.
It operates in the VAE's compressed geometry representation: **latent tokens,
not XYZ points**. In [`common.flow_batch`](common.py), let:

- `z` be a clean ground-truth shape latent from the frozen VAE;
- `ε` be independent standard Gaussian noise with the same tensor shape;
- `σ` be a sampled noise fraction between zero and one.

The noisy input is

```text
x_σ = (1 − σ)z + σε
```

At `σ = 1` this is pure noise; at `σ = 0` it is the clean shape. The transformer
receives this mixture, timestep `1000σ`, and image features. It predicts the
direction and magnitude of movement toward the clean latent:

```text
v_target = z − ε
L_flow = mean((v_θ(x_σ, 1000σ, image_features) − (z − ε))²)
```

For a scalar example, let `z = 2`, `ε = 10`, and `σ = 0.5`. The mixed input is
`6`, and the target velocity is `−8`. Moving through the remaining half of the
ideal straight path gives `6 + 0.5 × (−8) = 2`.

At inference the clean latent is unknown. Start from noise, repeatedly predict
velocity and update the current latent, then decode the final latent into
geometry. Training teaches those movements without running the whole generation
trajectory for every example. See the [flow-matching paper](https://arxiv.org/abs/2210.02747).
The signs here follow TripoSG's released scheduler: it **adds**
`(σ − σ_next) × predicted_velocity`. A different time convention may use the
opposite sign; the target and solver must agree.

### Alternative adapter architectures

| Option | What changes | Experiment rationale |
|---|---|---|
| Attention + feed-forward LoRA | Extend the existing attention adapters to selected transformer MLP projections. | First capacity experiment: adapt token transformations as well as attention. Identify actual upstream projection names before configuring targets. |
| DoRA | Adapt weight magnitude and direction separately, with low-rank direction updates. | Reasonable LoRA comparison; benefits on food geometry remain unproven. |
| Residual bottleneck adapters | Add a nonlinear branch, `h′ = h + W_up φ(W_down h)`. | More flexibility, with extra inference work and implementation complexity. Initialize to preserve pretrained behavior. |
| Image-conditioning adapter | Learn a residual transformation of frozen DINO features before cross-attention. | Tests whether conditioning is a bottleneck; cannot recover information absent from frozen features. |
| Selective block fine-tuning | Fully train a subset of transformer blocks. | Later capacity comparison, with greater memory and overfitting risk than small adapters. |

DoRA's decomposition comes from its [original paper](https://arxiv.org/abs/2402.09353).
That does not establish that DoRA will outperform LoRA on this task.

### Loss and training experiments

1. **Measure validation loss across noise levels.** Report low-, medium-, and
   high-`σ` bins instead of relying on one average. This diagnostic is not yet
   logged. Uniform and logit-normal sampling are already configurable; compare
   them after establishing a baseline. Changing the sampling distribution changes
   how much training each part of the generation path receives.
2. **Evaluate decoded geometry for checkpoint selection.** Latent MSE weights all
   channels equally, while their effects on geometry can differ. Generate a fixed
   validation subset periodically and measure oracle-radius Chamfer. The current
   trainer selects by latent loss; generation and evaluation are separate commands.
   Adding this validation hook could improve selection without changing the main
   objective. Keep physical-size evaluation alongside the shape diagnostic.
3. **Consider a small geometry-aware auxiliary loss later.** With this scheduler,
   the predicted clean latent is

   ```text
   z_hat = x_σ + σ v_θ
   ```

   Pass it through the frozen, differentiable VAE decoder and supervise implicit
   geometry at sampled XYZ query positions. Frozen decoder parameters can remain
   unchanged while gradients propagate through the decoder into the adapter.
   This requires suitable SDF/geometry targets and additional memory. The current
   mesh extraction and surface-sampling pipeline is not a ready-made
   differentiable Chamfer loss. Start with lower-noise examples, where the clean
   latent estimate is less ambiguous, and validate the target construction.

Adding plain latent reconstruction MSE on `z_hat` is **not an independent
geometric objective**. Substituting the mixture equation yields

```text
||z_hat − z||² = σ² ||v_θ − (z − ε)||²
```

It only reweights the existing flow loss by noise level. Decoder-based geometry
supervision would introduce a different signal.

Recommended progression: **train attention LoRA → compare attention + MLP LoRA
→ test noise sampling → consider decoder supervision if latent-loss gains do not
translate into better shapes**. Change one factor at a time, keeping the same
holdout and generation settings. For size, first compare the implemented log-radius
and linear-radius loss recipes with the same head; add richer spatial inputs or
uncertainty modeling only if measured results justify them.

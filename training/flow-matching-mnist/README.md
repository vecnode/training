# flow-matching-mnist

A flow-matching / rectified-flow generative model trained **from scratch**
on MNIST - no diffusion or flow library (no `diffusers`, no `torchcfm`, no
`torchdiffeq`, no `torchvision`). The UNet velocity field, the sinusoidal
time embedding, the probability path, the loss, the EMA, and the Euler/Heun
ODE samplers are all written out by hand in `train_flow.py` /
`evaluate_flow.py`; torch is used only for tensor ops, autograd, and GPU
execution, the same role it plays in `training/mnist-vae` and
`fine-tuning/*-lora` (which likewise write their own training loop rather
than importing one).

This is a `training/` pipeline (from-scratch, non-LoRA), an independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml` (pinned to the CUDA 12.8 torch build, same as
`training/mnist-vae`), own `.python-version`, no shared root environment.

Flow matching is the model family that replaced DDPM in practice - it is
what Stable Diffusion 3 and Flux are trained with. The point of this folder
is that the entire method is about eighty lines of difference from a
diffusion model, and none of those lines are a schedule.

## Method

The model learns a time-dependent velocity field `v(x, t)` whose ODE

```
dx/dt = v(x, t)
```

transports noise `p_0 = N(0, I)` at `t=0` into the data distribution `p_1`
at `t=1`. Training never solves that ODE. For one image `x1` and one noise
draw `x0`, the conditional path between them is a straight line, and its
velocity is regressed directly:

```
t   ~ U(0, 1)                         one t per sample
x0  ~ N(0, I)
x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
u   = x1 - (1 - sigma_min) * x0       the line's velocity, constant in t
L   = || v_theta(x_t, t) - u ||^2     plain MSE
```

That is the conditional optimal-transport path of
[Lipman et al., 2023](https://arxiv.org/abs/2210.02747) with the
independent coupling. At the default `--sigma-min 0.0` it is exactly the
rectified flow of [Liu et al., 2022](https://arxiv.org/abs/2209.03003):
`x_t = (1-t)*x0 + t*x1` and `u = x1 - x0`. The flag exists so the code
carries the general form rather than a simplification of it.

Regressing the *conditional* velocity is enough to recover the *marginal*
field, because `E[u | x_t, t]` is that marginal field and MSE regression
converges to the conditional expectation. This is the whole trick, and it
is why the objective is one `mse_loss` call.

What is absent, compared to a DDPM: no noise schedule (no betas, no
`alpha_bar` table), no variance parameterization, no ELBO, no loss
reweighting, no ancestral sampling chain. Sampling is an ODE solve.

Pixels arrive from `build_mnist_dataset.py` in `[0,1]` and are rescaled to
`[-1,1]` in `train_flow.py`, so the data sits on the same scale as the
`N(0,I)` end of the path. That rescale is deliberately *not* baked into the
dataset builder, which writes the same `data/mnist.npz` contract
`training/mnist-vae` uses so the two pipelines train on byte-identical data.

## Model

A small UNet, `--base-channels C` wide (default 32), channels `C x (1,2,4)`
across three resolutions - **1,175,841 parameters**:

```
stem Conv(1->C)                                        28x28
ResBlock(C)                              -> skip s1    28x28
Downsample(C->2C)                                      14x14
ResBlock(2C)                             -> skip s2    14x14
Downsample(2C->4C)                                       7x7
ResBlock(4C) -> SelfAttention(4C) -> ResBlock(4C)        7x7
Upsample(4C->2C), ResBlock(cat with s2 -> 2C)          14x14
Upsample(2C->C),  ResBlock(cat with s1 -> C)           28x28
GroupNorm -> SiLU -> Conv(C->1)                        28x28
```

`t` enters through a sinusoidal embedding (scaled by 1000, since `t` here
is continuous in `[0,1]` rather than an integer step index) and a two-layer
MLP, added as a per-channel bias inside every ResBlock. The output has the
same shape as the input: it **is** the velocity, not a noise prediction.

Details that are choices, not boilerplate:

- **GroupNorm, not BatchNorm** - a batch mixes wildly different `t`, so
  batch statistics are meaningless here.
- **Zero-initialized output convolution** - the field starts at "no
  movement" instead of at noise, which stabilises the first few hundred
  steps.
- **Nearest-neighbour upsample + conv**, not `ConvTranspose2d` - avoids
  checkerboard artifacts, which show up on generated images far more than
  on reconstructed ones.
- **EMA of the weights** (`--ema-decay 0.999`, hand-written) is what gets
  sampled from, on the standard reasoning that the final weights otherwise
  sit wherever the last SGD step left them. `--weights raw` evaluates the
  non-averaged weights instead.
- **One self-attention block at 7x7** (49 positions, single head, two
  `bmm` calls) so distant strokes of a digit can agree with each other.

None of these five have been ablated in this repo — they are standard
practice for this model family, stated as rationale, not as findings
measured here. `--base-channels`, `--ema-decay`, and `--sigma-min` are CLI
flags precisely so they can be tested rather than assumed.

## Dataset

Point `--data-dir` at a local folder containing the raw MNIST IDX files
(same format/paths as `training/mnist-kmeans` and `training/mnist-vae`) -
not checked into this repo (`data/` is gitignored via the root
`.gitignore`).

## Commands

```sh
uv run --directory training/flow-matching-mnist python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/flow-matching-mnist python train_flow.py --data-path data/mnist.npz --base-channels 32 --num-epochs 40 --batch-size 128 --output-dir runs/mnist_flow
uv run --directory training/flow-matching-mnist python evaluate_flow.py --data-path data/mnist.npz --checkpoint-path runs/mnist_flow/flow_best.pt --num-steps 50 --output-dir runs/mnist_flow
```

One-time environment bootstrap (creates `.venv`, installs the CUDA torch
wheel, verifies `torch.cuda.is_available()`):

```sh
training\flow-matching-mnist\uv_setup.bat
```

`train_flow.py` runs a hand-written Adam training loop (numpy-permutation
batching, no `torch.utils.data.DataLoader`), printing train/val velocity
MSE each epoch, and saves the best-val-loss checkpoint to
`runs/mnist_flow/flow_best.pt` plus a final checkpoint. Both carry the raw
and EMA weights.

`evaluate_flow.py` writes three PNGs (encoded by hand with stdlib `zlib`,
no imaging library) and prints three measurements - see below.

## How this is judged

Sample quality has no honest scalar here. A real FID needs a pretrained
Inception network, which contradicts the from-scratch rule of this folder,
and inventing a substitute number would be worse than not having one. So
`samples_grid.png` is looked at, and everything printed is something that
was actually measured:

- **`samples_grid.png`** - 64 unconditional generations: `x0 ~ N(0,I)`
  integrated to `t=1`. This is the file to put next to `mnist-vae`'s
  `prior_samples.png`.
- **`reconstruction_grid.png`** - digits 0-9 real on top, round-tripped
  below, in the same two-row layout `mnist-vae`'s file uses. A flow model
  has no encoder, but its ODE is deterministic and time-reversible:
  integrating backwards from `t=1` maps a real digit to the noise it would
  have come from, and forwards again brings it back.
- **`nearest_neighbours.png`** + L2 distances - the memorization guard.
  Each generated sample above the closest training image to it (brute-force
  pixel L2 over all 60k, no index library, no learned features). Real
  held-out test digits are measured the same way as a control, because the
  raw distance means nothing on its own — 60k training images cover MNIST's
  simple shapes densely, so even a genuinely novel digit lands close to
  something. Samples much closer than the control would mean memorization.
- **Test-set velocity MSE** - the training objective on genuinely unseen
  data, with `t`/noise drawn from a fixed seed so it is comparable to the
  val numbers in the training log.
- **Solver sweep** - round-trip MAE/PSNR across step counts and both
  solvers. **Read this as ODE discretization error, not quality**: it says
  how many steps the solver needs before the forward and backward
  integrations agree, which is how to choose `--num-steps`. It is not a
  quality score - an untrained model whose velocity field is near zero
  round-trips almost perfectly, since the identity map is its own inverse.
  (Confirmed on a 2-epoch smoke run, which posted a *better* round-trip
  PSNR than a converged model does. That is the trap this note exists for.)

## Verified runs

Real run on the repo owner's RTX 3090, against the actual MNIST IDX files —
the commands in the section above, verbatim:

| | |
|---|---|
| Parameters | **1,175,841** (`--base-channels 32`) |
| Training | 40 epochs, batch 128, Adam 3e-4, EMA 0.999 |
| Wall clock | **317.5 s (~5.3 min)**, ~7.9 s/epoch |
| Train velocity MSE | 0.4419 → 0.1715 |
| Val velocity MSE | 0.2263 → **0.1704** (best, epoch 38) |
| Test velocity MSE | **0.1687** on the held-out 10k, EMA weights |

A second independent run of the same commands on the same machine
reproduced this within run-to-run variance: 312.1 s, best val 0.1705 at
epoch 38, test 0.1687, generated-sample NN mean 4.117. Expect the third
decimal to move; the numbers above are one run, not an average.

Train and val track each other the whole way and the best checkpoint lands
at epoch 38 of 40 — no overfitting, unlike `training/imdb-sentiment-cnn`
where val peaks at epoch 3. The run was still improving slowly at the end;
more epochs is the obvious knob.

**The loss floor is ~0.17, and that is not a defect.** The regression
target `u = x1 - x0` is irreducibly random given `(x_t, t)`: many
(image, noise) pairs pass through the same point at the same time, and the
network can only ever predict their mean. So the MSE floors at the
conditional variance of `u` and cannot go to zero. Judge this pipeline by
`samples_grid.png`, the same way `fine-tuning/vicuna-7b-lora` is judged by
its reconstruction test rather than by its loss plateau.

**Samples** (64 unconditional, 50 Euler steps): mostly clean, unambiguously
readable digits, with a handful of malformed glyphs per grid of 64.

**Solver sweep** (round trip over 256 test images — ODE discretization
error, *not* quality; see the caveat above):

| steps | solver | net evals | MAE | PSNR (dB) |
|---|---|---|---|---|
| 5 | euler | 5 | 0.1161 | 10.91 |
| 10 | euler | 10 | 0.0430 | 16.63 |
| 20 | euler | 20 | 0.0141 | 24.91 |
| 50 | euler | 50 | 0.0054 | 32.45 |
| 100 | euler | 100 | 0.0030 | 36.67 |
| 5 | heun | 10 | 0.0361 | 20.94 |
| 10 | heun | 20 | 0.0090 | 34.40 |
| 20 | heun | 40 | 0.0027 | 44.39 |
| 50 | heun | 100 | 0.0006 | 57.79 |
| 100 | heun | 200 | 0.0002 | 66.38 |

Heun is worth it per *network evaluation*, not just per step: 10 Heun steps
(20 evals) beat 20 Euler steps (20 evals) by ~9.5 dB, and beat 50 Euler
steps (50 evals) by ~2 dB. The second-order correction buys more than
halving the step size does.

**Nearest-neighbour check** (8 samples vs all 60k training images, pixel
L2), with real held-out test digits measured the same way as the control:

| | mean | min | max |
|---|---|---|---|
| generated | 4.029 | 1.858 | 6.067 |
| real test data (control) | 3.611 | 1.169 | 5.109 |

Generated samples sit slightly *farther* from the training set than genuine
unseen digits do — the model is not reproducing its training data.

## Compared with `training/mnist-vae`

Same dataset, same `data/mnist.npz`, same hand-written PNG writer, so
`runs/mnist_flow/samples_grid.png` and
`../mnist-vae/runs/mnist_vae/prior_samples.png` are directly comparable:
both are pure prior samples, decoded from noise with no real image
involved. The flow model's are substantially sharper — the VAE's blur comes
from Gaussian-posterior averaging in the ELBO, the same mechanism
`training/cifar10-vqvae` was built to remove.

**This is a model-family comparison, not a controlled ablation.** The flow
model is a 1,175,841-parameter UNet with skip connections and attention;
the VAE is a 370,945-parameter plain conv encoder/decoder. Some of the gap
is the objective and some is capacity, and these two runs cannot separate
those. What they do show honestly is what each family costs: the VAE
generates in one forward pass, the flow model needs 20–50 network
evaluations per image.

## Files

- `build_mnist_dataset.py` - IDX ubyte parsing, saves `data/mnist.npz` (same contract as `training/mnist-vae`)
- `train_flow.py` - UNet velocity field, time embedding, conditional-OT path, MSE objective, EMA, training loop
- `evaluate_flow.py` - Euler/Heun ODE samplers, test velocity MSE, round-trip solver sweep, nearest-neighbour check, hand-written PNG output

# fashion-mnist-dcgan

A DCGAN ([Radford et al., 2015](https://arxiv.org/abs/1511.06434)) trained
**from scratch** on Fashion-MNIST - no GAN library (no `torchvision`, no
`kagglehub`, no `pytorch-gan-metrics`). The generator, discriminator, the
hand-written `N(0, 0.02)` weight init, one-sided label smoothing, and the
balanced D/G update loop are all written out by hand as plain `torch.nn`
modules in `train_dcgan.py`; torch is used only for tensor ops, autograd,
and GPU execution, the same role it plays in `training/mnist-vae` and
`fine-tuning/*-lora`.

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml` (pinned to the CUDA 12.8 torch build), own
`.python-version`, no shared root environment.

## Model

```
Generator:  z ~ N(0,I) (100-dim) -> Linear -> 7x7x256 -> BN+ReLU
            -> ConvT(4x4,s2) -> 14x14x128 -> BN+ReLU
            -> ConvT(4x4,s2) -> 28x28x64  -> BN+ReLU
            -> ConvT(3x3,s1) -> 28x28x1   -> Tanh          (output in [-1,1])
Discriminator: Conv(4x4,s2) 1->64   -> 14x14x64  -> LeakyReLU(0.2)   [no BN on input]
               Conv(4x4,s2) 64->128 -> 7x7x128   -> BN+LeakyReLU(0.2)
               Conv(4x4,s2) 128->256 -> 3x3x256  -> BN+LeakyReLU(0.2)
               Conv(3x3,s1) 256->1   -> 1x1x1 logit (BCEWithLogits)
```

**28x28 does not divide cleanly down DCGAN's canonical 32x32 ladder.** Three
stride-2 convs take 28 -> 14 -> 7 -> 3, so the discriminator's last feature
map is 3x3 (a final 3x3 conv reduces it to a single logit), and the
generator must start from a 7x7 grid rather than the paper's 4x4. The
shapes above are the verified ones - "fixing" them to the paper's 32x32
numbers breaks the tensors.

## Training

Binary cross-entropy with logits and **one-sided label smoothing** (real
images labeled 0.9 by default, fakes 0.0); **Adam lr 2e-4, betas
(0.5, 0.999)** on both nets (DCGAN's tuned values); one D step and one G
step per batch (balanced k=1), the same fake batch reused (detached for
D's step); every weight initialized `N(0, 0.02)` by hand. A fixed-z sample
grid is saved every `--sample-every` epochs - all grids draw from the same
100 latent points, so you can watch the generator learn (or collapse)
across training.

## Dataset

Point `--data-dir` at a local folder with the raw Fashion-MNIST files.
**Both formats are accepted:**

- the original Zalando **IDX ubyte files** (`train-images-idx3-ubyte`,
  `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`,
  `t10k-labels-idx1-ubyte`) - preferred, parsed by hand with the stdlib
  `struct` module, the same parser as `training/mnist-vae`;
- the **Kaggle CSVs** (`fashion-mnist_train.csv`, `fashion-mnist_test.csv`)
  - parsed with the stdlib `csv` module, no pandas.

`build_fashion_mnist_dataset.py` verifies exactly **60,000 train / 10,000
test** images and refuses to build on a partial extraction, writing
`data/fashion_mnist.npz` (float32 `[0,1]` `X_train`/`X_test`, `y_train`/
`y_test`, plus the 10 class names). The trainer rescales to `[-1,1]` for
the Tanh output. Data is not checked into this repo (`data/` is gitignored
via the root `.gitignore`).

## Commands

```sh
uv run --directory training/fashion-mnist-dcgan python build_fashion_mnist_dataset.py --data-dir "E:\datasets\fashionmnist" --output-dir data
uv run --directory training/fashion-mnist-dcgan python train_dcgan.py --data-path data/fashion_mnist.npz --num-epochs 50 --batch-size 128 --output-dir runs/fashion_mnist_dcgan
uv run --directory training/fashion-mnist-dcgan python evaluate_dcgan.py --data-path data/fashion_mnist.npz --checkpoint-path runs/fashion_mnist_dcgan/dcgan_final.pt --output-dir runs/fashion_mnist_dcgan
```

A 2-epoch smoke run (`--num-epochs 2 --sample-every 1`) checks the tensor
shapes in seconds.

## Evaluation - judge the samples, not the loss

A GAN's D/G loss curves cannot tell a good generator from a collapsed one,
so `evaluate_dcgan.py` emits:

- `samples_grid.png` - generated samples in a square grid (hand-written
  stdlib `zlib` PNG, same writer as `training/mnist-vae`/
  `flow-matching-mnist`);
- `real_samples.png` - one real test example per class, for side-by-side
  comparison;
- the **nearest-neighbour memorization check** (`nearest_neighbours.png` +
  L2 distances): for each generated sample, the pixel-L2 distance to the
  closest training image, printed against the same statistic for a control
  of real training images - samples much closer than the control would mean
  the generator is copying the training set;
- a **pairwise-diversity probe**: mean pairwise L2 over generated samples
  vs real ones - a large gap is the mode-collapse signature.

There is deliberately **no FID / Inception Score**: both need a pretrained
Inception, the same rule that keeps FID out of `flow-matching-mnist` and
ViSQOL/PESQ/NISQA out of `rvq-audio-codec`.

## Files

- `build_fashion_mnist_dataset.py` - IDX ubyte (struct) or Kaggle CSV
  (csv) parsing, verified 60,000/10,000 counts, saves `data/fashion_mnist.npz`
- `train_dcgan.py` - generator/discriminator modules, `N(0,0.02)` init,
  label smoothing, balanced D/G training loop, fixed-z sample grids
- `evaluate_dcgan.py` - sample grid + real grid, nearest-neighbour
  memorization check, pairwise-diversity probe, hand-written PNG output

## Verified runs

**First full run (repo owner's RTX 3090)**: 50 epochs, batch 128, z_dim
100, label smoothing 0.9, Adam 2e-4 / betas (0.5, 0.999). Generator
1,923,969 params, discriminator 659,905. Loss shape: D quickly dominates
(D(x) logit ~ -2.6, i.e. sigmoid ~0.07, by epoch 50, with G loss climbing
to ~3.98) - the classic DCGAN "strong discriminator" regime, which is why
this pipeline judges the generator by its samples, not its losses. All
numbers below are on the epoch-50 checkpoint:

- **Diversity** (mean pairwise L2 over 256 samples): generated 11.51 vs
  real 11.60 (ratio 0.992) - no mode collapse.
- **Nearest training image, mean L2**: generated 4.36 vs held-out real
  control 3.23 - generated samples are no closer to the training set than
  real ones; no memorization.
- **Pixel statistics** (1000 generated vs 1000 real): mean 0.286 vs 0.285,
  std 0.356 vs 0.353, per-image std 0.323 vs 0.322, blank (std < 0.05) 0%
  both, dark (mean < 0.1) 3.9% vs 3.8% - brightness, contrast and
  background fraction match the real distribution.
- **Grids produced**: `samples_epoch0005..0050.png` (fixed-z progression),
  `samples_grid.png`, `real_samples.png`, `nearest_neighbours.png` - the
  fixed-z sequence shows what the generator did across training.

The numbers say the generator has learned the Fashion-MNIST distribution;
a final look at `samples_grid.png` vs `real_samples.png` is the last word.

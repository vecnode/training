# cifar10-vqvae

A VQ-VAE ([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937))
trained **from scratch** on CIFAR-10 - no VQ-VAE library. The encoder,
decoder, the vector quantizer's nearest-neighbor codebook lookup +
straight-through gradient estimator, and the EMA codebook updates are all
written out by hand as plain `torch.nn` modules in `train_vqvae.py`; torch
is used only for tensor ops, autograd, and GPU execution, the same role it
plays in `training/mnist-vae` and `fine-tuning/*-lora`.

This pipeline is the **in-place successor of the former `cifar10-vae`**
(renamed/converted): a plain VAE's blur comes from Gaussian-posterior
averaging in the ELBO - the BCE/KL objective trains the decoder to output
the *mean* of all plausible details. VQ-VAE removes that mechanism: no KL,
a **discrete codebook**, and reconstruction driven purely by MSE recon + a
commitment term, so the decoder is free to reproduce edges and texture
instead of a mean image.

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml` (pinned to the CUDA 12.8 torch build), own
`.python-version`, no shared root environment.

## Model

```
Encoder: Conv(3->64,s2) -> Conv(64->128,s2) -> Conv(128->256,3x3)
         -> Conv(256->D,1x1)                    (32x32 -> 8x8 grid of D-dim vectors)
Vector quantizer: replace each of the 8x8 D-dim vectors with its nearest
                  neighbor in a learned K x D codebook (K=512, D=64 by
                  default); straight-through gradient; EMA codebook
                  updates (entries are moving averages of the encoder
                  outputs that selected them - no gradient into the
                  codebook, the standard fix for codebook collapse)
Decoder: ConvT(D->128,s2) -> ConvT(128->64,s2) -> Conv(64->3,3x3) -> sigmoid
```

Loss = MSE reconstruction + `--commitment-beta` * commitment term
(`||z_e - sg[z_q]||^2`), both derived and computed by hand in
`train_vqvae.py`. With EMA codebook updates there is no separate codebook
loss term. Training logs **codebook perplexity** each epoch - a healthy
codebook uses most of its 512 codes (perplexity climbing toward 512), a
collapsed one only ever uses a handful (perplexity near 1).

**Reconstruction only.** There is no learned prior over the discrete code
grid, so this pipeline can encode a real image, quantize it, and decode it
back - but it cannot sample novel images the way a continuous-latent VAE
can sample from `N(0,I)`. Deliberate scope: the 8x8 discrete code grid is
the substrate for a later learned prior (the planned cascade), and adding
one means training a separate autoregressive/diffusion model over the code
indices.

## Dataset

Point `--data-dir` at a local folder with the raw CIFAR-10 python-format
pickle files (`data_batch_1..5`, `test_batch`, `batches.meta`); a
`cifar-10-batches-py` subfolder is also accepted. The pickles are parsed by
hand with stdlib `pickle` (`encoding='bytes'`, Python-2 format) - no
torchvision, no `keras.datasets` - the same "raw bytes by hand" spirit as
the MNIST IDX parsing in `training/mnist-*`. Data is not checked into this
repo (`data/` is gitignored via the root `.gitignore`).

## Commands

```sh
uv run --directory training/cifar10-vqvae python build_cifar10_dataset.py --data-dir "C:\path\to\cifar-10-python" --output-dir data
uv run --directory training/cifar10-vqvae python train_vqvae.py --data-path data/cifar10.npz --codebook-size 512 --embedding-dim 64 --commitment-beta 0.25 --num-epochs 100 --batch-size 128 --output-dir runs/cifar10_vqvae
uv run --directory training/cifar10-vqvae python evaluate_vqvae.py --data-path data/cifar10.npz --checkpoint-path runs/cifar10_vqvae/vqvae_best.pt --output-dir runs/cifar10_vqvae
```

`build_cifar10_dataset.py` parses the raw pickle batches and writes
`data/cifar10.npz` (50k train / 10k test, CHW [0,1] float32, plus labels
and the 10 class names; `--max-samples` caps each split for a smoke run).
`train_vqvae.py` runs a hand-written Adam training loop (numpy-permutation
batching, no `torch.utils.data.DataLoader`) with cosine LR and optional
horizontal-flip augmentation, holding out `--val-fraction` of the train
split, printing train/val recon + commitment + perplexity each epoch, and
saves the best-val-recon checkpoint to `runs/cifar10_vqvae/vqvae_best.pt`
plus a final checkpoint (checkpoints store the codebook/arch config, so
`evaluate_vqvae.py` can load them). `evaluate_vqvae.py` reports test-set
loss/perplexity plus the same metric suite used to judge the old VAE
(MAE, PSNR, SSIM, high-frequency detail retention - all computed by hand,
SSIM with a Gaussian window, high-frequency energy with a Laplacian), a
codebook-usage check, and writes `reconstruction_grid.png` (hand-written
stdlib `zlib` RGB PNG, no imaging library): one real image per class on
top, its reconstruction below. No `prior_samples.png` - a VQ-VAE has no
`N(0,I)` prior to sample.

## Verified runs (RTX 3090) - vs the predecessor VAE

**v1 - plain VAE, global 128-d latent, 40 epochs** (superseded): PSNR
20.5 dB, SSIM 0.675, MAE 0.071, 47.9% high-frequency detail kept.

**v2 - plain VAE, spatial 4x4 latent + tricks, 100 epochs** (superseded):
PSNR 21.5 dB, SSIM 0.742, MAE 0.065, 51.4% high-frequency detail kept -
real but still visibly soft; the plain-VAE ceiling.

**v3 - VQ-VAE, 8x8 grid x 512 codes, 100 epochs** (current):
codebook 512x64, commitment-beta 0.25, EMA codebook decay 0.99, flip
augmentation, cosine LR, batch 128. Val recon 0.0034 MSE, train/val
tracking all 100 epochs, codebook perplexity ~404/512 during training and
**512/512 codes fired on the test set** (no collapse). ~10–15 min wall
time on the RTX 3090 (742k params, no KL) — roughly 5x faster than the VAE
runs. Measured on the same held-out test set:

| Metric | v1 VAE | v2 VAE | **v3 VQ-VAE** | Δ vs v2 |
|---|---|---|---|---|
| Test loss | 1819.4 (BCE) | 1779.7 (BCE) | 0.0034 (MSE) | — (different objective) |
| MAE | 0.0710 | 0.0647 | **0.0416** | −36% |
| PSNR | 20.5 dB | 21.5 dB | **25.2 dB** | +3.7 dB |
| SSIM | 0.675 | 0.742 | **0.884** | +0.142 |
| High-freq kept | 47.9% | 51.4% | **73.8%** | +22.4 pts |

Per class: every class ≥ 0.84 SSIM (airplane 0.906 / ship 0.901 / dog
0.900 best, frog 0.843 worst — vs the VAE's 0.670 for frog). The VQ-VAE
removes the plain-VAE blur mechanism (Gaussian-posterior averaging) and
the numbers show it: edges/texture are retained at 74% vs 51%. Remaining
softness is the codebook bottleneck itself (8x8 grid x 512 codes), not
the blur averaging of a continuous latent — bigger codebook/grid or a
residual refinement stage would push it further.

## Files

- `build_cifar10_dataset.py` - stdlib-pickle CIFAR-10 batch parsing, saves `data/cifar10.npz`
- `train_vqvae.py` - encoder/decoder modules, vector quantizer (nearest-neighbor + straight-through + EMA updates), training loop
- `evaluate_vqvae.py` - test-set metrics (MAE/PSNR/SSIM/high-freq by hand), codebook usage, hand-written RGB PNG reconstruction grid

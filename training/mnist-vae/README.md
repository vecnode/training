# mnist-vae

A small convolutional VAE trained **from scratch** on MNIST - no VAE
library (no `pythae`, no `lightning-bolts`). The encoder, decoder,
reparameterization trick, and ELBO loss (reconstruction + KL) are all
written out by hand as plain `torch.nn` modules in `train_vae.py`; torch is
used only for tensor ops, autograd, and GPU execution, the same role it
plays in `fine-tuning/*-lora` (which likewise write their own training loop
rather than importing one).

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml` (pinned to the CUDA 12.8 torch build, same as
`fine-tuning/qwen25-3b-lora`), own `.python-version`, no shared root
environment.

## Model

```
Encoder: Conv(1->32, s2) -> Conv(32->64, s2) -> flatten -> fc_mu, fc_logvar
z = mu + eps * exp(0.5 * logvar),  eps ~ N(0, I)
Decoder: fc(z) -> reshape -> ConvT(64->32, s2) -> ConvT(32->1, s2) -> sigmoid
```

Loss = pixel-wise binary cross-entropy (reconstruction) + `--beta` * KL
divergence between `q(z|x)` and the standard normal prior, both derived and
computed by hand in `train_vae.py` (closed-form Gaussian KL, no library
call). `--beta` defaults to 1.0 (standard VAE ELBO); raising it trades
reconstruction sharpness for a more regularized latent space.

## Dataset

Point `--data-dir` at a local folder containing the raw MNIST IDX files
(same format/paths as `training/mnist-kmeans`) - not checked into this repo
(`data/` is gitignored via the root `.gitignore`).

## Commands

```sh
uv run --directory training/mnist-vae python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/mnist-vae python train_vae.py --data-path data/mnist.npz --latent-dim 32 --beta 1.0 --num-epochs 30 --batch-size 128 --output-dir runs/mnist_vae
uv run --directory training/mnist-vae python evaluate_vae.py --data-path data/mnist.npz --checkpoint-path runs/mnist_vae/vae_best.pt --output-dir runs/mnist_vae
```

`build_mnist_dataset.py` parses the raw IDX files and writes
`data/mnist.npz`. `train_vae.py` runs a hand-written Adam training loop
(numpy-permutation batching, no `torch.utils.data.DataLoader`), printing
train/val reconstruction and KL loss each epoch, and saves the
best-val-loss checkpoint to `runs/mnist_vae/vae_best.pt` plus a final
checkpoint. `evaluate_vae.py` reports test-set reconstruction/KL loss and
writes two PNGs (encoded by hand with stdlib `zlib`, no imaging library):
`reconstruction_grid.png` (one real digit per class on top, its
reconstruction below) and `prior_samples.png` (pure `z ~ N(0,I)`
generations, no encoder involved).

## Files

- `build_mnist_dataset.py` - IDX ubyte parsing, saves `data/mnist.npz`
- `train_vae.py` - encoder/decoder modules, reparameterization trick, ELBO loss, training loop
- `evaluate_vae.py` - test-set metrics, hand-written PNG grid output (reconstructions + prior samples)

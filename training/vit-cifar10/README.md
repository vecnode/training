# vit-cifar10

A Vision Transformer ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929),
in the pre-LN / norm-first layout popularized by
[DeiT](https://arxiv.org/abs/2012.12877)) trained **from scratch** on
CIFAR-10 — no `transformers`/`timm`/`torchvision` library. The patch
embedding, the learned CLS token + positional embeddings, the transformer
blocks, and the multi-head self-attention (QKV projections, scaled
dot-product, output projection) are all written out by hand as plain
`torch.nn` modules in `train_vit.py`; torch is used only for tensor ops,
autograd, and GPU execution, the same role it plays in every other
`training/` pipeline (no `DataLoader`, numpy-permutation batching).

This is the **repo's first attention-based vision model — and its first
from-scratch transformer of any kind**: every other `training/` pipeline
is a conv net, an autoencoder, a GAN, or a numpy/classic-ML model. It is
also the natural encoder backbone for a planned I-JEPA-style
self-supervised pipeline (same patch embed + transformer blocks, reused).

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder — own `pyproject.toml` (pinned to
the CUDA 12.8 torch build), own `.python-version`, no shared root
environment.

## Model

```
Patch embed:  Conv2d(3 -> dim, kernel=stride=4)         32x32 -> 8x8 = 64 patches
Prepend       learned CLS token; add learned positional embeddings (65 positions)
Blocks xN     pre-LN (norm-first) transformer block:
                  x = x + MHA(norm(x))    (hand-written multi-head self-attention)
                  x = x + MLP(norm(x))    (Linear -> GELU -> Dropout -> Linear -> Dropout)
Final         LayerNorm -> Linear(dim -> 10) read off the CLS position
```

Defaults `--dim 384 --depth 6 --heads 6 --mlp-ratio 4` = **~10.7M params**.
No pretrained weights anywhere: patch embed, CLS/pos embeddings and every
block are randomly initialized (truncated normal 0.02, the standard ViT
init). The norm-first (pre-LN) layout is the modern choice — it trains
stably from scratch without a huge pretraining budget.

## Augmentation (plain torch ops)

Per batch in the training loop: random horizontal flip (`torch.flip`),
4px zero-pad + random crop back to 32x32, then per-channel normalization
with the hardcoded CIFAR-10 training-set statistics (mean
0.4914/0.4822/0.4465, std 0.2470/0.2435/0.2616 — the standard published
values, computed from the data, not a pretrained network). Validation and
test batches get normalization only. `--no-augment` turns flip+crop off
for a clean A/B of what the augmentation is worth (on CIFAR-10 ViTs,
flip+crop is worth roughly +8–10 points — the difference between ~75% and
~85%).

## Dataset

Point `--data-dir` at a local folder with the raw CIFAR-10 python-format
pickle files (`data_batch_1..5`, `test_batch`, `batches.meta`); a
`cifar-10-batches-py` subfolder is also accepted. The pickles are parsed by
hand with stdlib `pickle` (`encoding='bytes'`, Python-2 format) — no
torchvision, no `keras.datasets` — the same "raw bytes by hand" spirit as
the MNIST IDX parsing in `training/mnist-*`. Same data contract as
`training/cifar10-vqvae`: `data/cifar10.npz` with 50k train / 10k test,
CHW `[0,1]` float32, plus labels and the 10 class names. Data is not
checked into this repo (`data/` is gitignored via the root `.gitignore`).

## Commands

```sh
uv run --directory training/vit-cifar10 python build_cifar10_dataset.py --data-dir "C:\path\to\cifar-10-python" --output-dir data
uv run --directory training/vit-cifar10 python train_vit.py --data-path data/cifar10.npz --num-epochs 60 --batch-size 128 --output-dir runs/vit_cifar10
uv run --directory training/vit-cifar10 python evaluate_vit.py --data-path data/cifar10.npz --checkpoint-path runs/vit_cifar10/vit_best.pt --output-dir runs/vit_cifar10
```

`build_cifar10_dataset.py` parses the raw pickle batches and writes
`data/cifar10.npz` (`--max-samples` caps each split for a smoke run).
`train_vit.py` runs a hand-written AdamW training loop (numpy-permutation
batching, no `torch.utils.data.DataLoader`) with weight decay 0.05 and a
hand-written linear-warmup-then-cosine LR schedule (the warmup ViTs need
to train stably from scratch, on top of the cosine anneal the rest of
`training/` uses), holding out `--val-fraction` of the train split,
printing train/val loss + accuracy each epoch, and saves the best-val-acc
checkpoint to `runs/vit_cifar10/vit_best.pt` plus a final checkpoint
(checkpoints store the arch config, so `evaluate_vit.py` can rebuild the
model). `evaluate_vit.py` reports test top-1 and top-5 accuracy, per-class
accuracy + confusion matrix (row sums = 1,000), writes `test_metrics.txt`
and a hand-written zlib RGB `predictions_grid.png` (no Pillow): 64 test
images — first 32 correct, first 32 misclassified — green-bordered for
correct, red for wrong — plus a console listing of example indices with
class names and confidence. There is deliberately **no pretrained-feature
evaluation of any kind** (no FID/IS style scores): the model is judged by
its own test accuracy and by looking at what it confuses, the same rule
that keeps FID out of `flow-matching-mnist` and ViSQOL out of
`rvq-audio-codec`.

## Verified runs (RTX 3090)

**v1 — defaults, 60 epochs** (current, run by the repo owner): 10,695,562
params, batch 128, flip+crop, AdamW 1e-3 / wd 0.05, 5-epoch warmup then
cosine to 1e-5. Train acc 0.3000 → 0.7138, val acc 0.3782 → **0.6750**
(best at epoch 60; the last ~10 epochs plateaued in a ~0.67 band). Wall
time **1,367 s (~23 min)** fp32. Measured on the held-out 10k test split:

| Metric | Value |
|---|---|
| Test top-1 | **66.82%** (6,682 / 10,000) |
| Test top-5 | **97.32%** |
| Best val acc | 67.50% (epoch 60) |

Per class (1,000 each): frog **81.4%** / ship 78.0% / airplane 74.6% /
truck 73.3% / automobile 71.7% / horse 69.9% / dog 60.4% / deer 58.6% /
bird 55.6% / cat **44.7%**. The confusion matrix shows the classic
CIFAR-10 failure pattern, not a broken model: cat↔dog is the worst pair
(23.4% of cats predicted as dog), bird spreads across the other animals,
and truck↔automobile cross-confuse (12.7% / 15.3%) — the same
animal-vs-animal and car-vs-truck confusions every CIFAR-10 model has.

**An honest correction to this README's earlier estimate:** the "~80–86%
top-1 at 60 epochs" guess was too optimistic for this setup. Measured
reality is **66.8% top-1** with flip+crop only. Published ~90% CIFAR-10
ViT numbers come from 200–300+ epoch runs with much stronger augmentation
(AutoAugment/RandAugment, mixup/cutmix) and usually weight EMA — all
deliberately out of this pipeline's minimal scope. Natural rungs from
here, in rough order: the `--no-augment` A/B (still unmeasured); a longer
run (`--num-epochs 120` — the cosine is designed for the full budget and
the val curve was still slowly improving at the end); stronger
hand-written augmentation; and a deeper/wider variant (`--depth 8
--dim 512`, ~24M params, still comfortable on 24 GB).

## Files

- `build_cifar10_dataset.py` — stdlib-pickle CIFAR-10 batch parsing, saves `data/cifar10.npz`
- `train_vit.py` — patch embed, CLS/pos embeddings, pre-LN blocks, hand-written multi-head self-attention, augmentation, AdamW + warmup/cosine training loop
- `evaluate_vit.py` — test top-1/top-5, per-class accuracy + confusion matrix, `test_metrics.txt`, hand-written zlib RGB `predictions_grid.png`

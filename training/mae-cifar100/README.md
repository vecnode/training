# mae-cifar100

A Masked Autoencoder ([He et al., 2022](https://arxiv.org/abs/2111.06377))
trained **from scratch** on CIFAR-100 — the repo's **first
representation-learning (self-supervised) pipeline**, and the
mask-reconstruct sibling of the I-JEPA-style rung already planned in
`ARCHITECTURE.md`. No `transformers`/`timm`/`torchvision` library: the
patch embedding, positional embeddings, the pre-LN transformer blocks, the
hand-written multi-head self-attention (QKV projections, scaled
dot-product, output projection), the random masking, and the lightweight
decoder are all written out by hand as plain `torch.nn` modules in
`train_mae.py`; torch is used only for tensor ops, autograd, and GPU
execution, the same role it plays in every other `training/` pipeline (no
`DataLoader`, numpy-permutation batching).

The **encoder is literally vit-cifar10's patch-embed/block stack** (copied
in by hand — each `training/` project is independent and never imports
another pipeline folder's code), at a denser patch grid by default: patch 2
→ 16×16 = **256 patches**, 75% masked → 64 visible per image, the paper's
masking regime at 32×32 (the sibling's patch-4 → 64-patch config stays
available via `--patch-size 4`). The **decoder is a separate, smaller
transformer that exists only for pretraining** — the linear probe discards
it and reads frozen encoder features.

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder — own `pyproject.toml` (pinned to
the CUDA 12.8 torch build), own `.python-version`, no shared root
environment.

## Model

```
patchify   32x32 / 2x2 patches           -> 16x16 = 256 patches
mask       fixed 75% per image (per-sample random permutation, not Bernoulli)
encoder    visible patches only -> vit-cifar10's stack:
              Conv2d(3 -> dim, kernel=stride=2) patch embed
              + learned positional embeddings (256 positions, no CLS token)
              + N pre-LN blocks (x = x + MHA(norm(x)); x = x + MLP(norm(x)))
              + final LayerNorm
decoder    latent + learned shared mask token, unshuffled to full order,
           + full decoder pos embed -> decoder blocks (--decoder-dim 192,
           --decoder-depth 2) -> Linear(decoder_dim -> patch^2 * 3)
loss       MSE on the masked patches only, targets normalized per patch
           (subtract patch mean / divide patch std — the MAE trick)
```

Defaults `--patch-size 2 --dim 384 --depth 6 --heads 6 --mlp-ratio 4
--decoder-dim 192 --decoder-depth 2` = **11,766,540 params** (~10.75M
encoder + ~1.02M decoder), ~36 min for 60 epochs fp32 on a single RTX 3090
(smoke-measured ~35 s/epoch; the verified run took 2,166 s). No
pretrained weights anywhere: everything is randomly initialized (truncated
normal 0.02, the standard ViT init — including the positional embeddings
and the mask token, per He et al.; vit-cifar10 leaves its pos embed at
zero init, this pipeline deliberately does not).

## Masking

`random_masking` in `train_mae.py` is the paper's scheme, not a per-patch
Bernoulli: each sample gets its own random permutation of the 256 patch
positions and keeps the first `round((1 - mask_ratio) * 256)` — so every
image is masked at *exactly* 75%, and the encoder's batch is one flat
(B×64, dim) token tensor. The decoder gets the visible tokens plus the
learned mask token at the masked positions (via the inverse permutation
`ids_restore`), so it must reconstruct a full image from 25% of it. The
loss touches **only the masked patches** (multiplied by the mask), with
targets normalized per patch — without that normalization the decoder can
cheat by predicting near-constant patches (low patch variance makes plain
pixel MSE small); `--no-patch-norm` turns it off for a clean A/B.

## The judge: a hand-written linear probe

MAE is self-supervised, so its training loss alone doesn't say how good
the features are. This pipeline is judged by `linear_probe.py`: a single
linear layer trained **from scratch** on the encoder's **frozen** features
(mean-pooled patch tokens after the final norm — MAE has no CLS token),
then scored on the held-out 10k test split. The probe is the point: the
features are the model's own, learned purely from masked-patch
reconstruction with no labels in pretraining, so probing them is a
trained-from-scratch evaluation that does **not** violate this repo's
no-pretrained-features rule the way a FID/Inception score would. Protocol:
features are precomputed once (normalize-only inputs — no probe-time
flip+crop, a documented simplification; the probe sees each image exactly
once), then the head trains with hand-written SGD momentum + cosine LR
(the paper's probe optimizer, not AdamW). The probe reports test top-1 /
top-5, coarse-label (20 superclass) top-1, per-class accuracy + 100×100
confusion matrix, `test_metrics.txt`, and a hand-written zlib RGB
`probe_grid.png` (first 16 correct, first 16 misclassified, green/red
borders — no Pillow).

## Augmentation (plain torch ops)

Per batch in the training loop: random horizontal flip (`torch.flip`),
4px zero-pad + random crop back to 32×32. Pixels stay raw `[0,1]` — **no
per-channel normalization during pretraining**, because the reconstruction
targets *are* the pixels and normalization would move them (the probe is
where inputs get normalized, with the hardcoded CIFAR-100 train
statistics). `--no-augment` turns flip+crop off for a clean A/B.

## Dataset

Point `--data-dir` at a local folder with the raw CIFAR-100 python-format
pickle files (`meta`, `train`, `test`); a nested `cifar-100-python/`
subfolder (the tarball layout) is also accepted. The pickles are parsed by
hand with stdlib `pickle` (`encoding='bytes'`, Python-2 format) — no
torchvision, no `keras.datasets` — the same "raw bytes by hand" spirit as
the CIFAR-10 parsing in `training/vit-cifar10`. Unlike CIFAR-10's five
train batches, CIFAR-100 ships one 50k train file and one 10k test file;
the builder verifies **exactly 50,000/10,000** and refuses a partial or
corrupt extraction. Writes `data/cifar100.npz`: CHW `[0,1]` float32
images, fine (100-class) + coarse (20-superclass) labels for both splits,
plus both name lists. Data is not checked into this repo (`data/` is
gitignored via the root `.gitignore`).

## Commands

```sh
uv run --directory training/mae-cifar100 python build_cifar100_dataset.py --data-dir "E:\datasets\cifar-100-python" --output-dir data
uv run --directory training/mae-cifar100 python train_mae.py --data-path data/cifar100.npz --num-epochs 60 --batch-size 128 --output-dir runs/mae_cifar100
uv run --directory training/mae-cifar100 python linear_probe.py --data-path data/cifar100.npz --checkpoint-path runs/mae_cifar100/mae_best.pt --output-dir runs/mae_cifar100
```

`build_cifar100_dataset.py` parses the raw pickles and writes
`data/cifar100.npz` (`--max-samples` caps each split for a smoke run).
`train_mae.py` runs a hand-written AdamW training loop (numpy-permutation
batching, no `DataLoader`) with weight decay 0.05 and the same
linear-warmup-then-cosine LR schedule as `vit-cifar10` (warmup is the part
ViTs need from scratch), holding out `--val-fraction` of the train split,
printing the masked-MSE train loss and a deterministic full-image
reconstruction-MSE val loss each epoch, saving the best-val checkpoint to
`runs/mae_cifar100/mae_best.pt` plus a final checkpoint, and writing a
`recon_epochNNNN.png` grid (original / masked / reconstruction rows for 8
fixed validation images) every `--sample-every` epochs. `linear_probe.py`
freezes the encoder, precomputes features, trains the probe head, and
writes `test_metrics.txt` + `probe_head.pt` + `probe_grid.png`. Probe
*both* `mae_best.pt` and `mae_final.pt` — the verified run showed the
final checkpoint's features probe better than the best-val one's (see
Verified runs). There is
deliberately **no pretrained-feature evaluation of any kind** (no
FID/Inception-style scores): the model is judged by the linear probe on
its own features, the same rule that keeps FID out of
`flow-matching-mnist` and ViSQOL out of `rvq-audio-codec`.

## Verified runs

**v1 — full 60-epoch run, defaults** (repo owner, RTX 3090): 11,766,540
params, patch 2 → 256 patches (64 visible at mask 75%), batch 128,
flip+crop, AdamW 1e-3 / wd 0.05, 5-epoch warmup then cosine to 1e-5.
Train masked-MSE 0.666 → **0.257** (still slowly improving at the end;
targets are patch-normalized, so ~1.0 is the predict-the-mean floor); val
full-image recon-MSE bottomed at **0.47463 (epoch 34)** then drifted up to
0.4905 by epoch 60. Wall time **2,166 s (~36 min)** fp32. Both checkpoints
were probed (60 probe epochs, SGD lr 0.1 / momentum 0.9 + cosine,
held-out 10k test):

| Checkpoint | Val recon-MSE | Test top-1 | Test top-5 | Coarse top-1 |
|---|---|---|---|---|
| `mae_best.pt` (epoch 34) | 0.47463 | 24.65% (2,465 / 10,000) | 53.15% | 36.80% |
| `mae_final.pt` (epoch 60) | 0.49050 | **25.56%** (2,556 / 10,000) | **53.41%** | **37.89%** |

**The final checkpoint is the record — and a genuine finding.** The
best-val checkpoint is selected by *reconstruction*, and val recon bottomed
at epoch 34 while the masked-MSE kept improving, so the late features probe
better (+0.91 top-1) even though full-image recon drifted. Probe both
checkpoints; don't assume `mae_best.pt` holds the best representation.

Per class (final checkpoint, 100 each): oak_tree **69.0%** / wardrobe
68.0% / sunflower 64.0% / plain 63.0% / skyscraper 62.0% best; bowl 1.0% /
mouse 1.0% / lamp 2.0% / otter 2.0% / snail 2.0% worst — the classic
CIFAR-100 pattern (distinctive, low-intra-class-variance classes are
linearly separable; small or texture-heavy objects at 32×32 are not). The
probe head does not overfit (probe train 27.1% vs test 25.56%).

**An honest correction to this README's earlier estimate:** the "~30–45%
top-1" guess was too optimistic for this setup — 60 epochs on 45k images,
no probe-time augmentation. Measured reality is **25.56% top-1** on the
final checkpoint (~25.6× chance, correct class in the top 5 more than
half the time). Published higher CIFAR-100 linear-probe numbers for MAE
come from much longer pretraining (200–400+ epochs), larger batches, and
usually probe-time augmentation — all deliberately out of this pipeline's
minimal scope. Natural rungs from here, in rough order: the
`--no-patch-norm` A/B (measures the MAE trick); the `--patch-size 4` A/B
(the sibling's literal 64-patch grid — the denser patch-2 default is
expected to win, but it's unmeasured); a `--mask-ratio` sweep (50/75/90);
more pretraining epochs (`--num-epochs 120`, the cosine is designed for
the full budget); and then the I-JEPA rung, which this pipeline's
mask-reconstruct machinery sets up directly.

## Files

- `build_cifar100_dataset.py` — stdlib-pickle CIFAR-100 parsing (fine +
  coarse labels, verified 50k/10k), saves `data/cifar100.npz`
- `train_mae.py` — `MaskedAutoencoderViT` (vit-cifar10 encoder stack + mask
  token + lightweight decoder), per-sample fixed-count masking, per-patch
  normalized masked-MSE loss, flip+crop augmentation, AdamW +
  warmup/cosine training loop, hand-written zlib recon grids
- `linear_probe.py` — frozen-feature linear probe (SGD momentum + cosine),
  test top-1/top-5, coarse top-1, per-class + 100×100 confusion matrix,
  `test_metrics.txt`, hand-written zlib `probe_grid.png`
- `png_utils.py` — shared hand-written zlib RGB PNG writer (no Pillow)

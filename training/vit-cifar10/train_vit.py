"""
A Vision Transformer ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929),
pre-LN / norm-first layout as popularized by DeiT) trained **from scratch**
on CIFAR-10 - no transformers/timm/torchvision library. The patch
embedding, the learned CLS token + positional embeddings, the transformer
blocks, and the multi-head self-attention (QKV projections, scaled
dot-product, output projection) are all written out by hand as plain torch
modules; torch is used only for tensor ops, autograd, and GPU execution,
the same role it plays everywhere else in this repo (no DataLoader,
numpy-permutation batching).

This is the repo's first attention-based vision model - and its first
from-scratch transformer of any kind. The same patch-embed + block stack
is the encoder a planned I-JEPA-style self-supervised pipeline would reuse.

The model (32x32 RGB in, 10 logits out):
    Patch embed: Conv2d(3 -> dim, kernel=stride=4)  -> 8x8 = 64 patches
    Prepend learned CLS token; add learned positional embeddings (65 positions)
    N pre-LN transformer blocks (norm-first, the stable-from-scratch layout):
        x = x + MHA(norm(x))     (hand-written multi-head self-attention)
        x = x + MLP(norm(x))     (Linear -> GELU -> dropout -> Linear -> dropout)
    Final LayerNorm -> Linear(dim -> 10) read off the CLS position

Default sizing (dim 384, depth 6, heads 6, mlp_ratio 4) is ~10.7M params -
a from-scratch CIFAR-10 ViT that fits a single RTX 3090 with room to spare
(~1.5-3 h for 60 epochs fp32). No pretrained weights anywhere: the patch
embed, CLS/pos embeddings and every block are randomly initialized
(truncated normal 0.02, the standard ViT init).

Augmentation is plain torch ops, applied per batch in the training loop:
random horizontal flip, 4px zero-pad + random crop back to 32x32, then
per-channel normalization with the hardcoded CIFAR-10 training-set
statistics (mean 0.4914/0.4822/0.4465, std 0.2470/0.2435/0.2616 - the
standard published values, not a pretrained network). Validation and test
batches get normalization only. `--no-augment` turns flip+crop off for a
clean A/B of how much the augmentation is worth.

Loss: cross-entropy over the 10 classes. Optimizer: AdamW (the ViT
default, unlike the plain-Adam trainers in this repo) with weight decay,
a hand-written linear-warmup-then-cosine LR schedule. A 10% holdout of
the training split is used for validation / best-checkpoint selection;
the official 10k test split stays fully unseen until evaluate_vit.py.

Usage:
    uv run --directory training/vit-cifar10 python train_vit.py \
        --data-path data/cifar10.npz \
        --num-epochs 60 \
        --batch-size 128 \
        --output-dir runs/vit_cifar10
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
NUM_CLASSES = 10

# Hardcoded CIFAR-10 training-set statistics (per-channel mean/std, the
# published values), used for normalization. Computed from the data, not a
# pretrained network - the same role stdlib pickle plays for parsing.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention written out by hand: one QKV
    projection, reshaped to (heads, head_dim), softmax attention, output
    projection. No nn.MultiheadAttention - the math stays visible."""

    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # (B, T, 3*h, head_dim) -> split q/k/v, each (B, heads, T, head_dim)
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    """Pre-LN (norm-first) transformer block: the modern ViT layout that
    trains stably from scratch without a huge pretraining budget."""

    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Patch embed -> CLS + pos embed -> N pre-LN blocks -> final norm ->
    linear head on the CLS token. No pretrained weights: everything is
    randomly initialized (truncated normal 0.02, the standard ViT init)."""

    def __init__(self, patch_size=4, dim=384, depth=6, heads=6,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        n_patches = (IMAGE_SIZE // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 64, dim)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # read off the CLS position


def augment_batch(x, pad=4):
    """Random horizontal flip + 4px zero-pad + random crop back to 32x32 +
    per-channel normalization - all plain torch ops, applied per batch.
    x: (B, 3, 32, 32) float [0,1]."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    x = F.pad(x, (pad, pad, pad, pad))  # (B, 3, 40, 40)
    offs = torch.randint(0, 2 * pad + 1, (B, 2), device=x.device)
    out = torch.empty_like(x[:, :, :IMAGE_SIZE, :IMAGE_SIZE])
    for i in range(B):
        h, w = int(offs[i, 0]), int(offs[i, 1])
        out[i] = x[i, :, h:h + IMAGE_SIZE, w:w + IMAGE_SIZE]
    mean = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (out - mean) / std


def normalize_batch(x):
    mean = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def iterate_batches(X, y, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - one permutation
    per epoch, sliced into batches, so the batching logic stays visible."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def run_epoch(model, X, y, batch_size, device, rng, optimizer=None,
              augment=False):
    """One pass over (X, y). If optimizer is given, trains (shuffled, grad
    updates, flip+crop augmentation); otherwise evaluates (no shuffle, no
    grad, normalization only) - shared loop so train/val accounting can't
    drift apart. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    n_seen = 0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, yb in iterate_batches(X, y, batch_size, rng, shuffle=training):
            x = torch.from_numpy(xb).to(device)
            x = augment_batch(x) if (training and augment) else normalize_batch(x)
            t = torch.from_numpy(yb).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, t)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == t).sum().item()
            n_seen += t.numel()
            n_batches += 1

    return total_loss / n_batches, total_correct / n_seen


def make_lr_lambda(warmup_epochs, total_epochs, eta_min_ratio=0.01):
    """Hand-written linear-warmup-then-cosine LR multiplier: ramps 0 -> 1
    over the warmup epochs, then cosine-anneals down to eta_min_ratio. The
    same schedule shape the rest of training/ uses (CosineAnnealingLR) plus
    the warmup ViTs need to train stably from scratch."""

    def lr_lambda(epoch):  # 0-based epoch from the scheduler
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return lr_lambda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar10.npz")
    parser.add_argument("--output-dir", default="runs/vit_cifar10")
    parser.add_argument("--patch-size", type=int, default=4,
                        help="Side length of each patch (32x32 / patch^2 patches).")
    parser.add_argument("--dim", type=int, default=384,
                        help="Transformer embedding dimension.")
    parser.add_argument("--depth", type=int, default=6,
                        help="Number of transformer blocks.")
    parser.add_argument("--heads", type=int, default=6,
                        help="Number of attention heads.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="MLP hidden size as a multiple of --dim.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout inside each block's MLP and attention.")
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW weight decay (the ViT default).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup before the cosine anneal.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable flip+crop augmentation on training "
                             "batches (A/B of what augmentation is worth).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training rows held out for "
                             "validation (the CIFAR-10 test split stays "
                             "fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_train_full = data["X_train"]  # (n, 3, 32, 32) float32 in [0,1]
    y_train_full = data["y_train"]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"patch={args.patch_size}  dim={args.dim}  depth={args.depth}  "
          f"heads={args.heads}  mlp_ratio={args.mlp_ratio}  "
          f"dropout={args.dropout}  augment={not args.no_augment}")

    model = VisionTransformer(
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(args.warmup_epochs, args.num_epochs),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = -1.0
    best_path = os.path.join(args.output_dir, "vit_best.pt")
    final_path = os.path.join(args.output_dir, "vit_final.pt")
    ckpt_meta = {
        "arch": "vit_cifar10",
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
    }

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, X_train, y_train, args.batch_size, device, rng,
            optimizer=optimizer, augment=not args.no_augment,
        )
        scheduler.step()

        val_loss, val_acc = run_epoch(
            model, X_val, y_val, args.batch_size, device, rng, optimizer=None
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            elapsed = time.time() - start_time
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: loss={train_loss:.4f} acc={train_acc:.4f}  "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.0f}s elapsed)"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(),
                 "epoch": epoch, "val_acc": val_acc,
                 **ckpt_meta},
                best_path,
            )

    torch.save(
        {"model_state": model.state_dict(),
         "epoch": args.num_epochs, "val_acc": val_acc,
         **ckpt_meta},
        final_path,
    )
    print(f"Saved best checkpoint (val_acc={best_val_acc:.4f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_vit.py against the held-out test split next.")


if __name__ == "__main__":
    main()

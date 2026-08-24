"""
A Masked Autoencoder ([He et al., 2022](https://arxiv.org/abs/2111.06377))
trained **from scratch** on CIFAR-100 - the repo's first
representation-learning (self-supervised) pipeline, and the
mask-reconstruct sibling of the I-JEPA-style rung already planned in
ARCHITECTURE.md. No transformers/timm/torchvision library: the patch
embedding, positional embeddings, transformer blocks, hand-written
multi-head self-attention, random masking, and the lightweight decoder are
all plain torch modules; torch is used only for tensor ops, autograd, and
GPU execution (no DataLoader, numpy-permutation batching).

The recipe is exactly the paper's, at 32x32:

    patchify  ->  32x32 / 2x2 patches = 16x16 = 256 patches
    mask      ->  a fixed 75% of patches per image, chosen by a per-sample
                  random permutation (not a per-patch Bernoulli)
    encoder   ->  visible patches only through vit-cifar10's patch-embed +
                  pre-LN block stack (copied in by hand; this project must
                  not import another pipeline folder's code), no CLS token,
                  no classification head
    decoder   ->  a separate lightweight transformer (--decoder-dim 192,
                  --decoder-depth 2): encoded visible tokens + a learned
                  shared mask token, unshuffled back to full order, full
                  positional embeddings, final norm + Linear to patch pixels
    loss      ->  MSE on the masked patches only, targets normalized per
                  patch (subtract patch mean, divide patch std - the MAE
                  trick that makes the pixels learnable at all)

Default sizing (patch 2, dim 384, depth 6, heads 6, decoder dim 192 /
depth 2) is ~11.5M params - the same encoder stack as training/vit-cifar10
at a denser patch grid, fitting a single RTX 3090 with room to spare
(~15-30 min for 60 epochs fp32). The decoder exists only for pretraining:
the linear probe discards it and reads frozen encoder features.

Augmentation is plain torch ops, applied per batch in the training loop:
random horizontal flip, 4px zero-pad + random crop back to 32x32. Pixels
stay raw [0,1] - no per-channel normalization, because the reconstruction
targets ARE the pixels and normalization would move them (the probe is
where inputs get normalized). `--no-augment` turns flip+crop off for a
clean A/B. `--no-patch-norm` turns off the per-patch target normalization
for a clean A/B of the MAE trick.

Loss reported on the train split is the masked-patch MSE. A 10% holdout of
the training split is used for validation / best-checkpoint selection with
a deterministic full-image reconstruction loss (all patches, patch-
normalized targets, no masking) so the curve is comparable across epochs;
the official 10k test split stays fully unseen until linear_probe.py.
`--sample-every` epochs the trainer writes a recon grid PNG (original /
masked / reconstruction rows for 8 fixed validation images, hand-written
zlib PNG - no Pillow).

Usage:
    uv run --directory training/mae-cifar100 python train_mae.py \
        --data-path data/cifar100.npz \
        --num-epochs 60 \
        --batch-size 128 \
        --output-dir runs/mae_cifar100
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from png_utils import tensor_to_uint8, upscale, write_png

IMAGE_SIZE = 32


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention written out by hand: one QKV
    projection, reshaped to (heads, head_dim), softmax attention, output
    projection. No nn.MultiheadAttention - the math stays visible. Same
    module as training/vit-cifar10's, copied in by hand."""

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
    """Pre-LN (norm-first) transformer block - the stable-from-scratch
    ViT layout, same as training/vit-cifar10's."""

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


class MaskedAutoencoderViT(nn.Module):
    """The MAE of He et al. 2022 at 32x32. The encoder is vit-cifar10's
    patch-embed/block stack minus the CLS token and head (MAE uses neither:
    no CLS during pretraining, and the probe mean-pools patch tokens). The
    decoder is a separate, smaller transformer used only for pretraining.
    Positional embeddings and the mask token are initialized with the
    paper's truncated normal 0.02 (unlike vit-cifar10, whose pos embed is
    left at zero init)."""

    def __init__(self, patch_size=2, dim=384, depth=6, heads=6,
                 mlp_ratio=4.0, dropout=0.1, decoder_dim=192,
                 decoder_depth=2, decoder_heads=3, mask_ratio=0.75):
        super().__init__()
        n_patches = (IMAGE_SIZE // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # encoder (vit-cifar10's stack, no CLS, no head)
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # decoder (pretraining only)
        self.decoder_embed = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches, decoder_dim)
        )
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio, 0.0)
             for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)

        self.apply(self._init_weights)
        # paper init for the learned embeddings / mask token (truncated
        # normal 0.02, the standard ViT init)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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

    def patchify(self, images):
        """(B, 3, 32, 32) -> (B, N, p*p*3): each patch's pixels, flattened."""
        patches = F.unfold(images, kernel_size=self.patch_size,
                           stride=self.patch_size)
        return patches.transpose(1, 2)

    def unpatchify(self, patches):
        """(B, N, p*p*3) -> (B, 3, 32, 32): the inverse of patchify."""
        out = F.fold(patches.transpose(1, 2),
                     output_size=(IMAGE_SIZE, IMAGE_SIZE),
                     kernel_size=self.patch_size, stride=self.patch_size)
        return out

    def random_masking(self, x, mask_ratio):
        """Per-sample fixed-count random masking (the paper's scheme, not a
        per-patch Bernoulli): each sample gets its own permutation of patch
        positions and keeps the first n_keep. Returns the visible tokens,
        a (B, N) mask with 1 at masked positions, and ids_restore, the
        inverse permutation that unshuffles decoder output back to image
        order."""
        B, N, D = x.shape
        n_keep = int(round((1.0 - mask_ratio) * N))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)  # ascending: first n_keep are kept
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(B, n_keep, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Patch embed + pos embed, then mask and run the visible tokens
        through the block stack. Returns (latent, mask, ids_restore)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        for block in self.blocks:
            x = block(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Visible latent -> decoder embed + mask tokens, unshuffled to full
        order, full positional embeddings, decoder blocks, pred head -> per-
        patch pixels (B, N, p*p*3)."""
        B, n_keep, D = x.shape
        N = ids_restore.shape[1]
        n_masked = N - n_keep
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(B, n_masked, 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1,
                         ids_restore.unsqueeze(-1).expand(B, N, x.shape[2]))
        x = x + self.decoder_pos_embed
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward_loss(self, images, pred, mask, patch_norm=True):
        """MSE on the masked patches only. Targets are the image's own
        patches, normalized per patch (zero mean / unit std) when
        patch_norm - the MAE trick that keeps the decoder from collapsing
        to predicting patch means. mask: 1 at masked positions."""
        target = self.patchify(images)  # (B, N, p*p*3)
        if patch_norm:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, unbiased=False, keepdim=True)
            target = (target - mean) / (var.sqrt() + 1e-6)
        loss = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        return (loss * mask).sum() / mask.sum()

    def forward(self, images, mask_ratio=None, patch_norm=True):
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        latent, mask, ids_restore = self.forward_encoder(images, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(images, pred, mask, patch_norm)
        return loss, pred, mask, ids_restore

    def forward_features(self, x):
        """Frozen-feature extractor for the linear probe: all patches (no
        masking), full block stack, final norm, then mean-pool the patch
        tokens. No CLS token exists in MAE, so mean pooling is the probe
        representation (the paper's ViT-probe convention)."""
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x).mean(dim=1)  # (B, dim)


def augment_batch(x, pad=4):
    """Random horizontal flip + 4px zero-pad + random crop back to 32x32 -
    all plain torch ops, applied per batch. Pixels stay raw [0,1]: the
    reconstruction targets ARE the pixels, so there is no per-channel
    normalization here (the probe is where inputs get normalized).
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
    return out


def iterate_batches(X, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - one permutation
    per epoch, sliced into batches, so the batching logic stays visible."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx]


def run_epoch(model, X, batch_size, device, rng, optimizer=None, augment=False,
              patch_norm=True):
    """One pass over the images. If optimizer is given, trains (shuffled,
    grad updates, flip+crop augmentation, masked-MSE loss); otherwise
    evaluates deterministically: full-image reconstruction (all patches,
    no masking) with per-patch-normalized targets, so the val curve is
    comparable across epochs. Returns the mean loss."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb in iterate_batches(X, batch_size, rng, shuffle=training):
            x = torch.from_numpy(xb).to(device)
            x = augment_batch(x) if (training and augment) else x
            if training:
                loss, _, _, _ = model(x, patch_norm=patch_norm)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # deterministic full-image reconstruction loss
                _, pred, _, _ = model(x, mask_ratio=0.0, patch_norm=patch_norm)
                target = model.patchify(x)
                if patch_norm:
                    mean = target.mean(dim=-1, keepdim=True)
                    var = target.var(dim=-1, unbiased=False, keepdim=True)
                    target = (target - mean) / (var.sqrt() + 1e-6)
                loss = ((pred - target) ** 2).mean()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def make_recon_grid(model, images, patch_norm=True):
    """Build the three rows of a reconstruction grid from a batch of
    [0,1] images on device: original, masked input (masked patches shown
    as mid-gray), and the model's reconstruction (de-normalized back out of
    the per-patch normalized space pred lives in). Returns
    (originals, masked_imgs, recon_imgs) as float [0,1] tensors on CPU."""
    with torch.no_grad():
        _, pred, mask, _ = model(images, patch_norm=patch_norm)
        patches = model.patchify(images)
        masked = torch.where(
            mask.unsqueeze(-1) > 0.5,
            torch.full_like(patches, 0.5),
            patches,
        )
        masked_img = model.unpatchify(masked)
        if patch_norm:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, unbiased=False, keepdim=True)
            pred = pred * (var.sqrt() + 1e-6) + mean
        recon_img = model.unpatchify(pred)
    return (images.cpu(), masked_img.cpu(), recon_img.cpu())


def write_recon_grid(path, originals, masked_imgs, recon_imgs, cols=8,
                     upscale_factor=4, border=2):
    """Stack original / masked / reconstruction rows into one RGB PNG (8
    images per row, 3 rows), each cell upscaled with a thin gray border."""
    cells = min(cols, originals.shape[0])
    cell_img = upscale_factor * IMAGE_SIZE
    cell = cell_img + 2 * border
    rows = 3
    canvas = np.zeros((rows * cell, cells * cell, 3), dtype=np.uint8)
    gray = np.full((cell, cell, 3), 128, dtype=np.uint8)
    for r, group in enumerate([originals, masked_imgs, recon_imgs]):
        for c in range(cells):
            img = tensor_to_uint8(group[c])
            img = upscale(img, upscale_factor)
            framed = gray.copy()
            framed[border:border + cell_img, border:border + cell_img] = img
            y0, x0 = r * cell, c * cell
            canvas[y0:y0 + cell, x0:x0 + cell] = framed
    write_png(path, canvas)


def make_lr_lambda(warmup_epochs, total_epochs, eta_min_ratio=0.01):
    """Hand-written linear-warmup-then-cosine LR multiplier: ramps 0 -> 1
    over the warmup epochs, then cosine-anneals down to eta_min_ratio. The
    same schedule training/vit-cifar10 uses - warmup is the part ViTs need
    that the rest of training/'s plain-cosine trainers don't."""

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
    parser.add_argument("--data-path", default="data/cifar100.npz")
    parser.add_argument("--output-dir", default="runs/mae_cifar100")
    parser.add_argument("--patch-size", type=int, default=2,
                        help="Side length of each patch (32x32 / patch^2 "
                             "patches). Default 2 -> 256 patches (64 "
                             "visible at 75% masking), a denser grid than "
                             "vit-cifar10's patch-4; --patch-size 4 gives "
                             "the literal 64-patch sibling config.")
    parser.add_argument("--dim", type=int, default=384,
                        help="Encoder embedding dimension (vit-cifar10's).")
    parser.add_argument("--depth", type=int, default=6,
                        help="Encoder transformer blocks (vit-cifar10's).")
    parser.add_argument("--heads", type=int, default=6,
                        help="Encoder attention heads (vit-cifar10's).")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="MLP hidden size as a multiple of --dim.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout inside each encoder block.")
    parser.add_argument("--decoder-dim", type=int, default=192,
                        help="Decoder embedding dimension (smaller than the "
                             "encoder - the decoder is lightweight).")
    parser.add_argument("--decoder-depth", type=int, default=2,
                        help="Decoder transformer blocks.")
    parser.add_argument("--decoder-heads", type=int, default=3,
                        help="Decoder attention heads.")
    parser.add_argument("--mask-ratio", type=float, default=0.75,
                        help="Fraction of patches masked per image (the "
                             "paper's 75%).")
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
    parser.add_argument("--no-patch-norm", action="store_true",
                        help="Disable per-patch target normalization "
                             "(A/B of the MAE trick; expects the loss to "
                             "get worse).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training rows held out for "
                             "validation (the CIFAR-100 test split stays "
                             "fully unseen).")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Write a reconstruction grid PNG every this "
                             "many epochs (0 disables).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_train_full = data["X_train"]  # (n, 3, 32, 32) float32 in [0,1]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]

    n_patches = (IMAGE_SIZE // args.patch_size) ** 2
    n_keep = int(round((1.0 - args.mask_ratio) * n_patches))
    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"patch={args.patch_size} -> {n_patches} patches "
          f"({n_keep} visible at mask {args.mask_ratio:.0%})  "
          f"dim={args.dim}  depth={args.depth}  heads={args.heads}  "
          f"decoder={args.decoder_dim}/{args.decoder_depth}  "
          f"augment={not args.no_augment}  patch_norm={not args.no_patch_norm}")

    model = MaskedAutoencoderViT(
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
    ).to(device)
    n_enc = sum(p.numel() for p in model.blocks.parameters()) + \
        sum(p.numel() for p in model.patch_embed.parameters()) + \
        sum(p.numel() for p in model.norm.parameters()) + \
        model.pos_embed.numel()
    n_dec = sum(p.numel() for p in model.parameters()) - n_enc
    print(f"Encoder parameters: {n_enc:,}  Decoder parameters: {n_dec:,}  "
          f"Total: {n_enc + n_dec:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(args.warmup_epochs, args.num_epochs),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "mae_best.pt")
    final_path = os.path.join(args.output_dir, "mae_final.pt")
    ckpt_meta = {
        "arch": "mae_cifar100",
        "image_size": IMAGE_SIZE,
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "decoder_dim": args.decoder_dim,
        "decoder_depth": args.decoder_depth,
        "decoder_heads": args.decoder_heads,
        "mask_ratio": args.mask_ratio,
    }

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(
            model, X_train, args.batch_size, device, rng,
            optimizer=optimizer, augment=not args.no_augment,
            patch_norm=not args.no_patch_norm,
        )
        scheduler.step()

        val_loss = run_epoch(
            model, X_val, args.batch_size, device, rng, optimizer=None,
            patch_norm=not args.no_patch_norm,
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            elapsed = time.time() - start_time
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: masked-mse={train_loss:.5f}  "
                f"val: recon-mse={val_loss:.5f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.0f}s elapsed)"
            )

        if args.sample_every and (epoch % args.sample_every == 0
                                  or epoch == args.num_epochs):
            originals, masked_imgs, recon_imgs = make_recon_grid(
                model, torch.from_numpy(X_val[:8]).to(device),
                patch_norm=not args.no_patch_norm,
            )
            grid_path = os.path.join(
                args.output_dir, f"recon_epoch{epoch:04d}.png"
            )
            write_recon_grid(grid_path, originals, masked_imgs, recon_imgs)
            print(f"Wrote {grid_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(),
                 "epoch": epoch, "val_loss": val_loss,
                 **ckpt_meta},
                best_path,
            )

    torch.save(
        {"model_state": model.state_dict(),
         "epoch": args.num_epochs, "val_loss": val_loss,
         **ckpt_meta},
        final_path,
    )
    print(f"Saved best checkpoint (val recon-mse={best_val_loss:.5f}) "
          f"to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run linear_probe.py against the held-out test split next.")


if __name__ == "__main__":
    main()

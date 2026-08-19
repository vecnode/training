"""
A VQ-VAE ([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937))
trained **from scratch** on CIFAR-10 - no VQ-VAE library. The encoder,
decoder, vector quantizer's nearest-neighbor lookup, straight-through
gradient estimator, and EMA codebook updates are all written out by hand
as plain torch modules; torch is used only for tensor ops, autograd, and
GPU execution, the same role it plays everywhere else in this repo.

This pipeline is the successor of `training/cifar10-vae` (converted in
place): a plain VAE's blur is caused by Gaussian-posterior averaging in
the ELBO objective, which a discrete codebook + straight-through estimator
removes - reconstruction is driven purely by MSE recon + a commitment
term, so the decoder is free to reproduce edges and texture instead of a
mean image.

The model (32x32 RGB in, 32x32 RGB out):
    Encoder: Conv(3->64,s2) -> Conv(64->128,s2) -> Conv(128->256,3x3)
             -> Conv(256->D,1x1)                    (32x32 -> 8x8 grid of D-dim vectors)
    Vector quantizer: replace each of the 8x8 D-dim vectors with its
             nearest neighbor in a learned K x D codebook (K=512, D=64 by
             default); straight-through gradient; EMA codebook updates
             (no gradient flows into the codebook - the entries are the
             exponentially-weighted means of the encoder outputs that
             selected them, Kingma-style)
    Decoder: ConvT(D->128,s2) -> ConvT(128->64,s2) -> Conv(64->3,3x3) -> sigmoid

The loss (all computed by hand, no library):
    L = MSE reconstruction(x, x_hat)
      + --commitment-beta * ||z_e - sg[z_q]||^2        (commitment term)
  - with EMA codebook updates there is no separate codebook loss term:
    the codebook entries are moving averages, not gradient-updated.
  - training logs codebook perplexity each epoch (usage / collapse check):
    a healthy codebook uses most of its K codes (perplexity toward K), a
    collapsed one only ever uses a handful (perplexity near 1).

Reconstruction only - no prior over the discrete codes, so the pipeline
can encode a real image, quantize it, and decode it back, but it cannot
sample novel images the way a continuous-latent VAE can sample from
N(0,I). That is deliberate: the discrete code grid is the substrate for a
later learned prior (the planned cascade), and adding one means training
a separate autoregressive/diffusion model over the code indices.

Usage:
    uv run --directory training/cifar10-vqvae python train_vqvae.py \
        --data-path data/cifar10.npz \
        --codebook-size 512 \
        --embedding-dim 64 \
        --commitment-beta 0.25 \
        --num-epochs 100 \
        --batch-size 128 \
        --output-dir runs/cifar10_vqvae
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial code grid: the encoder downsamples 32x32 twice (32 -> 16 -> 8),
# so the quantizer sees an 8x8 grid of D-dim vectors.
GRID_H = GRID_W = 8


class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)     # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 8x8 -> 8x8
        self.conv4 = nn.Conv2d(256, embedding_dim, kernel_size=1)             # 8x8 -> 8x8, D channels

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        return self.conv4(h)


class VectorQuantizer(nn.Module):
    """Nearest-neighbor lookup into a learned K x D codebook with a
    straight-through gradient estimator and exponential-moving-average
    (EMA) codebook updates. EMA updates are the standard modern fix for
    codebook collapse: entries are pushed toward the encoder outputs that
    selected them without any gradient flowing into the codebook."""

    def __init__(self, num_embeddings, embedding_dim, commitment_cost,
                 ema_decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_sum", torch.zeros(num_embeddings, embedding_dim))

    def forward(self, z_e):
        # z_e: (B, D, H, W) -> (B, H, W, D), then flat (N, D)
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.reshape(-1, self.embedding_dim)

        # squared-L2 distance to every codebook vector
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2.0 * flat @ self.embedding.weight.t()
        )
        encoding_indices = dist.argmin(1)
        z_q = self.embedding(encoding_indices).view(z_e.shape)

        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        if self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                embedding_sum = encodings.t() @ flat
                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1.0 - self.ema_decay
                )
                self.ema_embedding_sum.mul_(self.ema_decay).add_(
                    embedding_sum, alpha=1.0 - self.ema_decay
                )
                # sonnet-style normalization: density (cluster+eps)/(n+K*eps)
                # re-scaled by n, so dividing by it yields the per-code mean
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )
                self.embedding.weight.data.copy_(
                    self.ema_embedding_sum / smoothed.unsqueeze(1)
                )

        # perplexity from this batch's actual code usage (1 = collapsed,
        # K = perfectly uniform)
        p = encodings.mean(0)
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())

        # straight-through estimator: forward uses z_q, backward flows
        # through z_e unchanged
        z_q_st = z_e + (z_q - z_e).detach()
        # commitment = ||z_e - sg[z_q]||^2: gradient flows into z_e, pulling
        # encoder outputs toward their chosen codebook entries
        commitment_loss = F.mse_loss(z_e, z_q.detach())

        return z_q_st.permute(0, 3, 1, 2), commitment_loss, perplexity


class Decoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)             # 16x16 -> 32x32
        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z_q):
        h = F.relu(self.deconv1(z_q))
        h = F.relu(self.deconv2(h))
        return torch.sigmoid(self.out(h))


class VQVAE(nn.Module):
    def __init__(self, codebook_size=512, embedding_dim=64,
                 commitment_beta=0.25, ema_decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.encoder = Encoder(embedding_dim)
        self.quantizer = VectorQuantizer(
            codebook_size, embedding_dim, commitment_beta, ema_decay
        )
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, commitment, perplexity = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, commitment, perplexity

    def encode_decode(self, x):
        """Deterministic reconstruction: encode -> quantize -> decode."""
        z_e = self.encoder(x)
        z_q, _, _ = self.quantizer(z_e)
        return self.decoder(z_q)


def vqvae_loss(x_hat, x, commitment, commitment_beta):
    recon = F.mse_loss(x_hat, x)  # mean over pixels, standard for VQ-VAE
    return recon + commitment_beta * commitment, recon, commitment


def iterate_batches(X, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - a single
    permutation per epoch, sliced into batches, so the batching logic stays
    visible rather than delegated to a library abstraction."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx]


def run_epoch(model, X, batch_size, device, rng, optimizer=None, augment=False):
    """One pass over X. If optimizer is given, trains (shuffled, grad
    updates, optional horizontal flips); otherwise evaluates (no shuffle,
    no grad, no augmentation, no EMA codebook update) - shared loop so
    train/val accounting can't drift apart."""
    training = optimizer is not None
    model.train(training)

    total_loss = total_recon = total_commit = 0.0
    total_ppl = 0.0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterate_batches(X, batch_size, rng, shuffle=training):
            x = torch.from_numpy(batch).to(device)  # (B, 3, 32, 32), CHW in [0,1]

            if training and augment:
                flip = torch.rand(x.shape[0], device=device) < 0.5
                if flip.any():
                    x[flip] = torch.flip(x[flip], dims=[3])

            x_hat, commitment, perplexity = model(x)
            loss, recon, commit = vqvae_loss(
                x_hat, x, commitment, model.quantizer.commitment_cost
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_commit += commit.item()
            total_ppl += perplexity.item()
            n_batches += 1

    return (total_loss / n_batches, total_recon / n_batches,
            total_commit / n_batches, total_ppl / n_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar10.npz")
    parser.add_argument("--output-dir", default="runs/cifar10_vqvae")
    parser.add_argument("--codebook-size", type=int, default=512,
                         help="Number of discrete codes K.")
    parser.add_argument("--embedding-dim", type=int, default=64,
                         help="Codebook vector dimension D (also the encoder "
                              "output / decoder input channels).")
    parser.add_argument("--commitment-beta", type=float, default=0.25,
                         help="Weight on the commitment term ||z_e - sg[z_q]||^2.")
    parser.add_argument("--codebook-ema-decay", type=float, default=0.99,
                         help="EMA decay for codebook entry updates.")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-augment", action="store_true",
                         help="Disable random horizontal-flip augmentation on training batches.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of training rows held out for validation "
                              "(the CIFAR-10 test split stays fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train_full = data["X_train"]  # (n, 3, 32, 32) float32 in [0,1]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"codebook={args.codebook_size}x{args.embedding_dim}  "
          f"commitment_beta={args.commitment_beta}  ema_decay={args.codebook_ema_decay}  "
          f"augment={not args.no_augment}")

    model = VQVAE(codebook_size=args.codebook_size,
                  embedding_dim=args.embedding_dim,
                  commitment_beta=args.commitment_beta,
                  ema_decay=args.codebook_ema_decay).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (codebook {args.codebook_size}x{args.embedding_dim} "
          f"= {args.codebook_size * args.embedding_dim:,} of them)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_recon = float("inf")
    best_ppl = 0.0
    best_path = os.path.join(args.output_dir, "vqvae_best.pt")
    final_path = os.path.join(args.output_dir, "vqvae_final.pt")

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_recon, train_commit, train_ppl = run_epoch(
            model, X_train, args.batch_size, device, rng,
            optimizer=optimizer, augment=not args.no_augment,
        )
        scheduler.step()

        val_loss, val_recon, val_commit, val_ppl = run_epoch(
            model, X_val, args.batch_size, device, rng, optimizer=None
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: loss={train_loss:8.4f} recon={train_recon:8.4f} "
                f"commit={train_commit:7.4f} ppl={train_ppl:6.1f}  "
                f"val: loss={val_loss:8.4f} recon={val_recon:8.4f} "
                f"ppl={val_ppl:6.1f}  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_recon < best_val_recon:
            best_val_recon = val_recon
            best_ppl = val_ppl
            torch.save(
                {"model_state": model.state_dict(),
                 "codebook_size": args.codebook_size,
                 "embedding_dim": args.embedding_dim,
                 "commitment_beta": args.commitment_beta,
                 "arch": "vqvae", "epoch": epoch, "val_recon": val_recon,
                 "perplexity": val_ppl},
                best_path,
            )

    torch.save(
        {"model_state": model.state_dict(),
         "codebook_size": args.codebook_size,
         "embedding_dim": args.embedding_dim,
         "commitment_beta": args.commitment_beta,
         "arch": "vqvae", "epoch": args.num_epochs, "val_recon": val_recon,
         "perplexity": val_ppl},
        final_path,
    )
    print(f"Saved best checkpoint (val_recon={best_val_recon:.4f}, "
          f"ppl={best_ppl:.1f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_vqvae.py against the held-out test split next.")


if __name__ == "__main__":
    main()

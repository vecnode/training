"""
A small convolutional VAE, trained from scratch on MNIST - no VAE library
(no pythae, no lightning-bolts). The encoder, decoder, reparameterization
trick, and ELBO loss are all written out below as plain torch modules /
functions; torch is used only for tensor ops, autograd, and GPU execution
(the same role it plays in fine-tuning/*-lora), not for the model itself.

The model:
    Encoder: Conv(1->32,s2) -> Conv(32->64,s2) -> flatten -> fc_mu, fc_logvar
    z = mu + eps * exp(0.5 * logvar),  eps ~ N(0, I)          [reparameterization trick]
    Decoder: fc(z) -> reshape -> ConvT(64->32,s2) -> ConvT(32->1,s2) -> sigmoid

The loss (negative ELBO, minimized):
    L = reconstruction_loss(x, x_hat)  +  beta * KL(q(z|x) || N(0, I))
  - reconstruction_loss: pixel-wise binary cross-entropy (inputs are in
    [0,1], read as independent Bernoulli pixel probabilities), summed over
    pixels, averaged over the batch.
  - KL divergence between the encoder's Gaussian q(z|x) and the standard
    normal prior N(0, I) has a closed form for diagonal Gaussians:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    (derived from the Gaussian KL formula; no library computes this here).
  - beta (default 1.0) is the standard beta-VAE weight on the KL term -
    beta > 1 trades reconstruction sharpness for a more disentangled/
    regularized latent space, beta < 1 does the opposite.

Usage:
    uv run --directory training/mnist-vae python train_vae.py \
        --data-path data/mnist.npz \
        --latent-dim 32 \
        --beta 1.0 \
        --num-epochs 30 \
        --batch-size 128 \
        --output-dir runs/mnist_vae
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)   # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14x14 -> 7x7
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)   # 14x14 -> 28x28

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.deconv1(h))
        return torch.sigmoid(self.deconv2(h))


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


def vae_loss(x_hat, x, mu, logvar, beta):
    """Negative ELBO = reconstruction term + beta * KL term, both averaged
    over the batch (summed over pixels/latent dims within each example
    first, matching the standard VAE objective)."""
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon + beta * kl, recon, kl


def iterate_batches(X, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - a single
    permutation per epoch, sliced into batches, so the batching logic stays
    visible rather than delegated to a library abstraction."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx]


def run_epoch(model, X, batch_size, beta, device, rng, optimizer=None):
    """One pass over X. If optimizer is given, trains (shuffled, grad
    updates); otherwise evaluates (no shuffle, no grad) - shared loop so
    train/val accounting can't drift apart."""
    training = optimizer is not None
    model.train(training)

    total_loss = total_recon = total_kl = 0.0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterate_batches(X, batch_size, rng, shuffle=training):
            x = torch.from_numpy(batch).unsqueeze(1).to(device)  # (B, 1, 28, 28)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--output-dir", default="runs/mnist_vae")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0,
                         help="Weight on the KL term (beta-VAE); 1.0 = standard VAE ELBO.")
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of training rows held out for validation "
                              "(the MNIST test split stays fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train_full = data["X_train"]  # (n, 28, 28) float32 in [0,1]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"latent_dim={args.latent_dim}  beta={args.beta}")

    model = VAE(args.latent_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "vae_best.pt")
    final_path = os.path.join(args.output_dir, "vae_final.pt")

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_recon, train_kl = run_epoch(
            model, X_train, args.batch_size, args.beta, device, rng, optimizer=optimizer
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, X_val, args.batch_size, args.beta, device, rng, optimizer=None
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: loss={train_loss:8.2f} recon={train_recon:8.2f} kl={train_kl:7.2f}  "
                f"val: loss={val_loss:8.2f} recon={val_recon:8.2f} kl={val_kl:7.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state": model.state_dict(), "latent_dim": args.latent_dim,
                 "epoch": epoch, "val_loss": val_loss},
                best_path,
            )

    torch.save(
        {"model_state": model.state_dict(), "latent_dim": args.latent_dim,
         "epoch": args.num_epochs, "val_loss": val_loss},
        final_path,
    )
    print(f"Saved best checkpoint (val_loss={best_val_loss:.2f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_vae.py against the held-out test split next.")


if __name__ == "__main__":
    main()

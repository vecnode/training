"""
A DCGAN (Radford et al. 2015), trained from scratch on Fashion-MNIST - no
GAN library (no torchvision, no kagglehub, no pytorch-gan-metrics). The
generator, discriminator, the N(0, 0.02) weight init, one-sided label
smoothing, and the balanced D/G update loop are all written out below as
plain torch modules / functions; torch is used only for tensor ops,
autograd, and GPU execution (the same role it plays in training/mnist-vae
and fine-tuning/*-lora), not for the model itself.

The model (28x28 input - note that 28 does NOT divide cleanly down DCGAN's
canonical 32x32 ladder, so the verified shapes below differ from the
paper's; see the README):
    Generator:  z ~ N(0,I) (100-dim) -> Linear -> 7x7x256 -> BN+ReLU
                -> ConvT(4x4,s2) -> 14x14x128 -> BN+ReLU
                -> ConvT(4x4,s2) -> 28x28x64  -> BN+ReLU
                -> ConvT(3x3,s1) -> 28x28x1   -> Tanh
    Discriminator: Conv(4x4,s2) 1->64   -> 14x14x64  -> LeakyReLU(0.2)   [no BN on input]
                   Conv(4x4,s2) 64->128 -> 7x7x128   -> BN+LeakyReLU(0.2)
                   Conv(4x4,s2) 128->256 -> 3x3x256  -> BN+LeakyReLU(0.2)
                   Conv(3x3,s1) 256->1   -> 1x1x1 logit (BCEWithLogits)

Training (the DCGAN tuning that makes the two-player game converge):
  - binary cross-entropy with logits + one-sided label smoothing
    (real label = 1 - --label-smoothing, default 0.9);
  - Adam lr 2e-4, betas (0.5, 0.999) on both nets (DCGAN's values);
  - one D step and one G step per batch (balanced k=1), the same fake batch
    reused (detached for D's step);
  - all weights initialized N(0, 0.02) by hand (init_weights below).

A GAN is judged by its samples, not its loss: D/G losses move adversarially
and say little about quality, so this script saves a fixed-z sample grid
every --sample-every epochs (you can watch the generator learn - or
collapse - across training). evaluate_dcgan.py does the real judging.

Usage:
    uv run --directory training/fashion-mnist-dcgan python train_dcgan.py \
        --data-path data/fashion_mnist.npz \
        --num-epochs 50 \
        --batch-size 128 \
        --output-dir runs/fashion_mnist_dcgan
"""

import argparse
import os
import struct
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """z (B, z_dim) -> (B, 1, 28, 28) in [-1, 1] via Tanh. Starts from a
    7x7 grid, not DCGAN's 4x4: 28 = 7 * 2 * 2, so two stride-2 deconvs get
    us to 28x28 and a final 3x3 conv keeps it there."""

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 256 * 7 * 7)
        self.bn0 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 14x14 -> 28x28
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)     # 28x28 -> 28x28
        self.apply(init_weights)

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 7, 7)
        h = F.relu(self.bn0(h))
        h = F.relu(self.bn1(self.deconv1(h)))
        h = F.relu(self.bn2(self.deconv2(h)))
        return torch.tanh(self.deconv3(h))


class Discriminator(nn.Module):
    """(B, 1, 28, 28) -> (B,) raw logits (BCEWithLogitsLoss; no sigmoid
    here). Three stride-2 convs take 28 -> 14 -> 7 -> 3, so the last
    feature map is 3x3, not DCGAN's 4x4, and a final 3x3 conv reduces it to
    a single logit. No BatchNorm on the input layer, per DCGAN."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)    # 28 -> 14
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 14 -> 7
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 7 -> 3
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)    # 3 -> 1
        self.apply(init_weights)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)
        return self.conv4(h).view(-1)


def init_weights(m):
    """DCGAN's hand-written weight init: every weight ~ N(0, 0.02), every
    bias 0 (batch-norm weight also N(0, 0.02), its bias 0). Applied
    recursively via nn.Module.apply() from both nets' __init__."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.zeros_(m.bias)


def write_png(path, array_2d_uint8):
    """Write a single-channel 8-bit grayscale PNG from a (H, W) uint8
    array. Minimal encoder: IHDR + one zlib-compressed IDAT (filter byte 0
    / "None" prepended to every scanline) + IEND, each with its CRC32 - the
    same hand-written writer as training/mnist-vae / flow-matching-mnist.
    """
    height, width = array_2d_uint8.shape

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    raw = b"".join(
        b"\x00" + array_2d_uint8[y].tobytes() for y in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)


def make_grid(images, grid_cols, border=2):
    """Tile a list of (28, 28) float arrays in [0,1] into one uint8 image
    grid, `border` pixels of black padding between tiles."""
    n = len(images)
    image_size = images[0].shape[0]
    grid_rows = (n + grid_cols - 1) // grid_cols
    tile = image_size + border

    canvas = np.zeros((grid_rows * tile, grid_cols * tile), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, grid_cols)
        top, left = r * tile, c * tile
        canvas[top:top + image_size, left:left + image_size] = (
            np.clip(img, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return canvas


def iterate_batches(X, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - a single
    permutation per epoch, sliced into batches, so the batching logic stays
    visible rather than delegated to a library abstraction."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/fashion_mnist.npz")
    parser.add_argument("--output-dir", default="runs/fashion_mnist_dcgan")
    parser.add_argument("--z-dim", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="DCGAN's tuned Adam LR, used for BOTH nets.")
    parser.add_argument("--label-smoothing", type=float, default=0.9,
                        help="One-sided label smoothing: real images get this "
                             "label (default 0.9), fakes get 0.0.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Save a fixed-z sample grid PNG every N epochs "
                             "(watch the generator learn/collapse).")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save a checkpoint every N epochs (plus a final one).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train = data["X_train"]  # (n, 28, 28) float32 in [0,1]
    print(f"Train rows: {X_train.shape[0]}  z_dim={args.z_dim}  "
          f"batch_size={args.batch_size}  label_smoothing={args.label_smoothing}")

    generator = Generator(args.z_dim).to(device)
    discriminator = Discriminator().to(device)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    g_opt = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    bce = torch.nn.BCEWithLogitsLoss()
    real_target = 1.0 - args.label_smoothing
    fake_target = 0.0

    os.makedirs(args.output_dir, exist_ok=True)

    # Fixed z for the sample grid: seeded once, so every grid is drawn from
    # the same latent points and is directly comparable across epochs.
    grid_rng = torch.Generator(device=device).manual_seed(args.seed)
    z_grid = torch.randn(100, args.z_dim, generator=grid_rng, device=device)

    for epoch in range(1, args.num_epochs + 1):
        d_loss_sum = g_loss_sum = 0.0
        d_real_sum = d_fake_sum = 0.0
        n_batches = 0
        generator.train()
        discriminator.train()

        for batch in iterate_batches(X_train, args.batch_size, rng, shuffle=True):
            x = torch.from_numpy(batch * 2.0 - 1.0).unsqueeze(1).to(device)  # (B,1,28,28) in [-1,1]
            b = x.shape[0]
            z = torch.randn(b, args.z_dim, device=device)
            fake = generator(z)  # keeps grad; D's step must not update G

            # --- Discriminator step: real (label-smoothed) + fake (detached) ---
            d_opt.zero_grad()
            d_real_logits = discriminator(x)
            loss_real = bce(d_real_logits, torch.full((b,), real_target, device=device))
            d_fake_logits = discriminator(fake.detach())
            loss_fake = bce(d_fake_logits, torch.full((b,), fake_target, device=device))
            d_loss = 0.5 * (loss_real + loss_fake)
            d_loss.backward()
            d_opt.step()

            # --- Generator step: fool the just-updated discriminator ---
            g_opt.zero_grad()
            g_logits = discriminator(fake)  # same fake batch, still has grad
            g_loss = bce(g_logits, torch.ones(b, device=device))
            g_loss.backward()
            g_opt.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            d_real_sum += d_real_logits.mean().item()
            d_fake_sum += d_fake_logits.mean().item()
            n_batches += 1

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"D_loss={d_loss_sum / n_batches:7.4f}  "
                f"G_loss={g_loss_sum / n_batches:7.4f}  "
                f"D(x)={d_real_sum / n_batches:.3f}  "
                f"D(G(z))={d_fake_sum / n_batches:.3f}"
            )

        if epoch % args.sample_every == 0 or epoch == args.num_epochs:
            generator.eval()
            with torch.no_grad():
                grid01 = ((generator(z_grid) + 1.0) / 2.0).squeeze(1).cpu().numpy()  # [-1,1] -> [0,1]
            grid_path = os.path.join(args.output_dir, f"samples_epoch{epoch:04d}.png")
            write_png(grid_path, make_grid(list(grid01), grid_cols=10))
            print(f"Saved fixed-z sample grid to {grid_path}")

        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            ckpt_path = os.path.join(args.output_dir, f"dcgan_epoch{epoch:04d}.pt")
            torch.save(
                {"generator_state": generator.state_dict(),
                 "discriminator_state": discriminator.state_dict(),
                 "z_dim": args.z_dim,
                 "epoch": epoch,
                 "d_loss": d_loss_sum / n_batches,
                 "g_loss": g_loss_sum / n_batches},
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")
            if epoch == args.num_epochs:
                # Stable name for the last checkpoint - the documented default
                # for evaluate_dcgan.py (e.g. runs/.../dcgan_final.pt).
                final_path = os.path.join(args.output_dir, "dcgan_final.pt")
                torch.save(
                    {"generator_state": generator.state_dict(),
                     "discriminator_state": discriminator.state_dict(),
                     "z_dim": args.z_dim,
                     "epoch": epoch,
                     "d_loss": d_loss_sum / n_batches,
                     "g_loss": g_loss_sum / n_batches},
                    final_path,
                )
                print(f"Saved final checkpoint to {final_path}")

    print("Run evaluate_dcgan.py against the checkpoint to judge the samples.")


if __name__ == "__main__":
    main()

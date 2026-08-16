"""
Evaluate a trained VAE checkpoint against the held-out MNIST test split, and
save PNGs so reconstruction quality can actually be looked at:

  - reconstruction_grid.png - one example per digit 0-9, real image on top
    of its reconstruction, so you can see how well the model reconstructs
    each digit shape.
  - prior_samples.png - pure generations: z ~ N(0, I) decoded directly
    (no encoder involved), showing what the model has learned to generate
    from the latent prior alone.

No imaging library (no Pillow/matplotlib): PNG is written by hand with the
stdlib zlib module (deflate-compress raw 8-bit grayscale scanlines into a
single IDAT chunk) - the same "no extra dependency" spirit as the rest of
this pipeline.

Usage:
    uv run --directory training/mnist-vae python evaluate_vae.py \
        --data-path data/mnist.npz \
        --checkpoint-path runs/mnist_vae/vae_best.pt \
        --output-dir runs/mnist_vae
"""

import argparse
import os
import struct
import zlib

import numpy as np
import torch

from train_vae import VAE, vae_loss, run_epoch


def write_png(path, array_2d_uint8):
    """Write a single-channel 8-bit grayscale PNG from a (H, W) uint8
    array. Minimal encoder: IHDR + one zlib-compressed IDAT (filter byte 0
    / "None" prepended to every scanline) + IEND, each with its CRC32.
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


def reconstruction_grid(model, X, y, device, digits=range(10)):
    """One real/reconstructed pair per digit, stacked as: row 0 = 10 real
    digits, row 1 = their 10 reconstructions - laid out as a single 2-row
    grid so real vs. reconstructed is directly comparable at a glance."""
    model.eval()
    examples = []
    with torch.no_grad():
        for d in digits:
            idx = int(np.argmax(y == d))  # first test example of digit d
            x = torch.from_numpy(X[idx:idx + 1]).unsqueeze(1).to(device)
            mu, logvar = model.encoder(x)
            x_hat = model.decoder(mu)  # deterministic reconstruction (use mu, not a sample)
            examples.append(X[idx])
            examples.append(x_hat.squeeze().cpu().numpy())

    # Reorder into "all real, then all reconstructed" for a clean 2-row grid.
    reals = examples[0::2]
    recons = examples[1::2]
    return make_grid(reals + recons, grid_cols=len(digits))


def prior_sample_grid(model, latent_dim, device, num_samples, grid_cols, seed):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, generator=rng, device=device)
        samples = model.decoder(z).squeeze(1).cpu().numpy()
    return make_grid(list(samples), grid_cols=grid_cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--checkpoint-path", default="runs/mnist_vae/vae_best.pt")
    parser.add_argument("--output-dir", default="runs/mnist_vae")
    parser.add_argument("--beta", type=float, default=1.0,
                         help="Must match the beta used at training time for a "
                              "loss-value comparable to the training run's logs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-prior-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = VAE(checkpoint["latent_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.2f}, latent_dim={checkpoint['latent_dim']})")

    rng = np.random.default_rng(args.seed)
    test_loss, test_recon, test_kl = run_epoch(
        model, X_test, args.batch_size, args.beta, device, rng, optimizer=None
    )
    print(f"Test set: loss={test_loss:.2f}  recon={test_recon:.2f}  kl={test_kl:.2f}")

    os.makedirs(args.output_dir, exist_ok=True)

    recon_grid = reconstruction_grid(model, X_test, y_test, device)
    recon_path = os.path.join(args.output_dir, "reconstruction_grid.png")
    write_png(recon_path, recon_grid)
    print(f"Saved reconstruction grid (top=real, bottom=reconstructed, digits 0-9) to {recon_path}")

    grid_cols = int(np.ceil(np.sqrt(args.num_prior_samples)))
    sample_grid = prior_sample_grid(
        model, checkpoint["latent_dim"], device, args.num_prior_samples, grid_cols, args.seed
    )
    sample_path = os.path.join(args.output_dir, "prior_samples.png")
    write_png(sample_path, sample_grid)
    print(f"Saved {args.num_prior_samples} prior samples (z ~ N(0,I), no encoder involved) to {sample_path}")


if __name__ == "__main__":
    main()

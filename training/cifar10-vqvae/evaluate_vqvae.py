"""
Evaluate a trained VQ-VAE checkpoint against the held-out CIFAR-10 test
split, and save PNGs so reconstruction quality can actually be looked at:

  - reconstruction_grid.png - one example per class (airplane, automobile,
    bird, ... truck), real image on top of its reconstruction, so you can
    see how well the model reconstructs each class.

Unlike the predecessor cifar10-vae pipeline there is no prior_samples.png:
a VQ-VAE is reconstruction-only - sampling would require a separate
learned prior over the discrete code grid (the planned cascade), which is
not built here.

The metrics are the same ones used to judge the old VAE (MAE, PSNR, SSIM,
and high-frequency detail retention), all computed by hand - SSIM with a
Gaussian window, high-frequency energy with a 3x3 Laplacian - so the
VQ-VAE and the VAE numbers are directly comparable on the same test set.
No imaging library (no Pillow/matplotlib): the PNG is written by hand with
the stdlib zlib module, same as the rest of this pipeline.

Usage:
    uv run --directory training/cifar10-vqvae python evaluate_vqvae.py \
        --data-path data/cifar10.npz \
        --checkpoint-path runs/cifar10_vqvae/vqvae_best.pt \
        --output-dir runs/cifar10_vqvae
"""

import argparse
import os
import struct
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from train_vqvae import VQVAE, run_epoch


def write_png(path, array_3d_uint8):
    """Write an 8-bit RGB PNG from a (H, W, 3) uint8 array. Minimal
    encoder: IHDR (bit depth 8, color type 2 = truecolor) + one
    zlib-compressed IDAT (filter byte 0 / "None" prepended to every
    scanline) + IEND, each with its CRC32."""
    height, width, _ = array_3d_uint8.shape

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    raw = b"".join(
        b"\x00" + array_3d_uint8[y].tobytes() for y in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)


def make_grid(images, grid_cols, border=2):
    """Tile a list of (32, 32, 3) float arrays in [0,1] into one uint8 RGB
    image grid, `border` pixels of black padding between tiles."""
    n = len(images)
    image_size = images[0].shape[0]
    grid_rows = (n + grid_cols - 1) // grid_cols
    tile = image_size + border

    canvas = np.zeros((grid_rows * tile, grid_cols * tile, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, grid_cols)
        top, left = r * tile, c * tile
        canvas[top:top + image_size, left:left + image_size] = (
            np.clip(img, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return canvas


def gaussian_window(size=11, sigma=1.5):
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return g[:, None] * g[None, :]


def metrics(x, y, gauss):
    """x, y: (B, 3, 32, 32) in [0,1]. Returns per-image arrays of MAE,
    PSNR (dB), SSIM, and high-frequency energy of both."""
    err = (x - y).abs().mean(dim=(1, 2, 3))
    mse = ((x - y) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(1.0 / (mse + 1e-12))

    # SSIM with an 11x11 Gaussian window, per channel
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad = gauss.shape[0] // 2
    k = gauss[None, None].repeat(x.shape[1], 1, 1, 1)
    mu_x = F.conv2d(x, k, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, k, padding=pad, groups=x.shape[1])
    sx2 = F.conv2d(x * x, k, padding=pad, groups=x.shape[1]) - mu_x ** 2
    sy2 = F.conv2d(y * y, k, padding=pad, groups=x.shape[1]) - mu_y ** 2
    sxy = F.conv2d(x * y, k, padding=pad, groups=x.shape[1]) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sx2 + sy2 + C2)
    ssim = (num / den).mean(dim=(1, 2, 3))

    # high-frequency energy via 3x3 Laplacian, per channel
    lap = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                       dtype=torch.float32, device=x.device)
    lap = lap.repeat(x.shape[1], 1, 1, 1)
    hf_x = F.conv2d(x, lap, padding=1, groups=x.shape[1]).abs().mean(dim=(1, 2, 3))
    hf_y = F.conv2d(y, lap, padding=1, groups=x.shape[1]).abs().mean(dim=(1, 2, 3))

    return err, psnr, ssim, hf_x, hf_y


def report(name, maes, psnrs, ssims, hf_real, hf_recon):
    print(
        f"{name}: MAE={maes.mean():.4f}  PSNR={psnrs.mean():.2f} dB  "
        f"SSIM={ssims.mean():.4f}  HF kept={100 * hf_recon.mean() / hf_real.mean():.1f}%"
    )


def reconstruction_grid(model, X, y, device, num_classes=10):
    """One real/reconstructed pair per class, stacked as: row 0 = 10 real
    images, row 1 = their 10 reconstructions (encode -> quantize ->
    decode, deterministic) - directly comparable at a glance."""
    model.eval()
    reals, recons = [], []
    with torch.no_grad():
        for c in range(num_classes):
            idx = int(np.argmax(y == c))  # first test example of class c
            x = torch.from_numpy(X[idx:idx + 1]).to(device)
            x_hat = model.encode_decode(x)
            reals.append(X[idx].transpose(1, 2, 0))  # CHW -> HWC for tiling
            recons.append(x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy())
    return make_grid(reals + recons, grid_cols=num_classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar10.npz")
    parser.add_argument("--checkpoint-path", default="runs/cifar10_vqvae/vqvae_best.pt")
    parser.add_argument("--output-dir", default="runs/cifar10_vqvae")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    label_names = [str(n) for n in data["label_names"]]
    print(f"Classes: {', '.join(label_names)}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = VQVAE(codebook_size=checkpoint["codebook_size"],
                  embedding_dim=checkpoint["embedding_dim"],
                  commitment_beta=checkpoint["commitment_beta"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_recon={checkpoint['val_recon']:.4f}, "
          f"codebook={checkpoint['codebook_size']}x{checkpoint['embedding_dim']})")

    rng = np.random.default_rng(args.seed)
    val_loss, val_recon, val_commit, val_ppl = run_epoch(
        model, X_test, args.batch_size, device, rng, optimizer=None
    )
    print(f"Test set: loss={val_loss:.4f}  recon={val_recon:.4f}  "
          f"commit={val_commit:.4f}  perplexity={val_ppl:.1f}")

    # metric suite over the full test set
    gauss = gaussian_window().to(device)
    all_maes, all_psnrs, all_ssims, all_hf_real, all_hf_recon = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, X_test.shape[0], args.batch_size):
            x = torch.from_numpy(X_test[i:i + args.batch_size]).to(device)
            x_hat = model.encode_decode(x)
            mae, psnr, ssim, hf_real, hf_recon = metrics(x, x_hat, gauss)
            all_maes.append(mae.cpu().numpy())
            all_psnrs.append(psnr.cpu().numpy())
            all_ssims.append(ssim.cpu().numpy())
            all_hf_real.append(hf_real.cpu().numpy())
            all_hf_recon.append(hf_recon.cpu().numpy())

    maes = np.concatenate(all_maes)
    psnrs = np.concatenate(all_psnrs)
    ssims = np.concatenate(all_ssims)
    hf_real = np.concatenate(all_hf_real)
    hf_recon = np.concatenate(all_hf_recon)

    print("\nReconstruction metrics (vs cifar10-vae v2: MAE 0.065, PSNR 21.5, SSIM 0.742, HF 51.4%):")
    report("TEST overall", maes, psnrs, ssims, hf_real, hf_recon)
    for c in range(10):
        mask = y_test == c
        report(f"  {label_names[c]:<10}", maes[mask], psnrs[mask], ssims[mask],
               hf_real[mask], hf_recon[mask])

    # codebook usage on the test set: how many of the K codes actually fire
    model.eval()
    used = set()
    with torch.no_grad():
        for i in range(0, X_test.shape[0], args.batch_size):
            x = torch.from_numpy(X_test[i:i + args.batch_size]).to(device)
            z_e = model.encoder(x)
            z_e = z_e.permute(0, 2, 3, 1).contiguous()
            flat = z_e.reshape(-1, model.embedding_dim)
            dist = (flat.pow(2).sum(1, keepdim=True)
                    + model.quantizer.embedding.weight.pow(2).sum(1)
                    - 2.0 * flat @ model.quantizer.embedding.weight.t())
            used.update(dist.argmin(1).cpu().numpy().tolist())
    print(f"\nCodebook usage on test set: {len(used)}/{model.codebook_size} codes fired "
          f"({100 * len(used) / model.codebook_size:.1f}%)")

    os.makedirs(args.output_dir, exist_ok=True)
    grid = reconstruction_grid(model, X_test, y_test, device)
    grid_path = os.path.join(args.output_dir, "reconstruction_grid.png")
    write_png(grid_path, grid)
    print(f"Saved reconstruction grid (top=real, bottom=reconstructed, classes 0-9) to {grid_path}")
    print("No prior_samples.png: a VQ-VAE has no N(0,I) prior to sample - "
          "generation needs a learned prior over the discrete codes (the planned cascade).")


if __name__ == "__main__":
    main()

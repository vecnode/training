"""
Evaluate a trained flow-matching checkpoint against the held-out MNIST test
split, and save PNGs so generation quality can actually be looked at:

  - samples_grid.png - pure generations: x0 ~ N(0, I) at t=0, integrated to
    t=1 by the learned ODE. This is the file to put next to
    training/mnist-vae's prior_samples.png.
  - reconstruction_grid.png - one example per digit 0-9, real image on top
    of its round trip, in the same two-row layout mnist-vae's
    reconstruction_grid.png uses. A flow model has no encoder, but its ODE
    is deterministic and invertible: integrating *backwards* from t=1 to
    t=0 maps a real digit to the noise it would have come from, and
    integrating forwards again brings it back. Note the caveat on the step
    sweep below - the round trip is a real measurement, but of the solver's
    accuracy rather than of the model's quality.

It also reports:

  - test-set velocity MSE - the training objective on genuinely unseen
    data, with t/noise drawn from a fixed seed so it is comparable to the
    val numbers in the training log.
  - a solver step sweep - round-trip MAE/PSNR at several step counts and
    both solvers. Read this as a measure of *ODE discretization error*, not
    of sample quality: it says how many steps the solver needs before the
    forward and backward integrations agree, which is how to choose
    --num-steps. It is deliberately not a quality score - an untrained
    model whose velocity field is near zero round-trips almost perfectly,
    because the identity map is trivially self-inverse.
  - a nearest-neighbour check - for a handful of generated samples, the L2
    distance to the closest training image, plus nearest_neighbours.png
    showing each sample above the training image it is closest to. This is
    the memorization guard: samples should resemble the data without being
    copies of it. The same distance is also measured for real held-out test
    digits as a control, because the raw number means nothing on its own -
    60k training images cover MNIST's simple shapes densely, so even a
    genuinely novel digit lands close to something.

Deliberately no FID: a real one needs a pretrained Inception network, which
would contradict the from-scratch rule of this folder. Rather than invent a
substitute number, sample quality here is judged by looking at
samples_grid.png; everything printed below is something actually measured.

No imaging library (no Pillow/matplotlib): PNG is written by hand with the
stdlib zlib module (deflate-compress raw 8-bit grayscale scanlines into a
single IDAT chunk) - the same "no extra dependency" spirit as the rest of
this pipeline, and the same writer training/mnist-vae uses.

Usage:
    uv run --directory training/flow-matching-mnist python evaluate_flow.py \
        --data-path data/mnist.npz \
        --checkpoint-path runs/mnist_flow/flow_best.pt \
        --num-steps 50 \
        --output-dir runs/mnist_flow
"""

import argparse
import os
import struct
import zlib

import numpy as np
import torch

from train_flow import VelocityUNet, run_epoch, to_model_scale, to_pixel_scale


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


@torch.no_grad()
def integrate(model, x, t_start, t_end, num_steps, solver="euler"):
    """Solve dx/dt = v(x, t) from t_start to t_end in num_steps fixed steps.

    Sampling runs t_start=0 -> t_end=1 (noise to data); encoding a real
    image runs t_start=1 -> t_end=0, which works because the ODE is
    deterministic and time-reversible - the same field integrated the other
    way. dt carries the sign, so one routine serves both directions.

      euler: x <- x + v(x, t) * dt                    1 eval  per step
      heun:  predictor/corrector, averaging the       2 evals per step
             velocity at the current and predicted
             points - second-order, so it needs far
             fewer steps for the same error.
    """
    dt = (t_end - t_start) / num_steps
    for step in range(num_steps):
        t = t_start + step * dt
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)
        v = model(x, t_batch)
        if solver == "euler":
            x = x + v * dt
        elif solver == "heun":
            t_next = torch.full((x.shape[0],), t + dt, device=x.device, dtype=torch.float32)
            v_next = model(x + v * dt, t_next)
            x = x + 0.5 * (v + v_next) * dt
        else:
            raise ValueError(f"unknown solver: {solver}")
    return x


@torch.no_grad()
def generate(model, num_samples, device, num_steps, solver, seed):
    """x0 ~ N(0,I) at t=0, integrated forward to t=1. No encoder, no data."""
    generator = torch.Generator(device=device).manual_seed(seed)
    x0 = torch.randn(num_samples, 1, 28, 28, device=device, generator=generator)
    return to_pixel_scale(integrate(model, x0, 0.0, 1.0, num_steps, solver))


@torch.no_grad()
def round_trip(model, x01, device, num_steps, solver):
    """Real image -> (backwards ODE) -> noise -> (forwards ODE) -> image."""
    x1 = to_model_scale(x01.to(device))
    x0_hat = integrate(model, x1, 1.0, 0.0, num_steps, solver)
    return to_pixel_scale(integrate(model, x0_hat, 0.0, 1.0, num_steps, solver))


def round_trip_metrics(model, X, device, num_steps, solver, batch_size):
    """MAE and PSNR between real test images and their round trips, over
    the first X.shape[0] test rows."""
    abs_error_sum = 0.0
    squared_error_sum = 0.0
    count = 0
    for start in range(0, X.shape[0], batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).unsqueeze(1)
        recon = round_trip(model, batch, device, num_steps, solver).cpu()
        diff = (recon - batch).numpy()
        abs_error_sum += float(np.abs(diff).sum())
        squared_error_sum += float((diff ** 2).sum())
        count += diff.size
    mae = abs_error_sum / count
    rmse = (squared_error_sum / count) ** 0.5
    psnr = 20.0 * np.log10(1.0 / rmse) if rmse > 0 else float("inf")
    return mae, psnr


def nearest_training_neighbours(samples01, X_train, device, batch_size=4096):
    """For each generated sample, the L2-nearest training image and that
    distance. Brute force over the whole training set on the GPU (54k x 784
    is small); no index library, and no pretrained feature extractor -
    plain pixel distance, which is what makes "is this a copy?" answerable.
    """
    flat_samples = samples01.reshape(samples01.shape[0], -1).to(device)
    best_distance = torch.full((flat_samples.shape[0],), float("inf"), device=device)
    best_index = torch.zeros(flat_samples.shape[0], dtype=torch.long, device=device)

    for start in range(0, X_train.shape[0], batch_size):
        block = torch.from_numpy(X_train[start:start + batch_size]).to(device)
        block = block.reshape(block.shape[0], -1)
        distances = torch.cdist(flat_samples, block)          # (num_samples, block)
        block_best, block_argmin = distances.min(dim=1)
        improved = block_best < best_distance
        best_distance = torch.where(improved, block_best, best_distance)
        best_index = torch.where(improved, block_argmin + start, best_index)

    return best_distance.cpu().numpy(), best_index.cpu().numpy()


def reconstruction_grid(model, X, y, device, num_steps, solver, digits=range(10)):
    """One real/round-tripped pair per digit, laid out as: row 0 = 10 real
    digits, row 1 = their 10 round trips - the same two-row layout
    training/mnist-vae's reconstruction_grid.png uses, so the two PNGs can
    be read side by side."""
    reals, recons = [], []
    for d in digits:
        idx = int(np.argmax(y == d))  # first test example of digit d
        x01 = torch.from_numpy(X[idx:idx + 1]).unsqueeze(1)
        recon = round_trip(model, x01, device, num_steps, solver)
        reals.append(X[idx])
        recons.append(recon.squeeze().cpu().numpy())
    return make_grid(reals + recons, grid_cols=len(digits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--checkpoint-path", default="runs/mnist_flow/flow_best.pt")
    parser.add_argument("--output-dir", default="runs/mnist_flow")
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema",
                        help="Which weights to evaluate: the EMA shadow (what the "
                             "training run keeps for sampling) or the raw last-step weights.")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="ODE steps used for the two PNG grids.")
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=64,
                        help="How many unconditional samples to generate for samples_grid.png.")
    parser.add_argument("--sweep-steps", type=int, nargs="*", default=[5, 10, 20, 50, 100],
                        help="Step counts to compare in the solver sweep; empty to skip.")
    parser.add_argument("--sweep-samples", type=int, default=256,
                        help="Test images used for each sweep row (round-trip MAE/PSNR).")
    parser.add_argument("--nn-check-samples", type=int, default=8,
                        help="Generated samples to match against the training set as a "
                             "memorization check; 0 to skip.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = VelocityUNet(checkpoint["base_channels"]).to(device)
    if args.weights == "ema" and checkpoint.get("ema_state") is not None:
        model.load_state_dict(checkpoint["ema_state"])
    else:
        if args.weights == "ema":
            print("Checkpoint has no EMA weights; falling back to raw weights.")
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_mse={checkpoint['val_loss']:.4f}, base_channels={checkpoint['base_channels']}, "
          f"sigma_min={checkpoint['sigma_min']}, weights={args.weights})")

    rng = np.random.default_rng(args.seed)
    test_mse = run_epoch(
        model, X_test, args.batch_size, checkpoint["sigma_min"], device, rng,
        optimizer=None, eval_seed=args.seed,
    )
    print(f"Test set: velocity mse={test_mse:.4f}  (training objective, unseen data)")

    os.makedirs(args.output_dir, exist_ok=True)

    grid_cols = int(np.ceil(np.sqrt(args.num_samples)))
    samples = generate(model, args.num_samples, device, args.num_steps, args.solver, args.seed)
    sample_grid = make_grid(list(samples.squeeze(1).cpu().numpy()), grid_cols=grid_cols)
    sample_path = os.path.join(args.output_dir, "samples_grid.png")
    write_png(sample_path, sample_grid)
    print(f"Saved {args.num_samples} samples (x0 ~ N(0,I) integrated to t=1, "
          f"{args.num_steps} {args.solver} steps) to {sample_path}")

    recon_grid = reconstruction_grid(model, X_test, y_test, device, args.num_steps, args.solver)
    recon_path = os.path.join(args.output_dir, "reconstruction_grid.png")
    write_png(recon_path, recon_grid)
    print(f"Saved round-trip grid (top=real, bottom=ODE round trip, digits 0-9) to {recon_path}")

    if args.sweep_steps:
        X_sweep = X_test[:args.sweep_samples]
        print(f"\nSolver sweep (round trip over {X_sweep.shape[0]} test images).")
        print("Measures ODE discretization error - how many steps before forward and")
        print("backward integration agree - NOT sample quality (a near-zero velocity")
        print("field round-trips perfectly, since the identity is its own inverse).")
        print(f"  {'steps':>6}  {'solver':>6}  {'evals':>6}  {'MAE':>7}  {'PSNR(dB)':>9}")
        for solver in ("euler", "heun"):
            for steps in args.sweep_steps:
                mae, psnr = round_trip_metrics(
                    model, X_sweep, device, steps, solver, args.batch_size
                )
                evals = steps * (1 if solver == "euler" else 2)
                print(f"  {steps:>6}  {solver:>6}  {evals:>6}  {mae:>7.4f}  {psnr:>9.2f}")

    if args.nn_check_samples:
        X_train = data["X_train"]
        check = generate(
            model, args.nn_check_samples, device, args.num_steps, args.solver, args.seed + 1
        )
        distances, indices = nearest_training_neighbours(check.squeeze(1).cpu(), X_train, device)

        # Control: the same measurement for real held-out test digits. Without
        # it the sample distance has no scale - MNIST's 60k training images
        # cover simple digit shapes densely, so even a genuinely novel digit
        # lands close to *something*. Samples are suspicious only if they sit
        # markedly closer than real unseen data does.
        real = torch.from_numpy(X_test[:args.nn_check_samples])
        real_distances, _ = nearest_training_neighbours(real, X_train, device)

        print(f"\nNearest-neighbour check ({args.nn_check_samples} fresh samples vs "
              f"{X_train.shape[0]} training images, L2 on raw pixels):")
        print(f"  generated:      mean {distances.mean():.3f}  min {distances.min():.3f}  "
              f"max {distances.max():.3f}")
        print(f"  real test data: mean {real_distances.mean():.3f}  min {real_distances.min():.3f}  "
              f"max {real_distances.max():.3f}   <- control")
        print("  Samples much closer than the control would mean memorization; a near-zero")
        print("  minimum would mean the model is reproducing training images outright.")

        pairs = list(check.squeeze(1).cpu().numpy()) + [X_train[i] for i in indices]
        nn_path = os.path.join(args.output_dir, "nearest_neighbours.png")
        write_png(nn_path, make_grid(pairs, grid_cols=args.nn_check_samples))
        print(f"  Saved samples (top) above their nearest training images (bottom) to {nn_path}")


if __name__ == "__main__":
    main()

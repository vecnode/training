"""
Evaluate a trained class-conditional DiT checkpoint against the held-out
CIFAR-100 test split, and save PNGs so generation quality can actually be
looked at (all hand-written zlib RGB PNGs, no imaging library):

  - samples_grid.png - 10x10 = 100 class-conditional generations, one per
    CIFAR-100 fine class (row-major class order, cell i = class i), each a
    fresh x0 ~ N(0,I) integrated to t=1 by the guided velocity field. This
    is the file to look at: it answers "does the class conditioning work,
    and what do the samples look like?" at a glance.
  - cfg_sweep.png - the same few fixed latents generated at several
    classifier-free-guidance strengths (rows = cfg scale, columns =
    classes, the latent per column is fixed), so the guidance-vs-diversity
    trade-off is visible in one image instead of asserted.
  - nearest_neighbours.png + L2 distances - the memorization guard, in the
    same form as training/flow-matching-mnist and
    training/fashion-mnist-dcgan: each generated sample above the closest
    training image to it (brute-force pixel L2 over all 50k, no index
    library, no learned features), with real held-out test images measured
    the same way as a control - the raw distance means nothing on its own,
    because 50k images cover CIFAR-100's simple 32x32 shapes densely.
    Samples much closer than the control would mean memorization.

It also reports the test-set velocity MSE - the training objective on
genuinely unseen data, with t/noise/class-dropout drawn from a fixed seed
so it is comparable to the val numbers in the training log.

Deliberately no FID: a real one needs a pretrained Inception network, which
would contradict the from-scratch rule of this folder. Rather than invent a
substitute number, sample quality here is judged by looking at
samples_grid.png; everything printed below is something actually measured.

Usage:
    uv run --directory training/dit-cifar100 python evaluate_dit.py \
        --data-path data/cifar100.npz \
        --checkpoint-path runs/dit_cifar100/dit_best.pt \
        --num-steps 50 --cfg-scale 3.0 \
        --output-dir runs/dit_cifar100
"""

import argparse
import os

import numpy as np
import torch

from png_utils import make_rgb_grid, write_png
from train_dit import DiT, run_epoch, sample_ode

CFG_SWEEP_CLASSES = [0, 1, 2, 3]          # apple, aquarium_fish, baby, bear
CFG_SWEEP_SCALES = [1.0, 1.5, 2.0, 3.0, 5.0]


@torch.no_grad()
def nearest_training_neighbours(samples01, X_train, device, batch_size=4096):
    """For each generated sample, the L2-nearest training image and that
    distance. Brute force over the whole training set on the GPU (50k x
    3072 is small); no index library, and no pretrained feature extractor -
    plain pixel distance, which is what makes "is this a copy?" answerable.
    samples01: (n, 3, 32, 32) float [0,1] on CPU."""
    flat_samples = samples01.reshape(samples01.shape[0], -1).to(device)
    best_distance = torch.full((flat_samples.shape[0],), float("inf"), device=device)
    best_index = torch.zeros(flat_samples.shape[0], dtype=torch.long, device=device)

    for start in range(0, X_train.shape[0], batch_size):
        block = torch.from_numpy(X_train[start:start + batch_size]).to(device)
        block = block.reshape(block.shape[0], -1)
        distances = torch.cdist(flat_samples, block)          # (n, block)
        block_best, block_argmin = distances.min(dim=1)
        improved = block_best < best_distance
        best_distance = torch.where(improved, block_best, best_distance)
        best_index = torch.where(improved, block_argmin + start, best_index)

    return best_distance.cpu().numpy(), best_index.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar100.npz")
    parser.add_argument("--checkpoint-path", default="runs/dit_cifar100/dit_best.pt")
    parser.add_argument("--output-dir", default="runs/dit_cifar100")
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema",
                        help="Which weights to evaluate: the EMA shadow (what the "
                             "training run keeps for sampling) or the raw last-step weights.")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Euler steps used for every generation.")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="Classifier-free guidance strength for the main "
                             "sample grid and the nearest-neighbour check.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for the test velocity-MSE loop.")
    parser.add_argument("--nn-check-samples", type=int, default=8,
                        help="Generated samples to match against the training set as "
                             "a memorization check; 0 to skip.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]
    fine_names = [str(n) for n in data["fine_label_names"]]

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = DiT(
        patch_size=checkpoint["patch_size"],
        dim=checkpoint["dim"],
        depth=checkpoint["depth"],
        heads=checkpoint["heads"],
        mlp_ratio=checkpoint["mlp_ratio"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    if args.weights == "ema" and checkpoint.get("ema_state") is not None:
        model.load_state_dict(checkpoint["ema_state"])
    else:
        if args.weights == "ema":
            print("Checkpoint has no EMA weights; falling back to raw weights.")
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_mse={checkpoint['val_loss']:.4f}, dim={checkpoint['dim']}, "
          f"depth={checkpoint['depth']}, patch={checkpoint['patch_size']}, "
          f"sigma_min={checkpoint['sigma_min']}, cfg_dropout={checkpoint['cfg_dropout']}, "
          f"weights={args.weights})")

    rng = np.random.default_rng(args.seed)
    test_mse = run_epoch(
        model, X_test, y_test, args.batch_size, checkpoint["sigma_min"], 0.0,
        device, rng, optimizer=None, eval_seed=args.seed,
    )
    print(f"Test set: velocity mse={test_mse:.4f}  "
          f"(training objective, conditional on the true class, unseen data)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- samples_grid.png: 100 classes, one sample each -------------------
    classes = torch.arange(100, dtype=torch.long, device=device)
    samples = sample_ode(
        model, classes, args.num_steps, args.cfg_scale,
        checkpoint["sigma_min"], device, seed=args.seed,
    )
    grid = make_rgb_grid(list(samples), grid_cols=10)
    sample_path = os.path.join(args.output_dir, "samples_grid.png")
    write_png(sample_path, grid)
    print(f"\nSaved 100 class-conditional samples (one per fine class, "
          f"{args.num_steps} Euler steps, cfg={args.cfg_scale}) to {sample_path}")
    print("Layout: 10 rows x 10 cols, cell = class (row-major, 0..99), i.e.")
    for r in range(10):
        names = "  ".join(f"{r*10+c}:{fine_names[r*10+c]}" for c in range(0, 10, 5))
        print(f"  row {r}: {names}")

    # ---- cfg_sweep.png: fixed latents x guidance strengths ---------------
    # Rows = cfg scales, columns = classes; the latent per column is fixed
    # (same seed per class), so each column shows the same noise evolving
    # under stronger guidance.
    sweep = []
    for cls in CFG_SWEEP_CLASSES:
        col = []
        for cfg in CFG_SWEEP_SCALES:
            one = torch.full((1,), cls, dtype=torch.long, device=device)
            img = sample_ode(
                model, one, args.num_steps, cfg,
                checkpoint["sigma_min"], device, seed=args.seed + cls,
            )
            col.append(img[0])
        sweep.append(col)
    sweep_flat = [sweep[c][r] for r in range(len(CFG_SWEEP_SCALES))
                  for c in range(len(CFG_SWEEP_CLASSES))]
    sweep_grid = make_rgb_grid(sweep_flat, grid_cols=len(CFG_SWEEP_CLASSES))
    sweep_path = os.path.join(args.output_dir, "cfg_sweep.png")
    write_png(sweep_path, sweep_grid)
    print(f"\nSaved CFG sweep to {sweep_path} "
          f"(rows = cfg {CFG_SWEEP_SCALES}, cols = classes "
          f"{[fine_names[c] for c in CFG_SWEEP_CLASSES]}, latent per column fixed)")

    # ---- nearest-neighbour memorization check ----------------------------
    if args.nn_check_samples:
        X_train = data["X_train"]
        check_classes = torch.tensor(
            np.arange(args.nn_check_samples) % 100, dtype=torch.long, device=device
        )
        check = sample_ode(
            model, check_classes, args.num_steps, args.cfg_scale,
            checkpoint["sigma_min"], device, seed=args.seed + 1,
        )
        distances, indices = nearest_training_neighbours(check.cpu(), X_train, device)

        # Control: the same measurement for real held-out test images. Without
        # it the sample distance has no scale - 50k training images cover
        # CIFAR-100's simple shapes densely, so even a genuinely novel image
        # lands close to *something*. Samples are suspicious only if they sit
        # markedly closer than real unseen data does.
        real = torch.from_numpy(X_test[:args.nn_check_samples])
        real_distances, _ = nearest_training_neighbours(real, X_train, device)

        print(f"\nNearest-neighbour check ({args.nn_check_samples} fresh samples vs "
              f"{X_train.shape[0]} training images, L2 on raw pixels, "
              f"cfg={args.cfg_scale}):")
        print(f"  generated:      mean {distances.mean():.3f}  min {distances.min():.3f}  "
              f"max {distances.max():.3f}")
        print(f"  real test data: mean {real_distances.mean():.3f}  min {real_distances.min():.3f}  "
              f"max {real_distances.max():.3f}   <- control")
        print("  Samples much closer than the control would mean memorization; a near-zero")
        print("  minimum would mean the model is reproducing training images outright.")

        pairs = list(check.cpu()) + [torch.from_numpy(X_train[i]) for i in indices]
        nn_path = os.path.join(args.output_dir, "nearest_neighbours.png")
        write_png(nn_path, make_rgb_grid(pairs, grid_cols=args.nn_check_samples))
        print(f"  Saved samples (top row) above their nearest training images "
              f"(bottom row) to {nn_path}")


if __name__ == "__main__":
    main()

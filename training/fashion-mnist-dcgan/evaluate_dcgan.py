"""
Evaluate a trained DCGAN checkpoint - i.e. actually judge the generator,
since a GAN's D/G loss curves cannot tell a good generator from a collapsed
one. Outputs:

  - samples_grid.png - --num-samples generated images in a square grid
    (hand-written zlib PNG writer, same as training/mnist-vae).
  - real_samples.png - one real test example per class (labels the GAN
    never saw during training), so the two grids can be read side by side.
  - nearest_neighbours.png + L2 distances - the memorization guard: for a
    handful of generated samples, the L2-nearest training image (pixel
    space, brute force on the GPU), printed against the same statistic for
    a control of held-out real test images. Samples much closer than the
    control would mean the generator is copying the training set instead of
    learning the distribution.
  - pairwise-diversity probe - mean pairwise L2 distance over a batch of
    generated samples vs the same over real training images; a large gap
    (generated much lower) is the mode-collapse signature.

No pretrained networks anywhere: no FID, no Inception Score - both need a
pretrained Inception, the same rule that keeps FID out of
training/flow-matching-mnist. Pixel-space L2 only, so "is this a copy?"
stays answerable.

Usage:
    uv run --directory training/fashion-mnist-dcgan python evaluate_dcgan.py \
        --data-path data/fashion_mnist.npz \
        --checkpoint-path runs/fashion_mnist_dcgan/dcgan_final.pt \
        --output-dir runs/fashion_mnist_dcgan
"""

import argparse
import os

import numpy as np
import torch

from train_dcgan import Generator, make_grid, write_png


def nearest_training_neighbours(samples01, X_train, device, batch_size=4096):
    """For each sample, the L2-nearest training image and that distance.
    Brute force over the whole training set on the GPU (60k x 784 is
    small); no index library, and no pretrained feature extractor - plain
    pixel distance, which is what makes "is this a copy?" answerable."""
    flat_samples = torch.from_numpy(samples01.reshape(samples01.shape[0], -1)).to(device)
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


def mean_pairwise_l2(samples01, device):
    """Mean pairwise L2 distance over a (capped) set of samples - the
    mode-collapse probe. k=256 gives a 256x256 distance matrix, trivial on
    the GPU; self-pairs are excluded."""
    k = samples01.shape[0]
    flat = torch.from_numpy(samples01.reshape(k, -1)).to(device)
    dist = torch.cdist(flat, flat)
    mask = ~torch.eye(k, dtype=torch.bool, device=device)
    return dist[mask].mean().item()


def real_grid(X_test, y_test, class_names):
    """One real test example per class, laid out left to right."""
    examples = []
    for c in range(len(class_names)):
        idx = int(np.argmax(y_test == c))
        examples.append(X_test[idx])
    return make_grid(examples, grid_cols=len(class_names))


def sample_grid(generator, z_dim, num_samples, device, seed):
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, generator=rng, device=device)
        out = (generator(z) + 1.0) / 2.0  # [-1,1] -> [0,1]
    grid_cols = int(np.ceil(np.sqrt(num_samples)))
    return make_grid(list(out.squeeze(1).cpu().numpy()), grid_cols=grid_cols)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/fashion_mnist.npz")
    parser.add_argument("--checkpoint-path", default="runs/fashion_mnist_dcgan/dcgan_final.pt")
    parser.add_argument("--output-dir", default="runs/fashion_mnist_dcgan")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--nn-check", type=int, default=1,
                        help="Run the nearest-neighbour memorization check "
                             "(needs X_train; 0 to skip).")
    parser.add_argument("--nn-k", type=int, default=10,
                        help="How many generated samples get the NN check "
                             "(also drawn for the real-image control).")
    parser.add_argument("--diversity-k", type=int, default=256,
                        help="How many samples feed the pairwise-diversity probe.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"] if "class_names" in data else np.arange(10)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    generator = Generator(checkpoint["z_dim"]).to(device)
    generator.load_state_dict(checkpoint["generator_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(z_dim={checkpoint['z_dim']}, d_loss={checkpoint['d_loss']:.4f}, "
          f"g_loss={checkpoint['g_loss']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Sample grid ---
    grid = sample_grid(generator, checkpoint["z_dim"], args.num_samples, device, args.seed)
    grid_path = os.path.join(args.output_dir, "samples_grid.png")
    write_png(grid_path, grid)
    print(f"Saved {args.num_samples} generated samples to {grid_path}")

    # --- Real grid (one per class) ---
    real_path = os.path.join(args.output_dir, "real_samples.png")
    write_png(real_path, real_grid(X_test, y_test, class_names))
    print(f"Saved one real test example per class to {real_path}")

    # --- Pairwise diversity probe (mode collapse) ---
    gen_rng = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.diversity_k, checkpoint["z_dim"], generator=gen_rng, device=device)
        gen_batch = ((generator(z) + 1.0) / 2.0).squeeze(1).cpu().numpy()
    rng = np.random.default_rng(args.seed)
    real_batch = X_train[rng.choice(X_train.shape[0], args.diversity_k, replace=False)]
    gen_diversity = mean_pairwise_l2(gen_batch, device)
    real_diversity = mean_pairwise_l2(real_batch, device)
    print(f"Diversity (mean pairwise L2, k={args.diversity_k}): "
          f"generated={gen_diversity:.4f}  real={real_diversity:.4f}  "
          f"ratio={gen_diversity / real_diversity:.3f}")

    # --- Memorization check ---
    if args.nn_check:
        gen_z = torch.Generator(device=device).manual_seed(args.seed)
        with torch.no_grad():
            z = torch.randn(args.nn_k, checkpoint["z_dim"], generator=gen_z, device=device)
            gen_k = ((generator(z) + 1.0) / 2.0).squeeze(1).cpu().numpy()
        real_k = X_test[rng.choice(X_test.shape[0], args.nn_k, replace=False)]

        gen_dist, gen_idx = nearest_training_neighbours(gen_k, X_train, device)
        real_dist, _ = nearest_training_neighbours(real_k, X_train, device)

        print(f"Nearest training image, mean L2 (k={args.nn_k}):")
        print(f"  generated: {gen_dist.mean():.4f}   real control: {real_dist.mean():.4f}")
        if gen_dist.mean() < real_dist.mean() * 0.5:
            print("  WARNING: generated samples sit much closer to the training set")
            print("  than real ones do - the generator may be memorizing.")
        elif gen_dist.mean() < real_dist.mean() * 0.9:
            print("  Note: generated samples are somewhat closer to the training set")
            print("  than the real control - watch the grids.")
        else:
            print("  Generated samples are no closer to the training set than real")
            print("  ones are - no memorization signal.")

        # Visual: k generated samples (top) above their nearest training images (bottom).
        nn_grid = make_grid(
            list(gen_k) + [X_train[i] for i in gen_idx],
            grid_cols=args.nn_k,
        )
        nn_path = os.path.join(args.output_dir, "nearest_neighbours.png")
        write_png(nn_path, nn_grid)
        print(f"Saved generated (top) vs nearest training image (bottom) to {nn_path}")


if __name__ == "__main__":
    main()

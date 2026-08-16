"""
K-means clustering, implemented by hand with numpy - no scikit-learn.

The algorithm: given k, pick k starting centroids, then repeat until
assignments stop changing (or a max number of iterations):
  1. assign each point to its nearest centroid (Euclidean distance)
  2. move each centroid to the mean of the points assigned to it

Run unsupervised on raw MNIST pixels (labels are never touched here - only
build_mnist_dataset.py and evaluate_kmeans.py look at y_train/y_test, and
only for saving/scoring, never for clustering itself).

Usage:
    uv run --directory training/mnist-kmeans python train_kmeans.py \
        --data-path data/mnist.npz \
        --k 10 \
        --num-iters 50 \
        --output-dir runs/mnist_kmeans
"""

import argparse
import os

import numpy as np


def init_centroids(X, k, rng):
    """Pick k distinct training points at random as starting centroids.
    A known weak point vs. k-means++ (bad luck can start two centroids
    close together), but simple and the standard textbook baseline."""
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()


def squared_distances(X, centroids):
    """Pairwise squared Euclidean distance between every row of X (n, d)
    and every centroid (k, d), returned as an (n, k) matrix.

    Uses the expansion ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 so the
    expensive part is one matrix product (X @ centroids.T) instead of an
    explicit n*k*d loop - the same numbers, just vectorized.
    """
    x_sq = np.sum(X * X, axis=1, keepdims=True)          # (n, 1)
    c_sq = np.sum(centroids * centroids, axis=1)          # (k,)
    cross = X @ centroids.T                                # (n, k)
    return x_sq - 2.0 * cross + c_sq[np.newaxis, :]


def assign_clusters(X, centroids):
    dists = squared_distances(X, centroids)
    return np.argmin(dists, axis=1)


def update_centroids(X, assignments, k, rng):
    """Move each centroid to the mean of its assigned points. An empty
    cluster (no points assigned) is reseeded to a random training point
    instead of left as NaN, so a single unlucky init can't crash a run."""
    d = X.shape[1]
    centroids = np.zeros((k, d), dtype=X.dtype)
    for j in range(k):
        members = X[assignments == j]
        if members.shape[0] == 0:
            centroids[j] = X[rng.integers(X.shape[0])]
        else:
            centroids[j] = members.mean(axis=0)
    return centroids


def inertia(X, centroids, assignments):
    """Sum of squared distances from each point to its assigned centroid -
    the objective k-means is (locally) minimizing. Lower is tighter
    clusters; useful for comparing runs with different seeds."""
    diffs = X - centroids[assignments]
    return float(np.sum(diffs * diffs))


def kmeans(X, k, num_iters, seed):
    rng = np.random.default_rng(seed)
    centroids = init_centroids(X, k, rng)
    assignments = assign_clusters(X, centroids)

    for iteration in range(1, num_iters + 1):
        centroids = update_centroids(X, assignments, k, rng)
        new_assignments = assign_clusters(X, centroids)
        num_changed = int(np.sum(new_assignments != assignments))
        assignments = new_assignments
        print(f"iter {iteration:3d}/{num_iters}  "
              f"reassigned={num_changed}  inertia={inertia(X, centroids, assignments):.1f}")
        if num_changed == 0:
            print(f"Converged after {iteration} iterations (no reassignments).")
            break

    return centroids, assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--output-dir", default="runs/mnist_kmeans")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_train = data["X_train"]

    print(f"Train rows: {X_train.shape[0]}  Pixels/row: {X_train.shape[1]}  "
          f"k={args.k}")

    centroids, assignments = kmeans(
        X_train, k=args.k, num_iters=args.num_iters, seed=args.seed
    )

    cluster_sizes = np.bincount(assignments, minlength=args.k)
    print(f"Final cluster sizes: {cluster_sizes.tolist()}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "centroids.npz")
    np.savez(out_path, centroids=centroids, k=args.k)
    print(f"Saved trained centroids to {out_path}")
    print("Run evaluate_kmeans.py against the held-out test split next.")


if __name__ == "__main__":
    main()

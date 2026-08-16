"""
Evaluate trained k-means centroids against the held-out MNIST test split,
and render the centroids themselves as a viewable image grid.

K-means never sees labels during training, so clusters aren't numbered 0-9
by digit - this script builds a (cluster x true-label) confusion matrix on
the test set and greedily matches each cluster to the digit it overlaps
with most (largest cell first, no cluster or label reused), a hand-rolled
stand-in for scipy's Hungarian algorithm. All metrics are computed by hand
from that matrix - no scikit-learn.metrics.

Usage:
    uv run --directory training/mnist-kmeans python evaluate_kmeans.py \
        --data-path data/mnist.npz \
        --centroids-path runs/mnist_kmeans/centroids.npz \
        --output-dir runs/mnist_kmeans
"""

import argparse
import os

import numpy as np

from train_kmeans import assign_clusters


def confusion_matrix(assignments, y_true, k, num_labels=10):
    """counts[j, d] = number of test points assigned to cluster j whose
    true label is digit d."""
    counts = np.zeros((k, num_labels), dtype=np.int64)
    for j, d in zip(assignments, y_true):
        counts[j, d] += 1
    return counts


def match_clusters_to_labels(counts):
    """Greedy max-weight one-to-one matching between clusters (rows) and
    digit labels (columns): repeatedly take the largest remaining cell,
    lock in that cluster->label pairing, and remove its row and column from
    consideration. Not globally optimal like the Hungarian algorithm, but a
    simple, dependency-free approximation - good enough since MNIST's
    cluster/label overlap is usually dominated by one clear best match per
    cluster.

    A cluster with no test points ever assigned still gets matched to
    whatever label remains free (it just won't affect accuracy).
    """
    k, num_labels = counts.shape
    remaining_clusters = set(range(k))
    remaining_labels = set(range(num_labels))
    mapping = {}

    cells = sorted(
        ((counts[j, d], j, d) for j in range(k) for d in range(num_labels)),
        reverse=True,
    )
    for count, j, d in cells:
        if j in remaining_clusters and d in remaining_labels:
            mapping[j] = d
            remaining_clusters.discard(j)
            remaining_labels.discard(d)

    # Only reachable if k > num_labels; pair off whatever clusters are left.
    for j in list(remaining_clusters):
        d = remaining_labels.pop() if remaining_labels else -1
        mapping[j] = d

    return mapping


def write_pgm_grid(centroids, image_size, grid_cols, path):
    """Write the centroids as one plain-text PGM (P2) image: each centroid
    reshaped to image_size x image_size, tiled into a grid. PGM needs no
    imaging library to write or view (most image viewers/converters read
    it directly), which keeps this dependency-free like the rest of the
    pipeline.
    """
    k = centroids.shape[0]
    grid_rows = (k + grid_cols - 1) // grid_cols
    pixels = (np.clip(centroids, 0.0, 1.0) * 255.0).astype(np.uint8)
    pixels = pixels.reshape(k, image_size, image_size)

    border = 2
    tile = image_size + border
    canvas = np.zeros((grid_rows * tile, grid_cols * tile), dtype=np.uint8)
    for j in range(k):
        r, c = divmod(j, grid_cols)
        top, left = r * tile, c * tile
        canvas[top: top + image_size, left: left + image_size] = pixels[j]

    height, width = canvas.shape
    with open(path, "w") as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")
        for row in canvas:
            f.write(" ".join(str(v) for v in row.tolist()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--centroids-path", default="runs/mnist_kmeans/centroids.npz")
    parser.add_argument("--output-dir", default="runs/mnist_kmeans")
    parser.add_argument("--grid-cols", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]

    weights = np.load(args.centroids_path, allow_pickle=True)
    centroids, k = weights["centroids"], int(weights["k"])

    assignments = assign_clusters(X_test, centroids)
    counts = confusion_matrix(assignments, y_test, k)
    mapping = match_clusters_to_labels(counts)

    predicted_labels = np.array([mapping[j] for j in assignments])
    correct = predicted_labels == y_test
    accuracy = float(np.mean(correct))

    print(f"Test rows: {len(y_test)}  Clusters: {k}")
    print()
    print("Cluster -> digit mapping (greedy max-overlap matching):")
    for j in range(k):
        cluster_size = int(counts[j].sum())
        matched = mapping[j]
        hit = int(counts[j, matched]) if matched >= 0 else 0
        purity = hit / cluster_size if cluster_size > 0 else 0.0
        print(f"  cluster {j:2d} -> digit {matched}   "
              f"({hit}/{cluster_size} test points, purity={purity:.3f})")

    print()
    print(f"Overall clustering accuracy = {accuracy:.4f}")

    print()
    print("Per-digit precision/recall (after cluster->digit matching):")
    for d in range(10):
        tp = int(np.sum((predicted_labels == d) & (y_test == d)))
        pred_pos = int(np.sum(predicted_labels == d))
        actual_pos = int(np.sum(y_test == d))
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / actual_pos if actual_pos > 0 else 0.0
        print(f"  digit {d}: precision={precision:.3f}  recall={recall:.3f}  "
              f"(n={actual_pos})")

    os.makedirs(args.output_dir, exist_ok=True)
    grid_path = os.path.join(args.output_dir, "centroid_grid.pgm")
    write_pgm_grid(centroids, image_size=28, grid_cols=args.grid_cols, path=grid_path)
    print()
    print(f"Saved centroid visualization to {grid_path} "
          f"(plain-text PGM - open with any image viewer that reads PGM, "
          f"or convert with e.g. ImageMagick: magick {grid_path} centroid_grid.png)")


if __name__ == "__main__":
    main()

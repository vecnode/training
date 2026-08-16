# mnist-kmeans

K-means clustering trained **from scratch** (raw numpy - no scikit-learn) on
raw-pixel [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digits.
Unsupervised: the algorithm never sees digit labels while clustering - it
only groups 784-dimensional pixel vectors by similarity. Labels are used
afterward, only in `evaluate_kmeans.py`, to check whether the 10 clusters it
finds line up with the 10 real digit classes.

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml`, `.python-version`, no shared root environment.

## Why raw numpy, not scikit-learn

The point of this example is to see the k-means math directly: nearest-
centroid assignment, the centroid-mean update, and the cluster-to-label
matching used for scoring are all written out by hand in `train_kmeans.py`
and `evaluate_kmeans.py` rather than called from a library. numpy is used
only as a fast array container (matrix products, elementwise ops). The IDX
ubyte file format is also parsed by hand with the stdlib `struct` module in
`build_mnist_dataset.py`, rather than a dedicated MNIST-loading library.

## Dataset

Point `--data-dir` at a local folder containing the raw MNIST IDX files
(`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`,
`t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`, or the
`-idx3-ubyte`/`-idx1-ubyte` dashed-name variant) - not checked into this
repo (`data/` is gitignored via the root `.gitignore`, same as every other
pipeline). 60,000 train / 10,000 test 28x28 grayscale images, 10 digit
classes.

## Commands

```sh
uv run --directory training/mnist-kmeans python build_mnist_dataset.py --data-dir "C:\path\to\mnist-dataset" --output-dir data
uv run --directory training/mnist-kmeans python train_kmeans.py --data-path data/mnist.npz --k 10 --num-iters 50 --output-dir runs/mnist_kmeans
uv run --directory training/mnist-kmeans python evaluate_kmeans.py --data-path data/mnist.npz --centroids-path runs/mnist_kmeans/centroids.npz --output-dir runs/mnist_kmeans
```

`build_mnist_dataset.py` parses the raw IDX files and writes
`data/mnist.npz` (train/test pixel matrices scaled to `[0,1]` + labels).
`train_kmeans.py` runs k-means (nearest-centroid assignment, mean update,
repeat until assignments stop changing or `--num-iters` is hit), printing
reassignment count and inertia each iteration, and saves the learned
centroids to `runs/mnist_kmeans/centroids.npz`. `evaluate_kmeans.py` scores
the trained centroids against the held-out test split: builds a
cluster-by-digit confusion matrix, greedily matches each cluster to its
best-overlapping digit, then reports clustering accuracy and per-digit
precision/recall - and writes the 10 centroids as a viewable image grid to
`centroid_grid.pgm` so the "average digit" each cluster learned can be seen
directly.

## Files

- `build_mnist_dataset.py` - IDX ubyte parsing, pixel scaling, saves `data/mnist.npz`
- `train_kmeans.py` - centroid init, nearest-centroid assignment, mean update, convergence loop
- `evaluate_kmeans.py` - confusion matrix, greedy cluster-to-digit matching, precision/recall, PGM centroid grid

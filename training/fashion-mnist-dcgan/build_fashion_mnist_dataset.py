"""
Turn raw Fashion-MNIST files into numpy arrays ready for DCGAN training: a
flattened-then-reshaped [0,1]-scaled pixel tensor X and a label vector y
(labels are saved only so evaluate_dcgan.py can lay out the real-image grid
by class - the GAN itself never sees them).

Two input formats are accepted, both verified by exact count:

  - the original Zalando IDX ubyte files (train-images-idx3-ubyte,
    train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte),
    preferred and parsed by hand with the stdlib struct module - the same
    IDX format and parser as training/mnist-vae / training/mnist-kmeans,
    just with Fashion-MNIST's 60,000/10,000 counts;
  - the Kaggle CSVs (fashion-mnist_train.csv / fashion-mnist_test.csv),
    parsed with the stdlib csv module - no pandas, the same raw-file-by-hand
    spirit as training/adult-income-logreg.

No mnist-loading library (python-mnist, idx2numpy, kagglehub, etc.) is
used. Each split must contain exactly the published number of examples
(60,000 train / 10,000 test); a partial extraction is refused, matching the
count checks in training/imdb-sentiment-cnn and training/rvq-audio-codec.

Usage:
    uv run --directory training/fashion-mnist-dcgan python build_fashion_mnist_dataset.py \
        --data-dir "E:\\datasets\\fashionmnist" \
        --output-dir data
"""

import argparse
import csv
import os
import struct

import numpy as np

# Canonical Fashion-MNIST filenames, in the two naming conventions seen in
# the wild (Zalando's site uses "-idx3-ubyte", some mirrors ".idx3-ubyte").
IMAGE_FILE_CANDIDATES = {
    "train": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
    "test": ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte"],
}
LABEL_FILE_CANDIDATES = {
    "train": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
    "test": ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte"],
}
CSV_FILE_CANDIDATES = {
    "train": ["fashion-mnist_train.csv", "fashion-mnist-train.csv"],
    "test": ["fashion-mnist_test.csv", "fashion-mnist-test.csv"],
}
EXPECTED_COUNTS = {"train": 60000, "test": 10000}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def resolve_file(data_dir, candidates):
    """Return the first candidate filename that is an actual file (not a
    directory - dataset drops sometimes have stray directories that share a
    candidate's name)."""
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    return None


def read_idx_images(path):
    """Parse an IDX3 image file: 4-byte magic (0x00000803), then
    big-endian uint32 num_images/rows/cols, then num_images*rows*cols
    raw uint8 pixel bytes, row-major per image. Same format as MNIST."""
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"{path}: bad IDX3 magic {magic:#010x}")
        buf = f.read(num_images * rows * cols)
    images = np.frombuffer(buf, dtype=np.uint8)
    return images.reshape(num_images, rows, cols)


def read_idx_labels(path):
    """Parse an IDX1 label file: 4-byte magic (0x00000801), then a
    big-endian uint32 num_labels, then num_labels raw uint8 label bytes."""
    with open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"{path}: bad IDX1 magic {magic:#010x}")
        buf = f.read(num_labels)
    return np.frombuffer(buf, dtype=np.uint8)


def load_split_idx(data_dir, split):
    images_path = resolve_file(data_dir, IMAGE_FILE_CANDIDATES[split])
    labels_path = resolve_file(data_dir, LABEL_FILE_CANDIDATES[split])
    if images_path is None or labels_path is None:
        return None
    print(f"Reading IDX {split}: {os.path.basename(images_path)}, "
          f"{os.path.basename(labels_path)}")
    X = read_idx_images(images_path).astype(np.float32) / 255.0  # (n, 28, 28)
    y = read_idx_labels(labels_path).astype(np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: {X.shape[0]} images but {y.shape[0]} labels")
    return X, y


def load_split_csv(data_dir, split):
    """Kaggle layout: header 'label,pixel1,...,pixel784', then one row per
    image of ints 0-255 (label first, 784 pixels). Parsed with the stdlib
    csv module - no pandas."""
    csv_path = resolve_file(data_dir, CSV_FILE_CANDIDATES[split])
    if csv_path is None:
        return None
    print(f"Reading CSV {split}: {os.path.basename(csv_path)}")
    labels, pixels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header[0] != "label":
            raise ValueError(f"{csv_path}: unexpected header {header}")
        for row in reader:
            if not row:
                continue
            labels.append(int(row[0]))
            pixels.append([int(v) for v in row[1:]])
    X = np.asarray(pixels, dtype=np.float32).reshape(-1, 28, 28) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{split}: {X.shape[0]} images but {y.shape[0]} labels")
    return X, y


def load_split(data_dir, split):
    """IDX ubyte files are preferred (they are the original format); the
    Kaggle CSVs are the fallback. Either way the count is verified below."""
    loaded = load_split_idx(data_dir, split)
    if loaded is not None:
        return loaded
    loaded = load_split_csv(data_dir, split)
    if loaded is not None:
        return loaded
    raise FileNotFoundError(
        f"{split}: neither IDX ubyte files ({IMAGE_FILE_CANDIDATES[split]}) "
        f"nor Kaggle CSVs ({CSV_FILE_CANDIDATES[split]}) found under {data_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing the raw Fashion-MNIST IDX ubyte files or "
             "Kaggle CSVs.",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap each split at this many examples (smoke runs).",
    )
    args = parser.parse_args()

    X_train, y_train = load_split(args.data_dir, "train")
    X_test, y_test = load_split(args.data_dir, "test")

    for split, (X, y) in (("train", (X_train, y_train)),
                          ("test", (X_test, y_test))):
        if X.shape[0] != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split}: expected exactly {EXPECTED_COUNTS[split]} images, "
                f"got {X.shape[0]} - refusing to build on a partial "
                f"extraction (the dataset was once caught mid-download)."
            )
        if args.max_samples is not None:
            X, y = X[:args.max_samples], y[:args.max_samples]
            if split == "train":
                X_train, y_train = X, y
            else:
                X_test, y_test = X, y

    print(f"Train: {X_train.shape[0]} images, {X_train.shape[1]}x{X_train.shape[2]} each. "
          f"Test: {X_test.shape[0]} images.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "fashion_mnist.npz")
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        class_names=np.asarray(CLASS_NAMES, dtype=object),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

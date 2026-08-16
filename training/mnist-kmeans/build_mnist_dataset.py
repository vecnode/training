"""
Turn the raw MNIST IDX ubyte files into numpy arrays ready for k-means:
a flattened, [0,1]-scaled pixel matrix X and a label vector y (labels are
kept only for evaluation later - k-means itself never sees them).

No mnist-loading library (python-mnist, idx2numpy, etc.): the IDX file
format (a short big-endian header, then raw bytes) is parsed by hand with
the stdlib struct module.

Usage:
    uv run --directory training/mnist-kmeans python build_mnist_dataset.py \
        --data-dir "C:\\Users\\luisarandas\\Desktop\\mnist-dataset" \
        --output-dir data
"""

import argparse
import os
import struct

import numpy as np

# Canonical MNIST filenames, in the two naming conventions seen in the wild
# (LeCun's original site uses "-idx3-ubyte", some mirrors use ".idx3-ubyte").
IMAGE_FILE_CANDIDATES = {
    "train": ["train-images.idx3-ubyte", "train-images-idx3-ubyte"],
    "test": ["t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"],
}
LABEL_FILE_CANDIDATES = {
    "train": ["train-labels.idx1-ubyte", "train-labels-idx1-ubyte"],
    "test": ["t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"],
}


def resolve_file(data_dir, candidates):
    """Return the first candidate filename that is an actual file (not a
    directory - this particular dataset drop also has empty stray
    directories that happen to share a candidate's name)."""
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"None of {candidates} found as a file under {data_dir}"
    )


def read_idx_images(path):
    """Parse an IDX3 image file: 4-byte magic (0x00000803), then
    big-endian uint32 num_images/rows/cols, then num_images*rows*cols
    raw uint8 pixel bytes, row-major per image."""
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"{path}: bad IDX3 magic {magic:#010x}")
        buf = f.read(num_images * rows * cols)
    images = np.frombuffer(buf, dtype=np.uint8)
    return images.reshape(num_images, rows * cols)


def read_idx_labels(path):
    """Parse an IDX1 label file: 4-byte magic (0x00000801), then a
    big-endian uint32 num_labels, then num_labels raw uint8 label bytes."""
    with open(path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"{path}: bad IDX1 magic {magic:#010x}")
        buf = f.read(num_labels)
    return np.frombuffer(buf, dtype=np.uint8)


def load_split(data_dir, split):
    images_path = resolve_file(data_dir, IMAGE_FILE_CANDIDATES[split])
    labels_path = resolve_file(data_dir, LABEL_FILE_CANDIDATES[split])
    print(f"Reading {images_path}")
    X = read_idx_images(images_path).astype(np.float64) / 255.0
    print(f"Reading {labels_path}")
    y = read_idx_labels(labels_path).astype(np.int64)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{split}: {X.shape[0]} images but {y.shape[0]} labels"
        )
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing the raw MNIST IDX ubyte files.",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    X_train, y_train = load_split(args.data_dir, "train")
    X_test, y_test = load_split(args.data_dir, "test")

    print(f"Train: {X_train.shape[0]} images, {X_train.shape[1]} pixels each "
          f"(28x28). Test: {X_test.shape[0]} images.")
    print(f"Label counts train: {np.bincount(y_train)}")
    print(f"Label counts test:  {np.bincount(y_test)}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mnist.npz")
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

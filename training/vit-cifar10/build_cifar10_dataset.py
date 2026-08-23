"""
Turn the raw CIFAR-10 python-format pickle batches into numpy arrays ready
for ViT training: a (n, 3, 32, 32) [0,1]-scaled float32 tensor X and an
int64 label vector y per split, plus the 10 class names. Same data
contract as training/cifar10-vqvae (and the same parser): this pipeline
normalizes per-channel in the trainer (hardcoded train-set statistics),
so the .npz keeps raw [0,1] pixels like the sibling.

No cifar-loading library (no torchvision, no keras.datasets): each batch
file is a plain pickle of a dict with b'data' (10000 x 3072 uint8, row-
major RGB - 1024 red bytes, then 1024 green, then 1024 blue, i.e. already
planar, so a plain reshape gives (n, 3, 32, 32)) and b'labels' (list of
10000 ints); batches.meta carries the class names. The files are
Python-2 pickles, so they are loaded with pickle.load(f, encoding='bytes')
and the bytes keys/values are decoded by hand. This is a standalone
parser - each uv project here is independent and doesn't import another
pipeline folder's code.

Usage:
    uv run --directory training/vit-cifar10 python build_cifar10_dataset.py \
        --data-dir "C:\\path\\to\\cifar-10-python" \
        --output-dir data
"""

import argparse
import os
import pickle

import numpy as np

TRAIN_BATCH_NAMES = [f"data_batch_{i}" for i in range(1, 6)]
TEST_BATCH_NAME = "test_batch"
META_NAME = "batches.meta"


def resolve_batch_dir(data_dir):
    """CIFAR-10 python drops come two ways: the batch files directly in
    --data-dir, or nested under a cifar-10-batches-py/ subfolder. Accept
    either."""
    if os.path.isfile(os.path.join(data_dir, TRAIN_BATCH_NAMES[0])):
        return data_dir
    nested = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isfile(os.path.join(nested, TRAIN_BATCH_NAMES[0])):
        return nested
    raise FileNotFoundError(
        f"Could not find CIFAR-10 batch files under {data_dir} "
        f"(looked for {TRAIN_BATCH_NAMES[0]} directly or under a "
        f"cifar-10-batches-py subfolder)"
    )


def load_batch(path, max_samples=0):
    """Parse one Python-2 pickle batch file into (X, y). X is
    (n, 3, 32, 32) uint8 - the raw 3072-byte vectors are already planar
    R/G/B - and y is int64."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32)
    y = np.asarray(d[b"labels"], dtype=np.int64)
    if max_samples and X.shape[0] > max_samples:
        X, y = X[:max_samples], y[:max_samples]
    return X, y


def load_meta(batch_dir):
    with open(os.path.join(batch_dir, META_NAME), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    names = [name.decode("utf-8") for name in meta[b"label_names"]]
    if len(names) != 10:
        raise ValueError(f"batches.meta has {len(names)} classes, expected 10")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=r"C:\path\to\cifar-10-python",
        help="Directory containing the raw CIFAR-10 python-format pickle "
             "files (a cifar-10-batches-py subfolder is also accepted).",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="If >0, cap each split to this many images (quick smoke run "
             "before committing to the full 60k-image dataset).",
    )
    args = parser.parse_args()

    batch_dir = resolve_batch_dir(args.data_dir)
    label_names = load_meta(batch_dir)
    print(f"Classes: {', '.join(label_names)}")

    parts = []
    for name in TRAIN_BATCH_NAMES:
        path = os.path.join(batch_dir, name)
        print(f"Reading {path}")
        parts.append(load_batch(path, args.max_samples))
    X_train = np.concatenate([p[0] for p in parts])
    y_train = np.concatenate([p[1] for p in parts])
    print(f"Train: {X_train.shape[0]} images ({X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]} each)")

    test_path = os.path.join(batch_dir, TEST_BATCH_NAME)
    print(f"Reading {test_path}")
    X_test, y_test = load_batch(test_path, args.max_samples)
    print(f"Test: {X_test.shape[0]} images")

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cifar10.npz")
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        label_names=np.asarray(label_names),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

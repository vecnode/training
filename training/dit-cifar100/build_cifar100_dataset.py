"""
Turn the raw CIFAR-100 python-format pickles into numpy arrays ready for
class-conditional DiT training: a (n, 3, 32, 32) [0,1]-scaled float32 tensor
X and, per split, the fine (100-class) and coarse (20-superclass) label
vectors, plus both name lists. Same data contract as
training/mae-cifar100 (and the CIFAR-10 twins training/vit-cifar10 and
training/cifar10-vqvae): the trainer keeps raw [0,1] pixels and rescales to
[-1,1] itself (like training/flow-matching-mnist), so the .npz stores raw
pixels like the siblings.

No cifar-loading library (no torchvision, no keras.datasets): each file is
a plain Python-2 pickle of a dict with b'data' (n x 3072 uint8, row-major
RGB - 1024 red bytes, then 1024 green, then 1024 blue, i.e. already planar,
so a plain reshape gives (n, 3, 32, 32)), b'fine_labels' (100 classes) and
b'coarse_labels' (20 superclasses); the meta file carries both name lists.
Unlike CIFAR-10's five train batches, CIFAR-100 ships a single 50k-image
train file and a single 10k test file, both verified by exact count so a
partial/corrupt extraction refuses to build. This is a standalone parser -
each uv project here is independent and doesn't import another pipeline
folder's code.

Usage:
    uv run --directory training/dit-cifar100 python build_cifar100_dataset.py \
        --data-dir "E:\\datasets\\cifar-100-python" \
        --output-dir data
"""

import argparse
import os
import pickle

import numpy as np

TRAIN_NAME = "train"
TEST_NAME = "test"
META_NAME = "meta"
EXPECTED_TRAIN = 50_000
EXPECTED_TEST = 10_000


def resolve_data_dir(data_dir):
    """CIFAR-100 python drops come two ways: the pickle files directly in
    --data-dir, or nested under a cifar-100-python/ subfolder (the layout
    produced by extracting the tarball). Accept either."""
    if os.path.isfile(os.path.join(data_dir, TRAIN_NAME)):
        return data_dir
    nested = os.path.join(data_dir, "cifar-100-python")
    if os.path.isfile(os.path.join(nested, TRAIN_NAME)):
        return nested
    raise FileNotFoundError(
        f"Could not find CIFAR-100 pickle files under {data_dir} "
        f"(looked for {TRAIN_NAME} directly or under a "
        f"cifar-100-python subfolder)"
    )


def load_split(path, expected, max_samples=0):
    """Parse one Python-2 pickle split file into (X, y_fine, y_coarse).
    X is (n, 3, 32, 32) uint8 - the raw 3072-byte vectors are already
    planar R/G/B - and the label vectors are int64."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    n = d[b"data"].shape[0]
    if n != expected:
        raise ValueError(
            f"{path} has {n} images, expected exactly {expected} - "
            f"refusing to build on a partial or corrupt extraction"
        )
    X = d[b"data"].reshape(-1, 3, 32, 32)
    y_fine = np.asarray(d[b"fine_labels"], dtype=np.int64)
    y_coarse = np.asarray(d[b"coarse_labels"], dtype=np.int64)
    if max_samples and X.shape[0] > max_samples:
        X, y_fine, y_coarse = X[:max_samples], y_fine[:max_samples], y_coarse[:max_samples]
    return X, y_fine, y_coarse


def load_meta(data_dir):
    with open(os.path.join(data_dir, META_NAME), "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    fine_names = [name.decode("utf-8") for name in meta[b"fine_label_names"]]
    coarse_names = [name.decode("utf-8") for name in meta[b"coarse_label_names"]]
    if len(fine_names) != 100 or len(coarse_names) != 20:
        raise ValueError(
            f"meta has {len(fine_names)} fine / {len(coarse_names)} coarse "
            f"names, expected 100/20"
        )
    return fine_names, coarse_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=r"E:\datasets\cifar-100-python",
        help="Directory containing the raw CIFAR-100 python-format pickle "
             "files (meta, train, test; a cifar-100-python subfolder is "
             "also accepted).",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="If >0, cap each split to this many images (quick smoke run "
             "before committing to the full 60k-image dataset).",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    fine_names, coarse_names = load_meta(data_dir)
    print(f"Fine classes ({len(fine_names)}): {', '.join(fine_names[:12])} ...")
    print(f"Coarse classes ({len(coarse_names)}): {', '.join(coarse_names)}")

    train_path = os.path.join(data_dir, TRAIN_NAME)
    print(f"Reading {train_path}")
    X_train, y_train, y_coarse_train = load_split(
        train_path, EXPECTED_TRAIN, args.max_samples
    )
    print(f"Train: {X_train.shape[0]} images ({X_train.shape[1]}x"
          f"{X_train.shape[2]}x{X_train.shape[3]} each)")

    test_path = os.path.join(data_dir, TEST_NAME)
    print(f"Reading {test_path}")
    X_test, y_test, y_coarse_test = load_split(
        test_path, EXPECTED_TEST, args.max_samples
    )
    print(f"Test: {X_test.shape[0]} images")

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cifar100.npz")
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train, y_coarse_train=y_coarse_train,
        X_test=X_test, y_test=y_test, y_coarse_test=y_coarse_test,
        fine_label_names=np.asarray(fine_names),
        coarse_label_names=np.asarray(coarse_names),
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""
Turn the raw UCI Adult / Census Income files (adult.data, adult.test) into
numpy arrays ready for logistic regression: a numeric feature matrix X and a
binary label vector y (1 = ">50K", 0 = "<=50K").

No pandas: files are read and parsed line-by-line with the stdlib csv
module, and every encoding/scaling step below is written out explicitly so
it is clear exactly what each column turns into.

Usage:
    uv run --directory training/adult-income-logreg python build_income_dataset.py \
        --data-dir "C:\\path\\to\\adult" \
        --output-dir data
"""

import argparse
import csv
import os

import numpy as np

# Column order in adult.data / adult.test, per adult.names. The final column
# is the label (">50K" / "<=50K", with a trailing "." in adult.test only).
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "label",
]

CONTINUOUS_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


def read_rows(path, is_test_file):
    """Parse one adult.data/adult.test file into a list of dicts.

    Rows with any "?" (missing value) are dropped, matching the cleaned
    45,222-row variant described in adult.names. adult.test also has a
    leading "|1x3 Cross validator" comment line and a trailing "." on every
    label ("<=50K." instead of "<=50K") - both handled here.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for raw in reader:
            if not raw or len(raw) < len(COLUMNS):
                continue
            if raw[0].startswith("|"):
                continue  # the "|1x3 Cross validator" header line in adult.test
            values = raw[: len(COLUMNS)]
            if any(v == "?" for v in values):
                continue
            row = dict(zip(COLUMNS, values))
            if is_test_file:
                row["label"] = row["label"].rstrip(".")
            rows.append(row)
    return rows


def fit_categorical_vocab(rows, column):
    """Collect the sorted set of distinct values a categorical column takes
    in the *training* rows only - the test set must reuse this vocabulary
    (never fit encoders on test data)."""
    return sorted({row[column] for row in rows})


def one_hot(value, vocab):
    """Manual one-hot encoding: a length-len(vocab) vector with a single 1
    at the index of `value`. Unseen categories (can happen in principle for
    native-country) map to the all-zero vector."""
    vec = [0.0] * len(vocab)
    if value in vocab:
        vec[vocab.index(value)] = 1.0
    return vec


def build_feature_matrix(rows, vocabs, cont_mean, cont_std):
    """Assemble the final numeric feature matrix for a set of rows.

    Each row becomes: [z-scored continuous features] + [one-hot blocks for
    every categorical column, concatenated in CATEGORICAL_COLUMNS order].
    z-scoring (x - mean) / std uses statistics computed on the *training*
    set (cont_mean/cont_std), applied identically to train and test, so
    training data never leaks through test-set-derived statistics.
    """
    feature_names = []
    for c in CONTINUOUS_COLUMNS:
        feature_names.append(f"{c}_z")
    for c in CATEGORICAL_COLUMNS:
        for value in vocabs[c]:
            feature_names.append(f"{c}={value}")

    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        col = 0
        for j, c in enumerate(CONTINUOUS_COLUMNS):
            raw_value = float(row[c])
            X[i, col] = (raw_value - cont_mean[j]) / cont_std[j]
            col += 1
        for c in CATEGORICAL_COLUMNS:
            vec = one_hot(row[c], vocabs[c])
            X[i, col : col + len(vec)] = vec
            col += len(vec)
        y[i] = 1.0 if row["label"] == ">50K" else 0.0

    return X, y, feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing the raw adult.data / adult.test files "
             "(https://archive.ics.uci.edu/dataset/2/adult).",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "adult.data")
    test_path = os.path.join(args.data_dir, "adult.test")

    print(f"Reading {train_path}")
    train_rows = read_rows(train_path, is_test_file=False)
    print(f"Reading {test_path}")
    test_rows = read_rows(test_path, is_test_file=True)
    print(f"Kept {len(train_rows)} train rows, {len(test_rows)} test rows "
          f"after dropping rows with missing '?' values.")

    # Fit categorical vocabularies and continuous mean/std on TRAIN rows only.
    vocabs = {c: fit_categorical_vocab(train_rows, c) for c in CATEGORICAL_COLUMNS}

    cont_values = np.array(
        [[float(row[c]) for c in CONTINUOUS_COLUMNS] for row in train_rows],
        dtype=np.float64,
    )
    cont_mean = cont_values.mean(axis=0)
    cont_std = cont_values.std(axis=0)
    cont_std[cont_std == 0] = 1.0  # guard against a constant column

    X_train, y_train, feature_names = build_feature_matrix(
        train_rows, vocabs, cont_mean, cont_std
    )
    X_test, y_test, _ = build_feature_matrix(
        test_rows, vocabs, cont_mean, cont_std
    )

    print(f"Feature matrix: {X_train.shape[1]} columns "
          f"({len(CONTINUOUS_COLUMNS)} continuous z-scored + "
          f"{X_train.shape[1] - len(CONTINUOUS_COLUMNS)} one-hot).")
    print(f"Positive rate (>50K): train={y_train.mean():.3f}  test={y_test.mean():.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "adult_income.npz")
    np.savez(
        out_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(feature_names),
        cont_mean=cont_mean, cont_std=cont_std,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

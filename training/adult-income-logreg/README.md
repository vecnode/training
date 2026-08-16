# adult-income-logreg

Logistic regression trained **from scratch** (raw numpy - no scikit-learn,
no pandas) on the UCI [Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult)
dataset. Binary classification: does this person earn more than $50K/year?

This is a `training/` pipeline (from-scratch, non-LoRA), independent `uv`
project like every other pipeline folder in this repo - own
`pyproject.toml`, `.python-version`, no shared root environment.

## Why raw numpy, not scikit-learn

The point of this example is to see the logistic regression math directly:
the sigmoid, the cross-entropy loss, the gradient derivation, and the
gradient-descent update loop are all written out by hand in
[`train_logreg.py`](train_logreg.py) rather than called from a library.
numpy is used only as a fast array container (dot products, elementwise
ops) - the same role a hand-written dot product or elementwise loop would
play, just vectorized. Same reasoning for the dataset build: no pandas, the
raw CSV files are parsed with the stdlib `csv` module and one-hot/z-score
encoding is done by hand in [`build_income_dataset.py`](build_income_dataset.py)
so every column transformation is visible.

## Dataset

Download the raw files from
[archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)
(`adult.data`, `adult.test`, `adult.names`) into a local folder - they are
**not** checked into this repo (`data/` is gitignored via the root
`.gitignore`, same as every other pipeline).

14 features (6 continuous, 8 categorical) + a binary label:

| Type | Columns |
|---|---|
| Continuous | `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical | `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` |
| Label | `>50K` / `<=50K` |

Rows with a missing value (`"?"`) are dropped, matching the cleaned
45,222-row variant documented in `adult.names` (32,561 train / 16,281 test
before cleaning). Continuous columns are z-scored `(x - mean) / std` using
train-set statistics only; categorical columns are one-hot encoded using a
vocabulary built from the train set only - both applied identically to the
test set so no test-set information leaks into preprocessing.

## Commands

```sh
uv run --directory training/adult-income-logreg python build_income_dataset.py --data-dir "C:\path\to\adult" --output-dir data
uv run --directory training/adult-income-logreg python train_logreg.py --data-path data/adult_income.npz --num-epochs 300 --output-dir runs/adult_logreg
uv run --directory training/adult-income-logreg python evaluate_logreg.py --data-path data/adult_income.npz --weights-path runs/adult_logreg/logreg_weights.npz
```

`build_income_dataset.py` writes `data/adult_income.npz` (train/test
feature matrices + labels + feature names + normalization stats).
`train_logreg.py` runs batch gradient descent on binary cross-entropy (with
L2 weight decay), printing train/val loss and accuracy every `--log-every`
epochs, and saves the learned `(w, b)` to `runs/adult_logreg/logreg_weights.npz`.
`evaluate_logreg.py` scores the trained model against the fully held-out
`adult.test` split: confusion matrix, accuracy/precision/recall/F1, and the
top features by `|weight|`.

Reference point for this dataset: published results on the cleaned
train/test split cluster around 84-86% accuracy for tree-based methods
(C4.5, NBTree); a from-scratch logistic regression here is expected to land
somewhat below that (linear decision boundary vs. tree splits), a useful
sanity check that training is actually working rather than a target to beat.

## Files

- `build_income_dataset.py` - raw-file parsing, one-hot/z-score encoding, saves `data/adult_income.npz`
- `train_logreg.py` - sigmoid, cross-entropy loss, gradient derivation, gradient-descent training loop
- `evaluate_logreg.py` - confusion matrix, precision/recall/F1, top-weighted features

"""
Evaluate a trained logistic regression model against the held-out
adult.test split, and inspect which features it weighted most heavily.

All metrics (confusion matrix, precision/recall/F1) are computed by hand
from the raw prediction/label arrays - no scikit-learn.metrics.

Usage:
    uv run --directory training/adult-income-logreg python evaluate_logreg.py \
        --data-path data/adult_income.npz \
        --weights-path runs/adult_logreg/logreg_weights.npz
"""

import argparse

import numpy as np

from train_logreg import predict_proba


def confusion_counts(y_true, y_pred):
    """Manual confusion-matrix counts for binary labels in {0, 1}."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return tp, tn, fp, fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/adult_income.npz")
    parser.add_argument("--weights-path", default="runs/adult_logreg/logreg_weights.npz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-features", type=int, default=15)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"]

    weights = np.load(args.weights_path, allow_pickle=True)
    w, b, feature_names = weights["w"], float(weights["b"]), weights["feature_names"]

    proba = predict_proba(X_test, w, b)
    y_pred = (proba >= args.threshold).astype(np.float64)

    tp, tn, fp, fn = confusion_counts(y_test, y_pred)
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Test rows: {len(y_test)}  (positive rate: {y_test.mean():.3f})")
    print()
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"              pred <=50K   pred >50K")
    print(f"  actual <=50K   {tn:8d}   {fp:8d}")
    print(f"  actual  >50K   {fn:8d}   {tp:8d}")
    print()
    print(f"accuracy  = {accuracy:.4f}")
    print(f"precision = {precision:.4f}  (of predicted >50K, how many really are)")
    print(f"recall    = {recall:.4f}  (of actual >50K, how many were caught)")
    print(f"f1        = {f1:.4f}")

    # Because the continuous features were z-scored and categorical features
    # are one-hot (both on a comparable 0/1-ish scale), the raw weight
    # magnitude is a reasonable proxy for a feature's influence on the
    # log-odds of earning >50K - larger |w_i| moves the prediction more per
    # unit of that (scaled) feature.
    order = np.argsort(-np.abs(w))[: args.top_features]
    print()
    print(f"Top {args.top_features} features by |weight| (sign shows direction):")
    for idx in order:
        print(f"  {feature_names[idx]:35s}  w = {w[idx]:+.4f}")


if __name__ == "__main__":
    main()

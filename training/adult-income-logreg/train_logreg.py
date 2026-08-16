"""
Logistic regression, implemented by hand with numpy - no scikit-learn.

The model:  P(y=1 | x) = sigmoid(w . x + b)
The loss:   binary cross-entropy, averaged over the batch, plus L2 weight decay
The fit:    plain batch gradient descent on that loss

Every step below (forward pass, loss, gradient, update rule) is written out
explicitly rather than called from a library, so the math is visible end to
end. numpy is used purely as a fast array container (dot products,
elementwise ops), the same way you'd write this on paper with vectors.

Usage:
    uv run --directory training/adult-income-logreg python train_logreg.py \
        --data-path data/adult_income.npz \
        --num-epochs 300 \
        --output-dir runs/adult_logreg
"""

import argparse
import os

import numpy as np


def sigmoid(z):
    """sigmoid(z) = 1 / (1 + e^-z), squashes any real number into (0, 1) so
    it can be read as a probability. Clipping z avoids overflow in exp() for
    very large |z| early in training (a purely numerical-stability guard,
    doesn't change the math).
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w, b):
    """Forward pass: linear score z = X.w + b, then squash through sigmoid."""
    z = X @ w + b
    return sigmoid(z)


def binary_cross_entropy(y_true, y_pred, w, l2_lambda):
    """
    Binary cross-entropy for one example: -[y*log(p) + (1-y)*log(1-p)].
    Averaged over the batch, plus an L2 penalty (l2_lambda/2 * ||w||^2) that
    discourages large weights and reduces overfitting - the bias term b is
    deliberately excluded from the penalty (standard practice: regularizing
    b would just bias the model's baseline prediction, not its shape).

    eps guards log(0) when a prediction saturates to exactly 0 or 1.
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    data_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    reg_loss = (l2_lambda / 2.0) * np.sum(w * w)
    return data_loss + reg_loss


def gradients(X, y_true, y_pred, w, l2_lambda):
    """
    Gradient of the cross-entropy loss w.r.t. (w, b).

    Because the loss is cross-entropy and the model is a sigmoid, the
    derivative works out to the same clean form as linear regression's
    squared-error gradient - this cancellation is *why* cross-entropy is
    the standard loss for logistic regression rather than squared error:

        dL/dz  = (y_pred - y_true)                       [per example]
        dL/dw  = (1/n) * X^T . (y_pred - y_true)  + l2_lambda * w
        dL/db  = (1/n) * sum(y_pred - y_true)
    """
    n = X.shape[0]
    error = y_pred - y_true                    # shape (n,)
    dw = (X.T @ error) / n + l2_lambda * w      # shape (num_features,)
    db = np.mean(error)                         # scalar
    return dw, db


def accuracy(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(np.float64)
    return np.mean(y_pred == y_true)


def train(X_train, y_train, X_val, y_val, num_epochs, learning_rate, l2_lambda, log_every):
    num_features = X_train.shape[1]
    # Zero init is fine here: logistic regression's loss is convex, so
    # gradient descent converges to the same optimum regardless of the
    # (reasonable) starting point - unlike deep nets, there's no
    # symmetry-breaking reason to randomize.
    w = np.zeros(num_features, dtype=np.float64)
    b = 0.0

    for epoch in range(1, num_epochs + 1):
        y_pred = predict_proba(X_train, w, b)
        dw, db = gradients(X_train, y_train, y_pred, w, l2_lambda)

        # Gradient descent update: step opposite the gradient, scaled by
        # the learning rate.
        w -= learning_rate * dw
        b -= learning_rate * db

        if epoch % log_every == 0 or epoch == num_epochs:
            train_loss = binary_cross_entropy(y_train, y_pred, w, l2_lambda)
            train_acc = accuracy(y_train, y_pred)
            val_pred = predict_proba(X_val, w, b)
            val_loss = binary_cross_entropy(y_val, val_pred, w, l2_lambda)
            val_acc = accuracy(y_val, val_pred)
            print(
                f"epoch {epoch:4d}/{num_epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    return w, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/adult_income.npz")
    parser.add_argument("--output-dir", default="runs/adult_logreg")
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2-lambda", type=float, default=1e-4,
                         help="L2 regularization strength (0 disables it).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                         help="Fraction of the training rows held out as a "
                              "validation split (adult.test stays fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Carve a validation split out of the training rows so we can watch for
    # overfitting during training; adult.test is reserved for the final,
    # untouched-until-the-end evaluation in evaluate_logreg.py.
    rng = np.random.default_rng(args.seed)
    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"Held-out test rows: {X_test.shape[0]}  Features: {X_train.shape[1]}")

    w, b = train(
        X_train, y_train, X_val, y_val,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        l2_lambda=args.l2_lambda,
        log_every=args.log_every,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "logreg_weights.npz")
    np.savez(out_path, w=w, b=b, feature_names=data["feature_names"])
    print(f"Saved trained weights to {out_path}")
    print("Run evaluate_logreg.py against the held-out adult.test split next.")


if __name__ == "__main__":
    main()

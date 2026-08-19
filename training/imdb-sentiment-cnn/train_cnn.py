"""
A Text CNN for document classification ([Kim, 2014](https://arxiv.org/abs/1408.5882))
trained **from scratch** on the Large Movie Review Dataset - no NLP library.
The embedding layer and the multi-width 1D convolutions are written out by
hand as plain torch modules; torch is used only for tensor ops, autograd,
and GPU execution, the same role it plays everywhere else in this repo
(no torchtext, no DataLoader, numpy-permutation batching).

The model:
    Embedding(vocab, D)               - trained from scratch, random init
        (D=128 by default; no GloVe - strictly IMDB-only data by design)
    For each filter width w in {3, 4, 5}:
        Conv1d(D -> F, kernel=w) -> ReLU -> 1-max-pool over the time axis
    Concatenate the 3F pooled scalars -> Dropout(0.5) -> Linear(3F -> 2)

Each filter width scans the whole review for w-grams; max-pooling over
time keeps only each filter's strongest activation, so the classifier sees
a fixed-size vector no matter how long the review is. This is the classic
"CNN-rand" setup: embeddings are not pretrained, everything is learned
from the 25k labeled reviews.

Loss: cross-entropy over the 2 classes (pos/neg). Optimizer: Adam with a
cosine-annealed learning rate (the same schedule shape as
training/cifar10-vqvae). A 10% holdout of the training split is used for
validation / best-checkpoint selection; the official 25k test split stays
fully unseen until evaluate_cnn.py.

Usage:
    uv run --directory training/imdb-sentiment-cnn python train_cnn.py \
        --data-path data/imdb.npz \
        --num-epochs 20 \
        --batch-size 128 \
        --output-dir runs/imdb_cnn
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Embedding -> per-width 1D conv + ReLU + 1-max-pool -> concat ->
    dropout -> linear. Input: (B, L) int64 token ids (0 = pad)."""

    def __init__(self, vocab_size, embedding_dim=128, num_filters=128,
                 filter_widths=(3, 4, 5), dropout=0.5, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.embedding.weight[0].zero_()  # pad row stays zero
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=w)
            for w in filter_widths
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_widths), num_classes)

    def forward(self, x):
        emb = self.embedding(x)            # (B, L, D)
        emb = emb.transpose(1, 2)          # (B, D, L)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))          # (B, F, L - w + 1)
            pooled.append(h.max(dim=2).values)  # (B, F), 1-max-pool
        h = torch.cat(pooled, dim=1)       # (B, 3F)
        h = self.dropout(h)
        return self.fc(h)


def iterate_batches(X, y, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - one permutation
    per epoch, sliced into batches, so the batching logic stays visible."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def run_epoch(model, X, y, batch_size, device, rng, optimizer=None):
    """One pass over (X, y). If optimizer is given, trains (shuffled, grad
    updates); otherwise evaluates (no shuffle, no grad) - shared loop so
    train/val accounting can't drift apart. Returns (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    n_seen = 0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, yb in iterate_batches(X, y, batch_size, rng, shuffle=training):
            x = torch.from_numpy(xb).to(device)
            t = torch.from_numpy(yb).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, t)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == t).sum().item()
            n_seen += t.numel()
            n_batches += 1

    return total_loss / n_batches, total_correct / n_seen


def parse_filter_widths(s):
    try:
        widths = tuple(int(w) for w in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"filter widths must be comma-separated ints, got {s!r}")
    if not widths or any(w < 1 for w in widths):
        raise argparse.ArgumentTypeError("filter widths must be >= 1")
    return widths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/imdb.npz")
    parser.add_argument("--output-dir", default="runs/imdb_cnn")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--filter-widths", type=parse_filter_widths,
                        default="3,4,5")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training rows held out for "
                             "validation (the official 25k test split stays "
                             "fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_train_full = data["X_train"]  # (n, max_len) int32 token ids
    y_train_full = data["y_train"]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"seq len: {X_train.shape[1]}  emb={args.embedding_dim}  "
          f"filters={args.num_filters}x{args.filter_widths}  "
          f"dropout={args.dropout}")

    model = TextCNN(
        vocab_size=int(data["vocab_size"]),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        filter_widths=args.filter_widths,
        dropout=args.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} "
          f"({int(data['vocab_size']) * args.embedding_dim:,} of them in "
          f"the embedding layer)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = -1.0
    best_path = os.path.join(args.output_dir, "cnn_best.pt")
    final_path = os.path.join(args.output_dir, "cnn_final.pt")
    ckpt_meta = {
        "arch": "text_cnn",
        "vocab_size": int(data["vocab_size"]),
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "filter_widths": args.filter_widths,
        "dropout": args.dropout,
        "max_len": int(data["max_len"]),
    }

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, X_train, y_train, args.batch_size, device, rng,
            optimizer=optimizer,
        )
        scheduler.step()

        val_loss, val_acc = run_epoch(
            model, X_val, y_val, args.batch_size, device, rng, optimizer=None
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            elapsed = time.time() - start_time
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: loss={train_loss:.4f} acc={train_acc:.4f}  "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.0f}s elapsed)"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(),
                 "epoch": epoch, "val_acc": val_acc,
                 **ckpt_meta},
                best_path,
            )

    torch.save(
        {"model_state": model.state_dict(),
         "epoch": args.num_epochs, "val_acc": val_acc,
         **ckpt_meta},
        final_path,
    )
    print(f"Saved best checkpoint (val_acc={best_val_acc:.4f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_cnn.py against the held-out test split next.")


if __name__ == "__main__":
    main()

"""
Evaluate a trained Text-CNN checkpoint against the held-out 25k-review
IMDB test split, and show what the model actually gets right and wrong:

  - overall accuracy on the full test split (pos/neg balanced 12.5k each)
  - per-class (pos/neg) accuracy + confusion matrix
  - test_metrics.txt written to --output-dir with the same numbers
  - a handful of example reviews (reconstructed from token ids via
    vocab.txt), first correct and first wrong, with true label, predicted
    label, and confidence - plain text output, no plotting library (same
    no-Pillow/no-matplotlib ethos as the rest of training/).

Usage:
    uv run --directory training/imdb-sentiment-cnn python evaluate_cnn.py \
        --data-path data/imdb.npz \
        --checkpoint-path runs/imdb_cnn/cnn_best.pt \
        --vocab-path data/vocab.txt \
        --output-dir runs/imdb_cnn
"""

import argparse
import os

import numpy as np
import torch

from train_cnn import TextCNN, run_epoch

CLASS_NAMES = {0: "neg", 1: "pos"}


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def reconstruct(vocab, ids, max_chars=280):
    """Token ids -> display text (strip padding; <unk> shown literally)."""
    words = []
    for tid in ids:
        if tid == 0:
            continue  # padding
        words.append(vocab[tid])
    text = " ".join(words)
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/imdb.npz")
    parser.add_argument("--checkpoint-path",
                        default="runs/imdb_cnn/cnn_best.pt")
    parser.add_argument("--vocab-path", default="data/vocab.txt")
    parser.add_argument("--output-dir", default="runs/imdb_cnn")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=10,
                        help="How many example reviews to print (half "
                             "correct, half misclassified).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_test, y_test = data["X_test"], data["y_test"]

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    if ckpt.get("arch") != "text_cnn":
        raise SystemExit(f"{args.checkpoint_path} is not a text_cnn "
                         f"checkpoint (arch={ckpt.get('arch')!r})")
    model = TextCNN(
        vocab_size=ckpt["vocab_size"],
        embedding_dim=ckpt["embedding_dim"],
        num_filters=ckpt["num_filters"],
        filter_widths=tuple(ckpt["filter_widths"]),
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.checkpoint_path} (epoch {ckpt['epoch']}, "
          f"val_acc {ckpt['val_acc']:.4f})")

    vocab = load_vocab(args.vocab_path)
    print(f"Vocab: {len(vocab)} entries")

    # full test-split accounting (no shuffle, so predictions line up with
    # the stored row order for the example picker)
    _, test_acc = run_epoch(model, X_test, y_test, args.batch_size, device,
                            rng, optimizer=None)

    # per-class accuracy + confusion matrix over the whole test set
    logits = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], args.batch_size):
            x = torch.from_numpy(X_test[start:start + args.batch_size]).to(device)
            logits.append(model(x))
    logits = torch.cat(logits)
    preds = logits.argmax(1).cpu().numpy()
    conf = np.zeros((2, 2), dtype=np.int64)  # rows true, cols pred
    for t in range(2):
        mask = y_test == t
        for p in range(2):
            conf[t, p] = int((preds[mask] == p).sum())

    print()
    print(f"Test accuracy: {test_acc:.4f}  "
          f"({int(round(test_acc * y_test.size))}/{y_test.size} correct)")
    print(f"Per-class: neg {conf[0, 0] / conf[0].sum():.4f}  "
          f"pos {conf[1, 1] / conf[1].sum():.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"           pred_neg  pred_pos")
    print(f"  true_neg  {conf[0, 0]:7d}  {conf[0, 1]:7d}")
    print(f"  true_pos  {conf[1, 0]:7d}  {conf[1, 1]:7d}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint_path}\n")
        f.write(f"test_accuracy: {test_acc:.4f}\n")
        f.write(f"correct: {int(round(test_acc * y_test.size))}/{y_test.size}\n")
        f.write(f"neg_accuracy: {conf[0, 0] / conf[0].sum():.4f}\n")
        f.write(f"pos_accuracy: {conf[1, 1] / conf[1].sum():.4f}\n")
        f.write("confusion_matrix (rows=true, cols=pred):\n")
        f.write(f"  true_neg: {conf[0, 0]} {conf[0, 1]}\n")
        f.write(f"  true_pos: {conf[1, 0]} {conf[1, 1]}\n")
    print(f"\nWrote {metrics_path}")

    # example reviews: first half correct, first half misclassified, in
    # stored test order
    correct_idx = np.where(preds == y_test)[0]
    wrong_idx = np.where(preds != y_test)[0]
    n_ex = args.num_examples // 2

    def print_examples(idx_list, heading):
        print(f"\n=== {heading} ===")
        for i in idx_list[:n_ex]:
            probs = torch.softmax(logits[i], dim=0)
            conf_pct = float(probs.max()) * 100.0
            mark = "correct" if preds[i] == y_test[i] else "WRONG"
            print(f"\n[{mark}] true={CLASS_NAMES[int(y_test[i])]} "
                  f"pred={CLASS_NAMES[int(preds[i])]} "
                  f"(conf {conf_pct:.1f}%)")
            print(reconstruct(vocab, X_test[i]))

    print_examples(correct_idx, f"first {n_ex} correct predictions")
    print_examples(wrong_idx, f"first {n_ex} misclassifications")


if __name__ == "__main__":
    main()

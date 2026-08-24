"""
The judge for a trained MAE checkpoint: a hand-written **linear probe** - a
single linear layer trained **from scratch** on the MAE encoder's frozen
features, then scored on the held-out 10k CIFAR-100 test split. The probe
is this pipeline's whole point as an evaluation: the features are the
model's own (learned purely from masked-patch reconstruction, no labels in
pretraining), so it measures representation quality without violating this
repo's no-pretrained-features rule the way a FID/Inception score would.

Protocol (the paper's linear-probe convention, hand-written):

  1. Load the MAE checkpoint, rebuild the full model, freeze everything.
  2. Precompute features once: all patches (no masking), full block stack,
     final norm, then **mean-pool the patch tokens** (MAE has no CLS
     token, so mean pooling is the probe representation), inputs
     normalized with the hardcoded CIFAR-100 train statistics. Features
     are cached to CPU arrays, so the probe epochs train on cheap
     (50k x 384) / (10k x 384) matrices, not through the network.
     (Offline features mean no probe-time flip+crop - a documented
     simplification: the probe sees each image exactly once, normalized.)
  3. Train the linear head (dim -> 100) with hand-written SGD momentum +
     cosine LR - the paper's probe optimizer, not AdamW.
  4. Report test top-1/top-5, coarse-label (20 superclass) top-1,
     per-class accuracy + 100x100 confusion matrix, and write
     test_metrics.txt, probe_head.pt, and a hand-written zlib RGB
     predictions grid (first 16 correct, first 16 misclassified, green/red
     borders - no Pillow).

Usage:
    uv run --directory training/mae-cifar100 python linear_probe.py \
        --data-path data/cifar100.npz \
        --checkpoint-path runs/mae_cifar100/mae_best.pt \
        --output-dir runs/mae_cifar100
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from png_utils import tensor_to_uint8, upscale, write_png
from train_mae import MaskedAutoencoderViT, iterate_batches

NUM_CLASSES = 100
NUM_COARSE = 20

# Hardcoded CIFAR-100 training-set statistics (per-channel mean/std, the
# standard published values - computed from the data, not a pretrained
# network), used to normalize probe inputs.
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

GRID_COLS = 8
GRID_ROWS = 4
UPSCALE = 4
BORDER = 2
GREEN = (0, 180, 0)
RED = (220, 0, 0)


def normalize_batch(x):
    mean = torch.as_tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def extract_features(model, X, batch_size, device):
    """Precompute frozen-encoder features for a whole split (normalization
    only, no masking, mean-pooled patch tokens). Returns a float32 numpy
    array (n, dim)."""
    feats = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            x = torch.from_numpy(X[start:start + batch_size]).to(device)
            feats.append(model.forward_features(normalize_batch(x)).cpu())
    return torch.cat(feats).numpy()


def make_lr_lambda(total_epochs, eta_min_ratio=0.0):
    """Hand-written cosine LR multiplier 1 -> eta_min_ratio (the probe's
    schedule: plain cosine from the base LR; a linear head on fixed
    features needs no warmup)."""

    def lr_lambda(epoch):  # 0-based epoch from the scheduler
        progress = epoch / max(1, total_epochs)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

    return lr_lambda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar100.npz")
    parser.add_argument("--checkpoint-path",
                        default="runs/mae_cifar100/mae_best.pt")
    parser.add_argument("--output-dir", default="runs/mae_cifar100")
    parser.add_argument("--num-epochs", type=int, default=60,
                        help="Probe epochs on the cached features (cheap: "
                             "one linear layer, no network passes).")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="The paper's probe LR (SGD, not AdamW).")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-examples", type=int, default=10,
                        help="How many example images to list in the "
                             "console (half correct, half misclassified).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    y_coarse_train = data["y_coarse_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    y_coarse_test = data["y_coarse_test"]
    fine_names = [str(n) for n in data["fine_label_names"]]
    coarse_names = [str(n) for n in data["coarse_label_names"]]

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    if ckpt.get("arch") != "mae_cifar100":
        raise SystemExit(f"{args.checkpoint_path} is not a mae_cifar100 "
                         f"checkpoint (arch={ckpt.get('arch')!r})")
    model = MaskedAutoencoderViT(
        patch_size=ckpt["patch_size"],
        dim=ckpt["dim"],
        depth=ckpt["depth"],
        heads=ckpt["heads"],
        mlp_ratio=ckpt["mlp_ratio"],
        dropout=ckpt["dropout"],
        decoder_dim=ckpt["decoder_dim"],
        decoder_depth=ckpt["decoder_depth"],
        decoder_heads=ckpt["decoder_heads"],
        mask_ratio=ckpt["mask_ratio"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded {args.checkpoint_path} (pretrain epoch {ckpt['epoch']}, "
          f"val recon-mse {ckpt['val_loss']:.5f}), encoder frozen")

    print("Precomputing frozen features (normalize-only inputs)...")
    start_time = time.time()
    F_train = extract_features(model, X_train, args.batch_size, device)
    F_test = extract_features(model, X_test, args.batch_size, device)
    print(f"Features: train {F_train.shape} test {F_test.shape} "
          f"({time.time() - start_time:.0f}s)")

    head = nn.Linear(F_train.shape[1], NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        head.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(args.num_epochs),
    )

    Xf = torch.from_numpy(F_train).to(device)
    yf = torch.from_numpy(y_train).to(device)
    best_train_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0.0
        total_correct = 0
        n_seen = 0
        n_batches = 0
        head.train()
        for idx in iterate_batches(np.arange(F_train.shape[0]),
                                   args.batch_size, rng, shuffle=True):
            x, t = Xf[idx], yf[idx]
            logits = head(x)
            loss = F.cross_entropy(logits, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == t).sum().item()
            n_seen += t.numel()
            n_batches += 1
        scheduler.step()
        train_acc = total_correct / n_seen
        best_train_acc = max(best_train_acc, train_acc)
        print(f"probe epoch {epoch:3d}/{args.num_epochs}  "
              f"train: loss={total_loss / n_batches:.4f} "
              f"acc={train_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # full test-split accounting (no shuffle, so predictions line up with
    # the stored row order for the example picker / grid)
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(F_test).to(device))
    preds = logits.argmax(1).cpu().numpy()
    top5 = logits.topk(5, dim=1).indices.cpu().numpy()

    acc = float((preds == y_test).mean())
    top5_acc = float((top5 == y_test[:, None]).any(1).mean())

    # coarse top-1: predicted fine class -> its 20-class superclass
    fine_to_coarse = np.zeros(NUM_CLASSES, dtype=np.int64)
    for t in range(NUM_CLASSES):
        coarse_of_t = y_coarse_train[y_train == t]
        fine_to_coarse[t] = int(np.bincount(coarse_of_t).argmax())
    coarse_pred = fine_to_coarse[preds]
    coarse_acc = float((coarse_pred == y_coarse_test).mean())

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t in range(NUM_CLASSES):
        mask = y_test == t
        for p in range(NUM_CLASSES):
            conf[t, p] = int((preds[mask] == p).sum())
    per_class = conf.diagonal() / conf.sum(axis=1).astype(np.float64)

    print()
    print(f"Test top-1 accuracy: {acc:.4f}  "
          f"({int(round(acc * y_test.size))}/{y_test.size} correct)")
    print(f"Test top-5 accuracy: {top5_acc:.4f}")
    print(f"Test coarse (20 superclass) top-1 accuracy: {coarse_acc:.4f}")
    order = np.argsort(per_class)
    print("Best per-class (of 100):")
    for t in order[-5:][::-1]:
        print(f"  {fine_names[t]:>16}: {per_class[t]:.4f}")
    print("Worst per-class (of 100):")
    for t in order[:5]:
        print(f"  {fine_names[t]:>16}: {per_class[t]:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint_path}\n")
        f.write(f"pretrain_epoch: {ckpt['epoch']}\n")
        f.write(f"pretrain_val_recon_mse: {ckpt['val_loss']:.5f}\n")
        f.write(f"probe_epochs: {args.num_epochs}\n")
        f.write(f"probe_lr: {args.learning_rate}\n")
        f.write(f"probe_train_acc_best: {best_train_acc:.4f}\n")
        f.write(f"test_top1_accuracy: {acc:.4f}\n")
        f.write(f"correct: {int(round(acc * y_test.size))}/{y_test.size}\n")
        f.write(f"test_top5_accuracy: {top5_acc:.4f}\n")
        f.write(f"test_coarse_top1_accuracy: {coarse_acc:.4f}\n")
        f.write("per_class_accuracy:\n")
        for t in range(NUM_CLASSES):
            f.write(f"  {fine_names[t]}: {per_class[t]:.4f}\n")
        f.write("confusion_matrix (rows=true, cols=pred, 100x100):\n")
        for t in range(NUM_CLASSES):
            f.write("  " + " ".join(str(int(conf[t, c]))
                                    for c in range(NUM_CLASSES)) + "\n")
    print(f"\nWrote {metrics_path}")

    head_path = os.path.join(args.output_dir, "probe_head.pt")
    torch.save(
        {"head_state": head.state_dict(),
         "epochs": args.num_epochs,
         "train_acc": best_train_acc,
         "test_top1": acc,
         "test_top5": top5_acc,
         "feature_dim": F_train.shape[1],
         "num_classes": NUM_CLASSES,
         "checkpoint": args.checkpoint_path},
        head_path,
    )
    print(f"Wrote {head_path}")

    # predictions grid: first 16 correct, then first 16 wrong
    correct_idx = np.where(preds == y_test)[0]
    wrong_idx = np.where(preds != y_test)[0]
    shown = list(correct_idx[:GRID_COLS * GRID_ROWS // 2]) + \
            list(wrong_idx[:GRID_COLS * GRID_ROWS // 2])
    cell_img = UPSCALE * 32
    cell = cell_img + 2 * BORDER
    canvas = np.zeros((GRID_ROWS * cell, GRID_COLS * cell, 3),
                      dtype=np.uint8)
    for i, idx in enumerate(shown):
        r, c = divmod(i, GRID_COLS)
        img = tensor_to_uint8(torch.from_numpy(X_test[idx]))
        img = upscale(img, UPSCALE)
        color = GREEN if preds[idx] == y_test[idx] else RED
        bordered = np.full((cell, cell, 3), color, dtype=np.uint8)
        bordered[BORDER:BORDER + cell_img, BORDER:BORDER + cell_img] = img
        y0, x0 = r * cell, c * cell
        canvas[y0:y0 + cell, x0:x0 + cell] = bordered
    grid_path = os.path.join(args.output_dir, "probe_grid.png")
    write_png(grid_path, canvas)
    print(f"Wrote {grid_path}")

    # console examples: first half correct, first half misclassified
    def print_examples(idx_list, heading, n):
        print(f"\n=== {heading} ===")
        for i in idx_list[:n]:
            probs = torch.softmax(logits[i], dim=0)
            conf_pct = float(probs.max()) * 100.0
            mark = "correct" if preds[i] == y_test[i] else "WRONG"
            print(f"[{mark}] test#{i} "
                  f"true={fine_names[int(y_test[i])]:>16} "
                  f"pred={fine_names[int(preds[i])]:>16} "
                  f"(conf {conf_pct:.1f}%)")

    n_ex = args.num_examples // 2
    print_examples(correct_idx, f"first {n_ex} correct predictions", n_ex)
    print_examples(wrong_idx, f"first {n_ex} misclassifications", n_ex)


if __name__ == "__main__":
    main()

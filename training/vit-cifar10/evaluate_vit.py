"""
Evaluate a trained ViT checkpoint against the held-out 10k-image CIFAR-10
test split, and show what the model actually gets right and wrong:

  - overall top-1 accuracy on the full test split (1,000 images per class)
  - top-5 accuracy (whether the true class is in the top 5 logits - the
    standard ViT reporting metric, meaningful even at 10 classes)
  - per-class accuracy + confusion matrix
  - test_metrics.txt written to --output-dir with the same numbers
  - predictions_grid.png: a hand-written zlib RGB PNG (no Pillow) showing
    64 test images - the first 32 correct predictions, then the first 32
    misclassifications - with a green border for correct and a red border
    for wrong, so a glance shows whether errors are concentrated in a few
    classes (the confusion matrix says which)
  - a console listing of the first few correct/wrong examples with class
    names and confidence

The test split gets normalization only (the same hardcoded train
statistics the trainer uses), no augmentation, no shuffle.

Usage:
    uv run --directory training/vit-cifar10 python evaluate_vit.py \
        --data-path data/cifar10.npz \
        --checkpoint-path runs/vit_cifar10/vit_best.pt \
        --output-dir runs/vit_cifar10
"""

import argparse
import os
import struct
import zlib

import numpy as np
import torch

from train_vit import VisionTransformer, normalize_batch

GRID_COLS = 8
GRID_ROWS = 8
UPSCALE = 4
BORDER = 2
GREEN = (0, 180, 0)
RED = (220, 0, 0)


def write_png(path, rgb):
    """Hand-written 8-bit RGB PNG (stdlib zlib/struct): one filter byte 0
    per scanline, IHDR/IDAT/IEND chunks. No imaging library."""
    H, W, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(H))

    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def tensor_to_uint8(x):
    """(3, 32, 32) float [0,1] -> (32, 32, 3) uint8."""
    img = np.clip(x.cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255)
    return img.astype(np.uint8)


def upscale(img, scale):
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar10.npz")
    parser.add_argument("--checkpoint-path",
                        default="runs/vit_cifar10/vit_best.pt")
    parser.add_argument("--output-dir", default="runs/vit_cifar10")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-examples", type=int, default=10,
                        help="How many example images to list in the "
                             "console (half correct, half misclassified).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path)
    X_test, y_test = data["X_test"], data["y_test"]
    label_names = [str(n) for n in data["label_names"]]

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    if ckpt.get("arch") != "vit_cifar10":
        raise SystemExit(f"{args.checkpoint_path} is not a vit_cifar10 "
                         f"checkpoint (arch={ckpt.get('arch')!r})")
    model = VisionTransformer(
        patch_size=ckpt["patch_size"],
        dim=ckpt["dim"],
        depth=ckpt["depth"],
        heads=ckpt["heads"],
        mlp_ratio=ckpt["mlp_ratio"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.checkpoint_path} (epoch {ckpt['epoch']}, "
          f"val_acc {ckpt['val_acc']:.4f})")

    # full test-split accounting (no shuffle, so predictions line up with
    # the stored row order for the example picker / grid)
    logits = []
    with torch.no_grad():
        for start in range(0, X_test.shape[0], args.batch_size):
            x = torch.from_numpy(X_test[start:start + args.batch_size]).to(device)
            logits.append(model(normalize_batch(x)))
    logits = torch.cat(logits)
    preds = logits.argmax(1).cpu().numpy()
    top5 = logits.topk(5, dim=1).indices.cpu().numpy()

    acc = float((preds == y_test).mean())
    top5_acc = float((top5 == y_test[:, None]).any(1).mean())
    conf = np.zeros((10, 10), dtype=np.int64)  # rows true, cols pred
    for t in range(10):
        mask = y_test == t
        for p in range(10):
            conf[t, p] = int((preds[mask] == p).sum())

    print()
    print(f"Test top-1 accuracy: {acc:.4f}  "
          f"({int(round(acc * y_test.size))}/{y_test.size} correct)")
    print(f"Test top-5 accuracy: {top5_acc:.4f}")
    print("Per-class accuracy (1,000 images each):")
    for t in range(10):
        print(f"  {label_names[t]:>9}: {conf[t, t] / conf[t].sum():.4f}")
    print("Confusion matrix (rows=true, cols=pred, rightmost column = row total):")
    header = "        " + "".join(f"{label_names[c][:4]:>6}" for c in range(10)) + "   total"
    print(header)
    for t in range(10):
        row = "  ".join(f"{conf[t, c]:5d}" for c in range(10))
        print(f"{label_names[t][:4]:>6}  {row}  {conf[t].sum():5d}")

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint: {args.checkpoint_path}\n")
        f.write(f"test_top1_accuracy: {acc:.4f}\n")
        f.write(f"correct: {int(round(acc * y_test.size))}/{y_test.size}\n")
        f.write(f"test_top5_accuracy: {top5_acc:.4f}\n")
        for t in range(10):
            f.write(f"{label_names[t]}_accuracy: "
                    f"{conf[t, t] / conf[t].sum():.4f}\n")
        f.write("confusion_matrix (rows=true, cols=pred):\n")
        for t in range(10):
            f.write("  " + " ".join(str(int(conf[t, c])) for c in range(10))
                    + "\n")
    print(f"\nWrote {metrics_path}")

    # predictions grid: first 32 correct, then first 32 wrong
    correct_idx = np.where(preds == y_test)[0]
    wrong_idx = np.where(preds != y_test)[0]
    shown = list(correct_idx[:GRID_COLS * GRID_ROWS // 2]) + \
            list(wrong_idx[:GRID_COLS * GRID_ROWS // 2])
    cell_img = UPSCALE * 32
    cell = cell_img + 2 * BORDER
    canvas = np.zeros((GRID_ROWS * cell, GRID_COLS * cell, 3), dtype=np.uint8)
    for i, idx in enumerate(shown):
        r, c = divmod(i, GRID_COLS)
        img = tensor_to_uint8(torch.from_numpy(X_test[idx]))
        img = upscale(img, UPSCALE)
        color = GREEN if preds[idx] == y_test[idx] else RED
        bordered = np.full((cell, cell, 3), color, dtype=np.uint8)
        bordered[BORDER:BORDER + cell_img, BORDER:BORDER + cell_img] = img
        y0, x0 = r * cell, c * cell
        canvas[y0:y0 + cell, x0:x0 + cell] = bordered
    grid_path = os.path.join(args.output_dir, "predictions_grid.png")
    write_png(grid_path, canvas)
    print(f"Wrote {grid_path}")

    # console examples: first half correct, first half misclassified
    def print_examples(idx_list, heading, n):
        print(f"\n=== {heading} ===")
        for i in idx_list[:n]:
            probs = torch.softmax(logits[i], dim=0)
            conf_pct = float(probs.max()) * 100.0
            mark = "correct" if preds[i] == y_test[i] else "WRONG"
            print(f"[{mark}] test#{i} true={label_names[int(y_test[i])]:>9} "
                  f"pred={label_names[int(preds[i])]:>9} "
                  f"(conf {conf_pct:.1f}%)")

    n_ex = args.num_examples // 2
    print_examples(correct_idx, f"first {n_ex} correct predictions", n_ex)
    print_examples(wrong_idx, f"first {n_ex} misclassifications", n_ex)


if __name__ == "__main__":
    main()

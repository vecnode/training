"""
Turn the raw Large Movie Review Dataset (aclImdb_v1, Maas et al. 2011) text
files into numpy arrays ready for Text-CNN training: integer token-id
matrices padded to a fixed length, one per split, plus the vocab list.

No dataset library (no torchtext, no datasets, no nltk, no gensim): the
raw review .txt files under aclImdb/train/{pos,neg} and aclImdb/test/
{pos,neg} are read directly, tokenized by hand with a plain regex
(lowercased [a-z0-9']+ tokens, <br /> tags stripped), the vocabulary is
built from the **train split only** (test tokens not in it become <unk>),
and the result is saved as a single .npz. The labeled-file count per split
is verified (exactly 12500) so a partial/in-progress extraction fails
loudly instead of silently training on a subset. The unlabeled
train/unsup reviews are deliberately ignored - this pipeline is a pure
supervised classifier.

No pretrained word vectors (no GloVe): embeddings are trained from scratch
by train_cnn.py with random init. This pipeline is strictly IMDB-only by
design.

Usage:
    uv run --directory training/imdb-sentiment-cnn python build_imdb_dataset.py \
        --data-dir "E:\\datasets\\aclImdb_v1" \
        --output-dir data
"""

import argparse
import os
import re
from collections import Counter

import numpy as np

EXPECTED_SPLIT_COUNTS = {"train_pos": 12500, "train_neg": 12500,
                         "test_pos": 12500, "test_neg": 12500}
PAD_TOKEN = "<pad>"   # id 0 - used for sequence padding
UNK_TOKEN = "<unk>"   # id 1 - any train-only vocab miss
BR_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9']+")


def resolve_acl_dir(data_dir):
    """The aclImdb tar extracts to an aclImdb/ subfolder. Accept either
    --data-dir pointing straight at that folder (train/ and test/ directly
    inside) or at a parent that contains it - the same two-shapes handling
    as build_cifar10_dataset.py's resolve_batch_dir."""
    if os.path.isdir(os.path.join(data_dir, "train")) and \
       os.path.isdir(os.path.join(data_dir, "test")):
        return data_dir
    nested = os.path.join(data_dir, "aclImdb")
    if os.path.isdir(os.path.join(nested, "train")) and \
       os.path.isdir(os.path.join(nested, "test")):
        return nested
    raise FileNotFoundError(
        f"Could not find the aclImdb train/ and test/ folders under "
        f"{data_dir} (looked directly and under an aclImdb subfolder)"
    )


def load_split(acl_dir, split, label, max_samples=0):
    """Read every review .txt in aclImdb/<split>/<pos|neg> and tokenize it.
    Returns (list_of_token_lists, np.int64 labels)."""
    folder = os.path.join(acl_dir, split, label)
    files = sorted(os.listdir(folder))
    docs, labels = [], []
    for fname in files:
        path = os.path.join(folder, fname)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        # strip <br /> tags (present in essentially every review), then
        # lowercase and pull out word tokens; drop bare-apostrophe junk
        tokens = [t for t in TOKEN_RE.findall(BR_TAG.sub(" ", text).lower())
                  if t.strip("'")]
        docs.append(tokens)
        labels.append(label == "pos")
        if max_samples and len(docs) >= max_samples:
            break
    return docs, np.asarray(labels, dtype=np.int64)


def build_vocab(train_pos, train_neg, min_count, max_vocab):
    """Vocab from train-split tokens only: frequency-ordered, min-count
    filtered, capped. ids: 0=<pad>, 1=<unk>, 2.. = words."""
    counts = Counter()
    for doc in train_pos + train_neg:
        counts.update(doc)
    words = sorted(
        (w for w, c in counts.items() if c >= min_count),
        key=lambda w: (-counts[w], w),   # freq desc, then alphabetical
    )[:max_vocab]
    vocab = [PAD_TOKEN, UNK_TOKEN] + words
    return vocab, counts


def encode(docs, word_to_id, max_len):
    """Token lists -> (n, max_len) int32 matrix, truncated to the first
    max_len tokens, padded with 0."""
    n = len(docs)
    X = np.zeros((n, max_len), dtype=np.int32)
    for i, doc in enumerate(docs):
        ids = [word_to_id.get(t, 1) for t in doc[:max_len]]
        X[i, :len(ids)] = ids
    return X


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=r"E:\datasets\aclImdb_v1",
        help="Directory containing the extracted aclImdb reviews (an "
             "aclImdb subfolder is also accepted).",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--max-len", type=int, default=512,
        help="Truncate/pad every review to this many tokens "
             "(mean review length is ~230 words).",
    )
    parser.add_argument(
        "--min-count", type=int, default=2,
        help="Drop train tokens seen fewer times than this.",
    )
    parser.add_argument(
        "--max-vocab", type=int, default=50000,
        help="Cap the vocab (excluding <pad>/<unk>) at this many words.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="If >0, cap each split to this many reviews (quick smoke run "
             "before committing to the full 50k-review dataset).",
    )
    args = parser.parse_args()

    acl_dir = resolve_acl_dir(args.data_dir)
    print(f"Using aclImdb at {acl_dir}")

    # verify counts up front - a partial extraction must fail here, loudly
    for name, expected in EXPECTED_SPLIT_COUNTS.items():
        split, label = name.split("_")
        folder = os.path.join(acl_dir, split, label)
        n_files = len(os.listdir(folder))
        status = "ok" if n_files == expected else f"MISMATCH (expected {expected})"
        print(f"  {name}: {n_files} files - {status}")
        if n_files != expected:
            raise SystemExit(
                f"Split {name} has {n_files} files, expected {expected}. "
                f"Refusing to build on a partial dataset - re-extract "
                f"aclImdb_v1.tar.gz and rerun."
            )

    print("Reading and tokenizing reviews...")
    train_pos, y_train_pos = load_split(acl_dir, "train", "pos", args.max_samples)
    train_neg, y_train_neg = load_split(acl_dir, "train", "neg", args.max_samples)
    test_pos, y_test_pos = load_split(acl_dir, "test", "pos", args.max_samples)
    test_neg, y_test_neg = load_split(acl_dir, "test", "neg", args.max_samples)

    vocab, counts = build_vocab(train_pos, train_neg,
                                args.min_count, args.max_vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    print(f"Vocab: {len(vocab)} entries ({args.min_count}+ counts, "
          f"cap {args.max_vocab}); train tokens total: {sum(counts.values()):,}")

    X_train = encode(train_pos + train_neg, word_to_id, args.max_len)
    y_train = np.concatenate([y_train_pos, y_train_neg])
    X_test = encode(test_pos + test_neg, word_to_id, args.max_len)
    y_test = np.concatenate([y_test_pos, y_test_neg])

    # coverage: fraction of train tokens that survived the min-count filter
    covered = sum(c for w, c in counts.items() if w in word_to_id and w not in (PAD_TOKEN, UNK_TOKEN))
    print(f"Train tokens covered by vocab: {covered / sum(counts.values()):.1%}")

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, "imdb.npz")
    np.savez(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        max_len=args.max_len,
        vocab_size=len(vocab),
    )
    print(f"Saved {npz_path}  "
          f"train {X_train.shape}  test {X_test.shape}")

    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print(f"Saved {vocab_path} (id = line number: 0={PAD_TOKEN}, "
          f"1={UNK_TOKEN}, then words)")


if __name__ == "__main__":
    main()

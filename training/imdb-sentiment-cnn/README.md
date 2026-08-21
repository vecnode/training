# imdb-sentiment-cnn

Binary sentiment classification (positive/negative) on the
[Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
(Maas et al., 2011) with a **Text CNN trained from scratch** — a
[Kim 2014](https://arxiv.org/abs/1408.5882) "CNN-rand" style model: a
randomly-initialized trainable embedding layer, multi-width 1D convolutions
(3/4/5-grams), 1-max-pooling per filter, dropout, and a 2-way linear head.

No NLP library (no torchtext/transformers/nltk), no pretrained embeddings
(no GloVe): the tokenizer is a plain regex in `build_imdb_dataset.py`, the
model is plain `torch.nn` modules, torch is used only for tensor ops,
autograd, and GPU execution — the same hand-written philosophy as
`training/cifar10-vqvae` and the rest of `training/`. Everything is learned
from the 25k labeled reviews.

## Files

| File | Role |
|---|---|
| `build_imdb_dataset.py` | Parse the extracted `aclImdb/` tree by hand, tokenize (lowercase `[a-z0-9']+`, `<br />` stripped), build the vocab from **train only**, verify 12,500 files per split (a partial extraction fails loudly), save `data/imdb.npz` + `data/vocab.txt` |
| `train_cnn.py` | Hand-written Text CNN (embedding → 3 conv widths × 128 filters → ReLU → 1-max-pool → concat → dropout → linear-2), cross-entropy, Adam + cosine schedule, numpy-permutation batching (no DataLoader), 10% val holdout, best checkpoint by val acc |
| `evaluate_cnn.py` | Held-out test-split accuracy, per-class breakdown + confusion matrix, `test_metrics.txt`, and example reviews (first correct / first wrong) with confidence |

## Usage

```sh
# build (data dir accepts the aclImdb/ folder or a parent containing it)
uv run --directory training/imdb-sentiment-cnn python build_imdb_dataset.py \
    --data-dir "C:\path\to\aclImdb_v1" --output-dir data

# train (~5.2M params, random embeddings)
uv run --directory training/imdb-sentiment-cnn python train_cnn.py \
    --data-path data/imdb.npz --num-epochs 20 --batch-size 128 \
    --output-dir runs/imdb_cnn

# evaluate on the official 25k test split
uv run --directory training/imdb-sentiment-cnn python evaluate_cnn.py \
    --data-path data/imdb.npz --checkpoint-path runs/imdb_cnn/cnn_best.pt \
    --vocab-path data/vocab.txt --output-dir runs/imdb_cnn
```

## Why this accuracy target is what it is

| Setup | Expected test acc | Notes |
|---|---|---|
| Text CNN, **random** embeddings (this pipeline) | ~83–86% | Kim's published "CNN-rand" is 82.7%; tuned runs land a few points higher |
| Text CNN + GloVe vectors | ~87–89% | Pretrained embeddings are worth ~+4 pts — deliberately not used here |
| From-scratch ceiling (AWD-LSTM on IMDB-only) | ~91% | LM-pretrains on the 50k unlabeled reviews first — a much bigger project |
| Pretrained-transformer fine-tune | ~95% | That's `fine-tuning/`'s job, not `training/` |

## Verified runs

Real run on a single RTX 3090 (default config: emb 128, 128 filters,
widths 3/4/5, dropout 0.5, Adam 1e-3 + cosine over 20 epochs, batch 128,
seed 0):

| Metric | Value |
|---|---|
| Build | 50,000 reviews → `data/imdb.npz`, vocab 50,002 (99.3% of train tokens covered), a couple of minutes |
| Model | 6,598,018 params (6,400,256 in the embedding) |
| Training | 20 epochs in **31 s** |
| Best val acc | 90.2% (epoch 3 — train acc hits 100% soon after; overfitting is fast on 22.5k docs) |
| **Test acc (25k held-out)** | **89.2%** — 22,299/25,000, neg 88.96% / pos 89.43% |

A dropout-0.7 variant (10 epochs, same schedule) scored 87.9% on test —
worse, so dropout 0.5 with val-based early stopping stays the default.

89.2% from random embeddings is far above Kim's published CNN-rand
(82.7%): full-length reviews (512 tokens instead of short truncation) plus
picking the best epoch on validation is what makes the difference. The
misclassified examples are genuinely ambiguous — sarcastic/backhanded
positive reviews ("primed to expect hollywood fantasy revisionism...") —
classified with mid confidence.

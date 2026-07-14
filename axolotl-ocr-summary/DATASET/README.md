# DATASET/

Raw input data lives here. **Nothing in this folder except this README is committed to git** —
see `.gitignore`. Treat this as a local drop zone, not a versioned artifact.

## What goes here

Whatever raw OCR/summary pairs you're fine-tuning on, e.g. output pulled from the
[pre-training](https://github.com/vecnode/pre-training) pipeline:

```
DATASET/
├── DATASET_SUMMARIES.csv   # or your own CSV/JSONL
└── ...
```

Any tabular format works as long as it has one column with the source text (OCR output)
and one column with the target text (summary). Column names are not hardcoded — you pass
them as flags to the prep script.

## Turning this into training data

Run from the repo root:

```bash
uv run python scripts/prepare_dataset.py \
  --input DATASET/DATASET_SUMMARIES.csv \
  --text-col text \
  --summary-col summary \
  --val-split 0.1
```

This writes `train.jsonl` / `val.jsonl` into [`DATASET_JSONL/`](../DATASET_JSONL/README.md),
which is what the Axolotl configs in [`configs/`](../configs/) point at.

## Why this folder isn't versioned

Datasets are large, change often, and may contain content you don't want in git history.
Everything downloadable or regeneratable — raw data, HF model weights, uv/pip caches — is
kept local under `DATASET/` or `.cache/` and gitignored. Only the scripts that produce/consume
this data are versioned.

# DATASET_JSONL/

Generated training/validation splits, produced from [`DATASET/`](../DATASET/README.md) by
`scripts/prepare_dataset.py`. Like `DATASET/`, only this README is committed — the actual
`.jsonl` files are gitignored and regenerated on demand.

## Expected contents (after running prepare_dataset.py)

```
DATASET_JSONL/
├── train.jsonl
└── val.jsonl
```

Each line is an Alpaca-style record consumed directly by the `configs/*.yml` Axolotl configs
(`type: alpaca`):

```json
{"instruction": "Summarize the following OCR text.", "input": "<ocr text>", "output": "<summary>"}
```

Regenerate at any time with:

```bash
uv run python scripts/prepare_dataset.py --input DATASET/<your_file>.csv --text-col <col> --summary-col <col>
```

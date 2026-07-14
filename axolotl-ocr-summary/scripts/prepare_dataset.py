"""Convert a raw OCR-text/summary table into Axolotl-ready JSONL splits.

Reads any CSV with a source-text column and a target-summary column (column
names are flags, not hardcoded) and writes DATASET_JSONL/train.jsonl and
DATASET_JSONL/val.jsonl in the Alpaca record shape Axolotl's `type: alpaca`
dataset loader expects:

    {"instruction": ..., "input": ..., "output": ...}

Usage:
    uv run python scripts/prepare_dataset.py \
        --input DATASET/DATASET_SUMMARIES.csv \
        --text-col text --summary-col summary --val-split 0.1
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_INSTRUCTION = "Summarize the following OCR text."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to input CSV")
    parser.add_argument("--text-col", required=True, help="Column containing source/OCR text")
    parser.add_argument("--summary-col", required=True, help="Column containing target summary")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Fixed instruction string prepended to every record",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("DATASET_JSONL"),
        type=Path,
        help="Directory to write train.jsonl / val.jsonl into",
    )
    parser.add_argument("--val-split", default=0.1, type=float, help="Fraction held out for validation")
    parser.add_argument("--seed", default=42, type=int, help="Shuffle seed for the split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    missing = {args.text_col, args.summary_col} - set(df.columns)
    if missing:
        raise SystemExit(f"Input is missing column(s): {sorted(missing)}. Found: {list(df.columns)}")

    df = df[[args.text_col, args.summary_col]].dropna()
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val_size = int(len(df) * args.val_split)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_df, args, args.out_dir / "train.jsonl")
    _write_jsonl(val_df, args, args.out_dir / "val.jsonl")

    print(f"Wrote {len(train_df)} train / {len(val_df)} val records to {args.out_dir}")


def _write_jsonl(df: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            record = {
                "instruction": args.instruction,
                "input": getattr(row, args.text_col),
                "output": getattr(row, args.summary_col),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

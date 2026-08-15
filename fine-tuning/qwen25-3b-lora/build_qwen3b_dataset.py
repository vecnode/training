from __future__ import annotations

import argparse
import json
from pathlib import Path

# Pass this to train_qwen3b_lora.py's --instruction when training on this
# source (it's also the trainer's default, so it usually doesn't need to be
# passed explicitly - see train_qwen3b_lora.py's DEFAULT_INSTRUCTION).
CNN_DAILYMAIL_PROMPT_INSTRUCTION = (
    "Summarize this news article in one concise paragraph. "
    "Focus on key entities, dates, and events.\n\n"
    "Article text:\n"
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()

    parser = argparse.ArgumentParser(description="Build Qwen2.5-3B training JSONL from a CNN/DailyMail parquet dump")
    parser.add_argument(
        "--cnn-dailymail-dir",
        type=Path,
        required=True,
        help="Directory of CNN/DailyMail parquet files (e.g. .../cnn_dailymail/3.0.0), as downloaded from "
        "https://huggingface.co/datasets/abisee/cnn_dailymail. Maps article -> text, highlights -> summary.",
    )
    parser.add_argument(
        "--cnn-dailymail-split",
        choices=["train", "validation", "test"],
        default="train",
        help="Which CNN/DailyMail parquet split to read (default: train)",
    )
    parser.add_argument("--out-jsonl", type=Path, default=here.parent / "data" / "qwen3b_train.jsonl", help="Output JSONL path (default: data/qwen3b_train.jsonl)")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick tests (0 = all)")
    parser.add_argument("--max-text-chars", type=int, default=8000, help="Cap source text stored per sample (the trainer further truncates by tokens, head+tail, to fit the summary)")
    return parser.parse_args()


def build_from_cnn_dailymail(args: argparse.Namespace, out_jsonl: Path) -> None:
    import pandas as pd

    parquet_dir = args.cnn_dailymail_dir.resolve()
    if not parquet_dir.exists():
        raise FileNotFoundError(f"CNN/DailyMail parquet directory not found: {parquet_dir}")

    files = sorted(parquet_dir.glob(f"{args.cnn_dailymail_split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No '{args.cnn_dailymail_split}-*.parquet' files found in {parquet_dir}")

    df = pd.concat([pd.read_parquet(f, columns=["article", "highlights", "id"]) for f in files], ignore_index=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_jsonl.open("w", encoding="utf-8") as dst:
        for row in df.itertuples(index=False):
            article = (row.article or "").strip()
            summary = (row.highlights or "").strip()
            if not article or not summary:
                continue

            text = article[: args.max_text_chars]
            prompt = f"{CNN_DAILYMAIL_PROMPT_INSTRUCTION}{text}"

            sample = {
                "source": "cnn_dailymail",
                "id": row.id,
                "text": text,
                "prompt": prompt,
                "summary": summary,
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_samples and kept >= args.max_samples:
                break

    print(f"Wrote {kept} sample(s) from CNN/DailyMail ({args.cnn_dailymail_split} split) to: {out_jsonl}")
    print(f"Source rows available in split: {len(df)}")


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    out_jsonl = args.out_jsonl
    if not out_jsonl.is_absolute():
        out_jsonl = (script_dir / out_jsonl).resolve()

    build_from_cnn_dailymail(args, out_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# News-article instruction, distinct from the scanned-OCR-page one below —
# pass this to train_llava15_lora.py's --instruction when training on this
# source, since "scanned document page" / "UAP-related content" is specific
# to the pre-training repo's declassified-document use case.
CNN_DAILYMAIL_PROMPT_INSTRUCTION = (
    "Summarize this news article in one concise paragraph. "
    "Focus on key entities, dates, and events.\n\n"
    "Article text:\n"
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_root = here.parents[1]

    parser = argparse.ArgumentParser(description="Build LLaVA 1.5 training JSONL from OCR and summaries CSV files, or from a CNN/DailyMail parquet dump")
    parser.add_argument("--root", type=Path, default=default_root, help="Project root directory")
    parser.add_argument("--ocr-csv", type=Path, default=None, help="Path to OCR CSV (default: <root>/output/Release_1_OCR.csv)")
    parser.add_argument("--summaries-csv", type=Path, default=None, help="Path to summaries CSV (default: <root>/output/Release_1_SUMMARIES.csv)")
    parser.add_argument("--out-jsonl", type=Path, default=here.parent / "data" / "llava15_train.jsonl", help="Output JSONL path (default: training/data/llava15_train.jsonl)")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick tests (0 = all)")
    parser.add_argument("--max-ocr-chars", type=int, default=8000, help="Cap OCR text stored per sample (the trainer further truncates by tokens, head+tail, to fit the summary)")
    parser.add_argument(
        "--cnn-dailymail-dir",
        type=Path,
        default=None,
        help="Directory of CNN/DailyMail parquet files (e.g. .../cnn_dailymail/3.0.0), as downloaded from "
        "https://huggingface.co/datasets/abisee/cnn_dailymail. When set, --ocr-csv/--summaries-csv are ignored "
        "and this source is used instead: article -> ocr_text, highlights -> summary.",
    )
    parser.add_argument(
        "--cnn-dailymail-split",
        choices=["train", "validation", "test"],
        default="train",
        help="Which CNN/DailyMail parquet split to read (default: train)",
    )
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

            ocr_text = article[: args.max_ocr_chars]
            prompt = f"{CNN_DAILYMAIL_PROMPT_INSTRUCTION}{ocr_text}"

            sample = {
                "source": "cnn_dailymail",
                "id": row.id,
                "ocr_text": ocr_text,
                "prompt": prompt,
                "summary": summary,
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_samples and kept >= args.max_samples:
                break

    print(f"Wrote {kept} sample(s) from CNN/DailyMail ({args.cnn_dailymail_split} split) to: {out_jsonl}")
    print(f"Source rows available in split: {len(df)}")


def normalize_image_key(value: str) -> str:
    key = (value or "").strip()
    key = key.replace("\\", "/")
    while "//" in key:
        key = key.replace("//", "/")
    return key


def resolve_image_path(root: Path, image_key: str, full_path: str = "") -> Path | None:
    # Prefer the absolute full_path column from the OCR CSV when present
    if full_path:
        p = Path(full_path)
        if p.exists():
            return p

    if not image_key:
        return None

    raw = Path(image_key)
    if raw.is_absolute() and raw.exists():
        return raw

    candidate = root / image_key
    if candidate.exists():
        return candidate

    candidate_alt = root / image_key.replace("/", "\\")
    if candidate_alt.exists():
        return candidate_alt

    return None


def load_summaries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_key = normalize_image_key(row.get("image") or "")
            summary = (row.get("summary") or "").strip()
            status = (row.get("status") or "").strip().lower()

            if not image_key:
                continue
            if status != "ok":
                continue
            if not summary:
                continue

            out[image_key] = summary
    return out


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    script_dir = Path(__file__).resolve().parent
    out_jsonl = args.out_jsonl
    if not out_jsonl.is_absolute():
        out_jsonl = (script_dir / out_jsonl).resolve()

    if args.cnn_dailymail_dir is not None:
        build_from_cnn_dailymail(args, out_jsonl)
        return 0

    ocr_csv = (args.ocr_csv or (root / "output" / "Release_1_OCR.csv")).resolve()
    summaries_csv = (args.summaries_csv or (root / "output" / "Release_1_SUMMARIES.csv")).resolve()

    if not ocr_csv.exists():
        raise FileNotFoundError(f"OCR CSV not found: {ocr_csv}")
    if not summaries_csv.exists():
        raise FileNotFoundError(f"Summaries CSV not found: {summaries_csv}")

    summaries_by_image = load_summaries(summaries_csv)

    kept = 0
    skipped_missing_summary = 0
    skipped_bad_ocr = 0
    skipped_missing_image = 0

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with ocr_csv.open("r", encoding="utf-8", newline="") as src, out_jsonl.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        for row in reader:
            image_key = normalize_image_key(row.get("image") or "")
            if not image_key:
                continue

            ocr_text = (row.get("text") or "").strip()
            ocr_status = (row.get("status") or "").strip().lower()

            if (not ocr_text) or (ocr_text == "(no text detected)") or (ocr_status in {"error", "empty", "legacy"}):
                skipped_bad_ocr += 1
                continue

            summary = summaries_by_image.get(image_key, "")
            if not summary:
                skipped_missing_summary += 1
                continue

            full_path = (row.get("full_path") or "").strip()
            img_path = resolve_image_path(root, image_key, full_path)
            if img_path is None:
                skipped_missing_image += 1
                continue

            prompt = (
                "Summarize this scanned document page in one concise paragraph. "
                "Focus on key entities, dates, events, and any UAP-related content if present.\n\n"
                f"OCR text:\n{ocr_text[: args.max_ocr_chars]}"
            )

            sample = {
                "image_key": image_key,
                "image_path": str(img_path),
                "ocr_text": ocr_text[: args.max_ocr_chars],
                "prompt": prompt,
                "summary": summary,
            }
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_samples and kept >= args.max_samples:
                break

    print(f"Wrote {kept} sample(s) to: {out_jsonl}")
    print(f"Skipped due to OCR quality: {skipped_bad_ocr}")
    print(f"Skipped missing summary: {skipped_missing_summary}")
    print(f"Skipped missing image: {skipped_missing_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

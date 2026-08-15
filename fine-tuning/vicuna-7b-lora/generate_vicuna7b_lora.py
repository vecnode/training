from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

# Generated text can contain characters outside the Windows console's default
# cp1252 codepage (curly quotes, em dashes, etc.) - without this, printing
# such a summary crashes with UnicodeEncodeError instead of just printing it.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Keep all Hugging Face artifacts inside training/
_TRAINING_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_TRAINING_DIR / "hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_TRAINING_DIR / "hf_cache" / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_TRAINING_DIR / "hf_cache" / "datasets"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must match the --instruction used for the adapter's training run
# (train_vicuna7b_lora.py's DEFAULT_INSTRUCTION). Override with --instruction
# if the adapter was trained with a different one.
DEFAULT_INSTRUCTION = (
    "Summarize this news article in one concise paragraph. "
    "Focus on key entities, dates, and events.\n\n"
    "Article text:\n"
)

# In a --text-file you can stack several pages/articles separated by a line
# containing this token; each chunk is summarized independently.
PAGE_DELIM = "===PAGEBREAK==="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summaries from source text with a trained Vicuna-7B LoRA adapter")
    parser.add_argument("--adapter-dir", type=Path, default=_TRAINING_DIR / "runs" / "vicuna7b_lora" / "final_adapter", help="Path to trained LoRA adapter directory")
    parser.add_argument("--model-id", default="", help="Base model id (auto-detected from adapter config when omitted)")

    # Single raw-text mode (paste new text and get one summary):
    parser.add_argument("--text", default="", help="Raw source text to summarize directly (single page/article)")
    parser.add_argument("--text-file", type=Path, default=None, help="Read raw source text from this file (single page/article)")

    # Batch CSV mode:
    parser.add_argument("--source-csv", type=Path, default=None, help="Source-text CSV input with 'text' (and optional 'image'/'status') columns")
    parser.add_argument("--reference-csv", type=Path, default=None, help="Optional reference summaries CSV for token-F1 (e.g., ../output/Release_1_SUMMARIES.csv)")
    parser.add_argument("--out-csv", type=Path, default=_TRAINING_DIR / "runs" / "vicuna7b_lora" / "generated.csv", help="Output CSV with generated predictions (CSV mode)")
    parser.add_argument("--out-metrics", type=Path, default=None, help="Optional output metrics JSON path (CSV mode)")
    parser.add_argument("--max-rows", type=int, default=100, help="Maximum rows to generate (0 = all rows)")

    # Reconstruction-test mode (sanity check: how well did training work?):
    parser.add_argument("--jsonl-eval", type=Path, default=None, help="Print N generated summaries vs. reference from a training-format JSONL (held-out split, not manual input)")
    parser.add_argument("--num-samples", type=int, default=5, help="How many held-out examples to print in --jsonl-eval mode")
    parser.add_argument("--eval-seed", type=int, default=42, help="Must match training's --seed so the held-out split lines up (train_vicuna7b_lora.py default: 42)")
    parser.add_argument("--eval-val-ratio", type=float, default=0.1, help="Must match training's --val-ratio so the held-out split lines up (train_vicuna7b_lora.py default: 0.1)")

    parser.add_argument("--max-length", type=int, default=2048, help="Max input token budget (must match training)")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Max generated tokens per sample")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Instruction prefix (must match the adapter's training --instruction)")
    return parser.parse_args()


def truncate_text_ids(ids: list[int], budget: int) -> list[int]:
    """Fit source token ids into `budget`, keeping the head and tail of the page."""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head = int(budget * 0.75)
    tail = budget - head
    if tail <= 0:
        return ids[:budget]
    return ids[:head] + ids[-tail:]


def normalize_image_key(value: str) -> str:
    key = (value or "").strip().replace("\\", "/")
    while "//" in key:
        key = key.replace("//", "/")
    return key


def read_base_model_id(adapter_dir: Path, fallback: str) -> str:
    if fallback:
        return fallback

    cfg = adapter_dir / "adapter_config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            value = (data.get("base_model_name_or_path") or "").strip()
            if value:
                return value
        except json.JSONDecodeError:
            pass

    return "lmsys/vicuna-7b-v1.5"


def load_reference_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalize_image_key(row.get("image") or row.get("image_key") or "")
            summary = (row.get("summary") or "").strip()
            status = (row.get("status") or "ok").strip().lower()
            if not key or not summary or status != "ok":
                continue
            out[key] = summary
    return out


def token_f1(pred: str, ref: str) -> float:
    p = pred.lower().split()
    r = ref.lower().split()
    if not p or not r:
        return 0.0

    counts: dict[str, int] = {}
    for tok in r:
        counts[tok] = counts.get(tok, 0) + 1

    overlap = 0
    for tok in p:
        cur = counts.get(tok, 0)
        if cur > 0:
            overlap += 1
            counts[tok] = cur - 1

    precision = overlap / len(p)
    recall = overlap / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class Summarizer:
    """Loads the base model + LoRA adapter and turns source text into a summary."""

    def __init__(self, adapter_dir: Path, base_model_id: str, max_length: int, max_new_tokens: int, instruction: str = DEFAULT_INSTRUCTION):
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.to(self.device)
        self.model.eval()

        # Precompute the fixed wrapper so only the source body is re-tokenized per call.
        self.head_ids = self.tokenizer(f"USER: {instruction}", add_special_tokens=True).input_ids
        self.suffix_ids = self.tokenizer(" ASSISTANT: ", add_special_tokens=False).input_ids

    def build_input_ids(self, text: str) -> list[int]:
        budget = self.max_length - len(self.head_ids) - len(self.suffix_ids) - self.max_new_tokens
        text_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        text_ids = truncate_text_ids(text_ids, budget)
        return self.head_ids + text_ids + self.suffix_ids

    def summarize(self, text: str) -> str:
        ids = self.build_input_ids(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Only the newly generated tokens (everything after the prompt).
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _strip_comments(chunk: str) -> str:
    # Lines starting with '#' are human-readable labels, not source content.
    return "\n".join(ln for ln in chunk.splitlines() if not ln.lstrip().startswith("#")).strip()


def run_pages(summarizer: Summarizer, raw_text: str) -> int:
    pages = [_strip_comments(p) for p in raw_text.split(PAGE_DELIM)]
    pages = [p for p in pages if p]
    if not pages:
        raise SystemExit("No source text found in the input.")

    for idx, page in enumerate(pages, start=1):
        summary = summarizer.summarize(page)
        print(f"\n=== Generated summary {idx}/{len(pages)}  ({len(page)} chars) ===\n")
        print(summary)
    print()
    return 0


def run_csv(summarizer: Summarizer, args: argparse.Namespace) -> int:
    source_csv = args.source_csv.resolve()
    if not source_csv.exists():
        raise FileNotFoundError(f"Source CSV not found: {source_csv}")

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_metrics = args.out_metrics.resolve() if args.out_metrics else out_csv.with_name(out_csv.stem + "_metrics.json")

    reference_map = load_reference_map(args.reference_csv.resolve() if args.reference_csv else None)

    print(f"Source CSV: {source_csv}")
    print(f"Output CSV: {out_csv}")

    rows_written = 0
    skipped_bad_source = 0
    with_refs = 0
    f1_sum = 0.0

    with source_csv.open("r", encoding="utf-8", newline="") as src, out_csv.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["row_id", "image_key", "prediction", "reference", "token_f1", "text_chars"])
        writer.writeheader()

        for row in reader:
            if args.max_rows > 0 and rows_written >= args.max_rows:
                break

            image_key = normalize_image_key(row.get("image") or "")
            text = (row.get("text") or "").strip()
            row_status = (row.get("status") or "").strip().lower()

            if (not text) or text == "(no text detected)" or (row_status in {"error", "empty", "legacy"}):
                skipped_bad_source += 1
                continue

            pred = summarizer.summarize(text)

            ref = reference_map.get(image_key, "")
            f1 = token_f1(pred, ref) if ref else 0.0
            if ref:
                with_refs += 1
                f1_sum += f1

            rows_written += 1
            writer.writerow(
                {
                    "row_id": rows_written,
                    "image_key": image_key,
                    "prediction": pred,
                    "reference": ref,
                    "token_f1": f"{f1:.4f}" if ref else "",
                    "text_chars": len(text),
                }
            )

            if rows_written % 10 == 0:
                print(f"Generated {rows_written} rows...")

    metrics = {
        "adapter_dir": str(args.adapter_dir.resolve()),
        "source_csv": str(source_csv),
        "output_csv": str(out_csv),
        "rows_written": rows_written,
        "skipped_bad_source": skipped_bad_source,
        "rows_with_reference": with_refs,
        "avg_token_f1": (f1_sum / with_refs) if with_refs else None,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nGeneration complete.")
    print(f"Predictions CSV: {out_csv}")
    print(f"Metrics JSON: {out_metrics}")
    if with_refs:
        print(f"Average token F1 vs reference summaries: {metrics['avg_token_f1']:.4f}")
    return 0


def run_jsonl_eval(summarizer: Summarizer, args: argparse.Namespace) -> int:
    """Reconstruction test: generate on held-out examples and print input/reference/generated.

    Replicates train_vicuna7b_lora.py's split (same seed + val_ratio applied to
    the same JSONL) so these are the actual validation examples the trainer
    reported eval_loss on, not just arbitrary rows - a genuine held-out check,
    not a training-set echo test.
    """
    path = args.jsonl_eval.resolve()
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict) and item.get("text") and item.get("summary"):
                records.append(item)

    if not records:
        raise SystemExit(f"No valid text/summary records found in {path}")

    rng = random.Random(args.eval_seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.eval_val_ratio))
    val_records = records[:n_val]

    n = min(args.num_samples, len(val_records))
    sample = val_records[:n]

    print(f"\nHeld-out split: {len(val_records)} record(s) (seed={args.eval_seed}, val_ratio={args.eval_val_ratio})")
    print(f"Printing {n} sample(s)\n")

    f1_sum = 0.0
    for i, item in enumerate(sample, start=1):
        text = item["text"]
        reference = item["summary"].strip()
        pred = summarizer.summarize(text)
        f1 = token_f1(pred, reference)
        f1_sum += f1

        print("=" * 100)
        print(f"[{i}/{n}]  id={item.get('id', 'n/a')}  token_f1={f1:.4f}")
        print("-" * 100)
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(f"SOURCE TEXT ({len(text)} chars):\n{preview}\n")
        print(f"REFERENCE SUMMARY:\n{reference}\n")
        print(f"GENERATED SUMMARY:\n{pred}\n")

    print("=" * 100)
    print(f"\nAverage token F1 over {n} sample(s): {f1_sum / n:.4f}")
    print("(token F1 is a rough word-overlap proxy, not a quality guarantee - read the")
    print(" printed pairs above; a low/zero F1 with a coherent, on-topic summary just")
    print(" means the model paraphrased instead of reusing reference wording.)")
    return 0


def main() -> int:
    args = parse_args()

    adapter_dir = args.adapter_dir.resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    base_model_id = read_base_model_id(adapter_dir, args.model_id)
    print(f"Adapter: {adapter_dir}")
    print(f"Base model: {base_model_id}")

    # Resolve the raw single-text input, if any.
    raw_text = ""
    if args.text_file is not None:
        raw_text = args.text_file.resolve().read_text(encoding="utf-8").strip()
    elif args.text:
        raw_text = args.text.strip()

    if not raw_text and args.source_csv is None and args.jsonl_eval is None:
        raise SystemExit(
            "Nothing to do: pass --text / --text-file for a single page, "
            "--source-csv for batch mode, or --jsonl-eval for a reconstruction test."
        )

    summarizer = Summarizer(
        adapter_dir=adapter_dir,
        base_model_id=base_model_id,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        instruction=args.instruction,
    )

    if args.jsonl_eval is not None:
        return run_jsonl_eval(summarizer, args)
    if raw_text:
        return run_pages(summarizer, raw_text)
    return run_csv(summarizer, args)


if __name__ == "__main__":
    raise SystemExit(main())

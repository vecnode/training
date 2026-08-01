"""Summarize OCR CSV rows using a local Gemma 3 model (no Ollama)."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

CSV_FIELDS = ["image", "summary", "status", "reason", "model"]
# unsloth's mirror of the same Gemma3ForConditionalGeneration weights as
# google/gemma-3-4b-it, but ungated - no Hugging Face license click-through or
# HF_TOKEN needed. Swap to google/gemma-3-4b-it via --model-id if preferred
# once that's been set up.
DEFAULT_MODEL_ID = "unsloth/gemma-3-4b-it"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 150

SYSTEM_PROMPT = "You are summarizing OCR text extracted from one scanned document page."
PROMPT_TEMPLATE = (
    "Write one concise paragraph (max 90 words) describing what this page is about, "
    "including key entities/dates if present.\n\nOCR text:\n{text}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OCR CSV rows with a local Gemma 3 model")
    parser.add_argument("--input", type=Path, required=True, help="Input OCR CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output summaries CSV path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id (gated - requires HF_TOKEN)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Pages summarized together per batch")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pending rows (0 = all)")
    parser.add_argument("--no-resume", action="store_true", help="Recompute existing summaries")
    return parser.parse_args()


def load_processed(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = (row.get("image") or "").strip()
            status = (row.get("status") or "").strip().lower()
            if image and status != "error":
                processed.add(image)
    return processed


def normalize_text(value: str) -> str:
    collapsed = " (newline) ".join(
        part.strip() for part in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip()
    )
    return collapsed or "(empty)"


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{model_id}'. If this is an official google/gemma-3-* "
            f"model id, it is gated on Hugging Face: accept the license at "
            f"https://huggingface.co/{model_id}, then set the HF_TOKEN environment "
            f"variable (or run `huggingface-cli login`) with a token that has been "
            f"granted access. The default model id (unsloth/gemma-3-4b-it) is an "
            f"ungated mirror of the same weights and needs neither. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    return processor, model


def build_conversation(text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": PROMPT_TEMPLATE.format(text=text)}]},
    ]


def summarize_batch(processor, model, texts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    conversations = [build_conversation(text) for text in texts]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    summaries = []
    for row in output_ids:
        decoded = processor.decode(row[input_len:], skip_special_tokens=True)
        summaries.append(normalize_text(decoded))
    return summaries


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        print(f"Input OCR CSV not found: {input_path}", file=sys.stderr)
        return 1

    with input_path.open("r", encoding="utf-8", newline="") as src:
        rows = list(csv.DictReader(src))

    processed = set() if args.no_resume else load_processed(output_path)
    pending = []
    for row in rows:
        image = (row.get("image") or "").strip()
        if not image or image in processed:
            continue
        pending.append(row)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Input rows: {len(rows)} | Pending summaries: {len(pending)}", flush=True)
    if not pending:
        print("Nothing to do.", flush=True)
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available in the current Python environment. "
            "Run uv_setup.bat to install CUDA-enabled PyTorch.",
            file=sys.stderr,
        )
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading {args.model_id}...", flush=True)
    try:
        processor, model = load_model(args.model_id)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() or output_path.stat().st_size == 0:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]

            summarizable = []
            skipped_rows = []
            for row in batch:
                image = (row.get("image") or "").strip()
                text = (row.get("text") or "").strip()
                status = (row.get("status") or "").strip().lower()
                if not text or text == "(no text detected)" or status in {"empty", "error"}:
                    skipped_rows.append((image, status))
                else:
                    summarizable.append((image, text))

            for image, status in skipped_rows:
                writer.writerow(
                    {
                        "image": image,
                        "summary": "",
                        "status": "skipped",
                        "reason": f"ocr_status={status or 'unknown'}",
                        "model": args.model_id,
                    }
                )

            if summarizable:
                t0 = time.time()
                try:
                    summaries = summarize_batch(
                        processor, model, [text for _, text in summarizable], args.max_new_tokens
                    )
                    for (image, _), summary in zip(summarizable, summaries):
                        writer.writerow(
                            {
                                "image": image,
                                "summary": summary,
                                "status": "ok",
                                "reason": "",
                                "model": args.model_id,
                            }
                        )
                except Exception as exc:
                    reason = normalize_text(str(exc))
                    for image, _ in summarizable:
                        writer.writerow(
                            {
                                "image": image,
                                "summary": "",
                                "status": "error",
                                "reason": reason,
                                "model": args.model_id,
                            }
                        )
                elapsed = time.time() - t0
            else:
                elapsed = 0.0

            out.flush()
            done += len(batch)
            per_page = elapsed / max(1, len(summarizable)) if summarizable else 0.0
            print(
                f"[{done}/{len(pending)}] batch of {len(batch)} "
                f"({len(summarizable)} summarized, {len(skipped_rows)} skipped, "
                f"{elapsed:.1f}s, {per_page:.2f}s/page)",
                flush=True,
            )

    total = time.time() - started
    print(f"Done. Wrote {done} row(s) to {output_path} in {total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate synthetic instruction-following QA pairs from OCR text using a
local Gemma 3 model.

Reads a dataset's OCR CSV (same input as summarize_ocr_gemma.py) and asks the
model for 2-3 short question/answer pairs per page ("who is mentioned", "what
date", "what's being requested"), expanding the training data beyond pure
summarization into instruction-following QA over documents. One CSV row per
QA pair (a page can produce zero to --num-qa rows).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

CSV_FIELDS = ["image", "qa_id", "question", "answer", "status", "reason", "model"]
# unsloth's mirror of google/gemma-3-4b-it's weights - ungated, no HF_TOKEN
# needed. Same model already used for summarization and layout description.
DEFAULT_MODEL_ID = "unsloth/gemma-3-4b-it"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 300
DEFAULT_NUM_QA = 3

SYSTEM_PROMPT = (
    "You write instruction-following question/answer pairs about a scanned "
    "document page, based only on its OCR text."
)
PROMPT_TEMPLATE = (
    "Read the OCR text of this page and write exactly {num_qa} short "
    "question/answer pairs a person might ask about it (for example: who is "
    "mentioned, what date, what is being requested or reported). Use only "
    "information present in the text - do not invent facts. "
    "Format your reply exactly as plain lines, nothing else:\n"
    "Q1: <question>\nA1: <answer>\n"
    "Q2: <question>\nA2: <answer>\n"
    "Q3: <question>\nA3: <answer>\n\n"
    "OCR text:\n{text}\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs from an OCR CSV")
    parser.add_argument("--input", type=Path, required=True, help="Input OCR CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Output QA pairs CSV path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Pages processed together per batch (higher = more throughput, more VRAM)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--num-qa", type=int, default=DEFAULT_NUM_QA, help="QA pairs requested per page")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pending rows (0 = all)")
    parser.add_argument("--no-resume", action="store_true", help="Recompute pages already in the output file")
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


def parse_qa_pairs(raw_text: str, num_qa: int) -> list[tuple[str, str]]:
    """Tolerant parser: pull every 'Qn: ...'/'An: ...' line in order, then pair
    them up positionally. Doesn't require strict interleaving or exact counts,
    since small models don't always follow the format perfectly."""
    questions: list[str] = []
    answers: list[str] = []
    for line in raw_text.splitlines():
        m = re.match(r"^\s*Q\d+\s*:\s*(.+?)\s*$", line)
        if m:
            questions.append(m.group(1))
            continue
        m = re.match(r"^\s*A\d+\s*:\s*(.+?)\s*$", line)
        if m:
            answers.append(m.group(1))
    pairs = list(zip(questions, answers))[:num_qa]
    return [(q, a) for q, a in pairs if q and a]


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


def build_conversation(text: str, num_qa: int) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": PROMPT_TEMPLATE.format(num_qa=num_qa, text=text)}]},
    ]


def generate_qa_batch(processor, model, texts: list[str], num_qa: int, max_new_tokens: int) -> list[list[tuple[str, str]]]:
    import torch

    conversations = [build_conversation(text, num_qa) for text in texts]
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
    results = []
    for row in output_ids:
        decoded = processor.decode(row[input_len:], skip_special_tokens=True)
        results.append(parse_qa_pairs(decoded, num_qa))
    return results


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
        text = (row.get("text") or "").strip()
        status = (row.get("status") or "").strip().lower()
        if not image or image in processed:
            continue
        if not text or text == "(no text detected)" or status in {"empty", "error"}:
            continue
        pending.append(row)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Input rows: {len(rows)} | Pending pages: {len(pending)}", flush=True)
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
    print(f"Loading {args.model_id} (batch_size={args.batch_size})...", flush=True)
    processor, model = load_model(args.model_id)

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
            t0 = time.time()
            try:
                batch_pairs = generate_qa_batch(
                    processor, model, [row["text"] for row in batch], args.num_qa, args.max_new_tokens
                )
                for row, pairs in zip(batch, batch_pairs):
                    image = row["image"].strip()
                    if not pairs:
                        writer.writerow(
                            {
                                "image": image,
                                "qa_id": "0",
                                "question": "",
                                "answer": "",
                                "status": "empty",
                                "reason": "no QA pairs parsed from model output",
                                "model": args.model_id,
                            }
                        )
                        continue
                    for idx, (question, answer) in enumerate(pairs, start=1):
                        writer.writerow(
                            {
                                "image": image,
                                "qa_id": str(idx),
                                "question": normalize_text(question),
                                "answer": normalize_text(answer),
                                "status": "ok",
                                "reason": "",
                                "model": args.model_id,
                            }
                        )
            except Exception as exc:
                reason = normalize_text(str(exc))
                for row in batch:
                    writer.writerow(
                        {
                            "image": row["image"].strip(),
                            "qa_id": "0",
                            "question": "",
                            "answer": "",
                            "status": "error",
                            "reason": reason,
                            "model": args.model_id,
                        }
                    )
            elapsed = time.time() - t0
            out.flush()

            done += len(batch)
            per_page = elapsed / max(1, len(batch))
            print(
                f"[{done}/{len(pending)}] batch of {len(batch)} ({elapsed:.1f}s, {per_page:.2f}s/page)",
                flush=True,
            )

    total = time.time() - started
    print(f"Done. Wrote rows for {done} page(s) to {output_path} in {total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

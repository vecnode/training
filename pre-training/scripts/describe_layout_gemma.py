"""Generate image-grounded page-layout descriptions using a local Gemma 3 model.

Unlike the OCR/summarize steps (which work on extracted text), this feeds the
PNG page image itself into Gemma 3's vision tower, so the model reasons about
visual structure - tables, stamps, redaction blocks, handwritten annotations,
letterhead - rather than prose content. Produces (image) -> layout description
training pairs.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

CSV_FIELDS = ["image", "full_path", "status", "reason", "layout_description", "model"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Release_1_PNG"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DATASET_1_LAYOUT.csv"
# unsloth's mirror of google/gemma-3-4b-it's weights - ungated, no HF_TOKEN
# needed. Same model already used for summarization; here it's fed the page
# image instead of just OCR text.
DEFAULT_MODEL_ID = "unsloth/gemma-3-4b-it"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 180

SYSTEM_PROMPT = (
    "You describe the visual structure of a scanned document page image. "
    "Do not transcribe or summarize the prose content."
)
PROMPT_TEXT = (
    "Describe this page's layout in one concise paragraph: tables, stamps, "
    "classification markings, redaction blocks, handwritten annotations, "
    "letterhead, signatures, photos, and general structure (columns, "
    "headers, forms). Only mention features that are actually present."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe page layout/structure to output/DATASET_1_LAYOUT.csv")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing PNG images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N pending images (0 = all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess images even if already present in the output file",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id for the vision-language model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of pages described together per batch (higher = more throughput, more VRAM)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    return parser.parse_args()


def load_processed_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    processed: set[str] = set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return processed
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            if status == "error":
                continue
            full_path = (row.get("full_path") or "").strip()
            image = (row.get("image") or "").strip()
            if image:
                processed.add(image.replace("/", "\\"))
                processed.add(Path(image).name)
            if full_path:
                processed.add(full_path)
                processed.add(Path(full_path).name)
    return processed


def relative_display_path(image_path: Path, image_dir: Path) -> str:
    try:
        rel = image_path.relative_to(image_dir.parent)
    except ValueError:
        rel = image_path.name
    return rel.as_posix().replace("/", "\\")


def normalize_text(value: str) -> str:
    collapsed = " (newline) ".join(
        part.strip() for part in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip()
    )
    return collapsed or "(empty)"


def format_row(
    image_path: Path,
    image_dir: Path,
    status: str,
    reason: str,
    layout_description: str,
    model: str,
) -> dict[str, str]:
    return {
        "image": relative_display_path(image_path, image_dir),
        "full_path": str(image_path.resolve()),
        "status": status,
        "reason": reason,
        "layout_description": normalize_text(layout_description) if layout_description else "",
        "model": model,
    }


def ensure_header(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())


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


def build_conversation(image) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        },
    ]


def describe_batch(processor, model, images: list, max_new_tokens: int) -> list[str]:
    import torch

    conversations = [build_conversation(image) for image in images]
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
    return [processor.decode(row[input_len:], skip_special_tokens=True).strip() for row in output_ids]


def describe_batch_with_fallback(processor, model, image_paths: list[Path], max_new_tokens: int) -> list[tuple[str, str, str]]:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        descriptions = describe_batch(processor, model, images, max_new_tokens)
        return [("ok", "", desc) if desc else ("empty", "no description generated", "") for desc in descriptions]
    except Exception:
        # A single bad image can otherwise sink an entire batch; retry one at a
        # time so the rest of the batch still saves.
        results = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                desc = describe_batch(processor, model, [image], max_new_tokens)[0]
                image.close()
                results.append(("ok", "", desc) if desc else ("empty", "no description generated", ""))
            except Exception as exc:
                results.append(("error", str(exc), ""))
        return results
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_path = args.output.resolve()

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 1

    images = collect_images(image_dir)
    if not images:
        print(f"No PNG files found in {image_dir}", file=sys.stderr)
        return 1

    processed = set() if args.no_resume else load_processed_paths(output_path)
    pending: list[Path] = []
    for image_path in images:
        rel = relative_display_path(image_path, image_dir)
        full = str(image_path)
        if rel in processed or full in processed or image_path.name in processed:
            continue
        pending.append(image_path)

    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Found {len(images)} PNG(s); pending {len(pending)}", flush=True)
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

    ensure_header(output_path)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]
            t0 = time.time()
            batch_results = describe_batch_with_fallback(processor, model, batch, args.max_new_tokens)
            elapsed = time.time() - t0

            for image_path, (status, reason, description) in zip(batch, batch_results):
                writer.writerow(format_row(image_path, image_dir, status, reason, description, args.model_id))
            out.flush()

            done += len(batch)
            per_page = elapsed / max(1, len(batch))
            print(
                f"[{done}/{len(pending)}] batch of {len(batch)} ({elapsed:.1f}s, {per_page:.2f}s/page)",
                flush=True,
            )

    total = time.time() - started
    print(f"Done. Wrote {done} row(s) to {output_path} in {total:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

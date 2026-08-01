"""Extract text from PNG images in Release_1_PNG using Surya OCR."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

CSV_FIELDS = ["image", "full_path", "status", "reason", "method", "confidence", "text"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Release_1_PNG"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DATASET_1_OCR.csv"
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR PNG images to output/DATASET_1_OCR.csv")
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of pages OCR'd together per batch (higher = more throughput, more VRAM)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU inference (slow; Surya otherwise auto-detects CUDA)",
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
            reason = (row.get("reason") or "").strip().lower()
            # Allow reruns to recover pages that previously failed due runtime/config errors.
            if status == "error":
                continue
            if status == "empty" and "ocr error" in reason:
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


def format_row(
    image_path: Path,
    image_dir: Path,
    status: str,
    reason: str,
    method: str,
    confidence: float,
    text: str,
) -> dict[str, str]:
    rel = relative_display_path(image_path, image_dir)
    full = str(image_path.resolve())
    raw = text.strip() if text.strip() else "(no text detected)"
    body = " (newline) ".join(part.strip() for part in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip())
    if not body:
        body = "(no text detected)"
    return {
        "image": rel,
        "full_path": full,
        "status": status,
        "reason": reason,
        "method": method,
        "confidence": f"{confidence:.4f}" if confidence else "0.0000",
        "text": body,
    }


def ensure_header(output_path: Path, image_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()


def normalize_existing_csv(output_path: Path) -> None:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        return

    if all(field in fieldnames for field in CSV_FIELDS):
        return

    upgraded_rows: list[dict[str, str]] = []
    for row in rows:
        image = (row.get("image") or "").strip()
        full_path = (row.get("full_path") or "").strip()
        text = (row.get("text") or "").strip()
        upgraded_rows.append(
            {
                "image": image,
                "full_path": full_path,
                "status": "ok" if text and text != "(no text detected)" else "legacy",
                "reason": "legacy markdown conversion",
                "method": "legacy",
                "confidence": "0.0000",
                "text": text or "(no text detected)",
            }
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(upgraded_rows)


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(image_dir.glob("*.png"), key=lambda p: p.name.lower())


def count_pdfs(image_dir: Path) -> int:
    return sum(1 for _ in image_dir.glob("*.pdf"))


def load_predictors(use_gpu: bool):
    if not use_gpu:
        os.environ.setdefault("TORCH_DEVICE", "cpu")

    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    return recognition_predictor, detection_predictor


def ocr_batch(recognition_predictor, detection_predictor, image_paths: list[Path]) -> list[tuple[str, str, float, str, str]]:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    try:
        predictions = recognition_predictor(images, det_predictor=detection_predictor)
    finally:
        for image in images:
            image.close()

    results = []
    for prediction in predictions:
        lines = [line for line in prediction.text_lines if line.text and line.text.strip()]
        if not lines:
            results.append(("empty", "no text detected", 0.0, "surya", ""))
            continue
        text = "\n".join(line.text.strip() for line in lines)
        confidence = round(sum(float(line.confidence) for line in lines) / len(lines), 4)
        results.append(("ok", "surya", confidence, "surya", text))
    return results


def ocr_batch_with_fallback(recognition_predictor, detection_predictor, image_paths: list[Path]) -> list[tuple[str, str, float, str, str]]:
    try:
        return ocr_batch(recognition_predictor, detection_predictor, image_paths)
    except Exception:
        # A single bad image (corrupt PNG, decode failure) can otherwise sink an
        # entire batch; retry one at a time so the rest of the batch still saves.
        results = []
        for path in image_paths:
            try:
                results.extend(ocr_batch(recognition_predictor, detection_predictor, [path]))
            except Exception as exc:
                results.append(("error", str(exc), 0.0, "", f"[OCR error: {exc}]"))
        return results


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_path = args.output.resolve()

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}", file=sys.stderr)
        return 1

    images = collect_images(image_dir)
    if not images:
        pdf_count = count_pdfs(image_dir)
        if pdf_count > 0:
            print(
                f"No PNG files found in {image_dir}. Found {pdf_count} PDF(s); run convert_pdf_to_png first.",
                file=sys.stderr,
            )
        else:
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

    use_gpu = not args.no_gpu and torch.cuda.is_available()
    if use_gpu:
        print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("Device: CPU", flush=True)

    print(f"Loading Surya OCR (batch_size={args.batch_size})...", flush=True)
    recognition_predictor, detection_predictor = load_predictors(use_gpu)

    normalize_existing_csv(output_path)
    ensure_header(output_path, image_dir)
    started = time.time()
    done = 0

    with output_path.open("a", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start:batch_start + args.batch_size]
            t0 = time.time()
            batch_results = ocr_batch_with_fallback(recognition_predictor, detection_predictor, batch)
            elapsed = time.time() - t0

            for image_path, (status, reason, confidence, method, text) in zip(batch, batch_results):
                writer.writerow(format_row(image_path, image_dir, status, reason, method, confidence, text))
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

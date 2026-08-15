"""Loads the OCR + SUMMARIES CSVs and joins them into training pairs.

A "row" here is one training example: a page that has usable OCR text AND a
reference summary - exactly the pairs the adapter was trained on. The front-end
browses these so you can compare, per page: raw OCR, the reference summary the
model trained against, and the adapter's live inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

csv.field_size_limit(10 ** 7)

_BAD_OCR_STATUS = {"error", "empty", "legacy"}


def _norm_key(value: str) -> str:
    key = (value or "").strip().replace("\\", "/")
    while "//" in key:
        key = key.replace("//", "/")
    return key


def _preview(text: str, n: int = 110) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


class DataStore:
    def __init__(self, ocr_csv: Path, summaries_csv: Path, image_root: Path | None = None) -> None:
        self.ocr_csv = Path(ocr_csv)
        self.summaries_csv = Path(summaries_csv)
        self.image_root = Path(image_root) if image_root else self.ocr_csv.parent.parent
        self.rows: list[dict] = []  # {image, ocr_text, summary, full_path}
        self._load()

    def _load(self) -> None:
        summaries: dict[str, str] = {}
        if self.summaries_csv.exists():
            with self.summaries_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = _norm_key(row.get("image") or "")
                    summary = (row.get("summary") or "").strip()
                    status = (row.get("status") or "").strip().lower()
                    if key and summary and status == "ok":
                        summaries[key] = summary

        if self.ocr_csv.exists():
            with self.ocr_csv.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    key = _norm_key(row.get("image") or "")
                    text = (row.get("text") or "").strip()
                    status = (row.get("status") or "").strip().lower()
                    if not key or not text or text == "(no text detected)":
                        continue
                    if status in _BAD_OCR_STATUS:
                        continue
                    summary = summaries.get(key)
                    if not summary:
                        continue
                    self.rows.append({
                        "image": key,
                        "ocr_text": text,
                        "summary": summary,
                        "full_path": (row.get("full_path") or "").strip(),
                    })

    def count(self) -> int:
        return len(self.rows)

    def list(self, limit: int = 0, offset: int = 0) -> list[dict]:
        items = self.rows[offset:] if limit <= 0 else self.rows[offset: offset + limit]
        out = []
        for i, r in enumerate(items, start=offset):
            out.append({
                "idx": i,
                "image": r["image"],
                "ocr_preview": _preview(r["ocr_text"]),
                "ocr_chars": len(r["ocr_text"]),
                "summary_preview": _preview(r["summary"]),
            })
        return out

    def get(self, idx: int) -> dict | None:
        if 0 <= idx < len(self.rows):
            r = self.rows[idx]
            return {"idx": idx, "image": r["image"], "ocr_text": r["ocr_text"], "summary": r["summary"]}
        return None

    def resolve_image_path(self, idx: int) -> Path | None:
        """Locate the source PNG for a row: prefer the CSV full_path, else
        fall back to <image_root>/<image key>."""
        if not (0 <= idx < len(self.rows)):
            return None
        r = self.rows[idx]
        full_path = r.get("full_path") or ""
        if full_path:
            p = Path(full_path)
            if p.exists():
                return p
        candidate = self.image_root / r["image"]
        if candidate.exists():
            return candidate
        candidate_win = self.image_root / r["image"].replace("/", "\\")
        if candidate_win.exists():
            return candidate_win
        return None

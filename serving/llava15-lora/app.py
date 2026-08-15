"""FastAPI deployment for the OCR -> summary model.

Serves a small front-end (text in, summary out) and a JSON API. The model is
loaded once at startup and reused for every request; GPU generation is
serialized with a lock so concurrent requests are safe.

By default it loads a fused/merged model from merged_model/ if present
(recommended for production - see merge_adapter.py), otherwise it falls back to
the LoRA adapter at ../../fine-tuning/llava15-lora/runs/llava15_lora/final_adapter.

Run (from this folder):

    .venv\\Scripts\\python.exe app.py                 # http://127.0.0.1:8008
    .venv\\Scripts\\python.exe app.py --port 9000 --host 0.0.0.0

or with uvicorn directly:

    .venv\\Scripts\\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8008
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dataset import DataStore
from infer import (
    DEFAULT_ADAPTER,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MERGED,
    Summarizer,
    resolve_source,
)

_DEPLOY_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _DEPLOY_DIR / "static"

# Configurable via environment so the same file works in dev and prod.
_ADAPTER_DIR = os.environ.get("DEPLOY_ADAPTER_DIR", str(DEFAULT_ADAPTER))
_MERGED_MODEL = os.environ.get("DEPLOY_MERGED_MODEL", "")  # empty -> auto-detect merged_model/ here
# No local default: the OCR/SUMMARIES CSVs live in the separate pre-training
# repo's outputs/<timestamp>_<dataset>/ folder. Set these env vars to point at
# that repo if you want the dataset browser (/api/rows etc.) populated - the
# core /api/summarize endpoint works fine without them.
_OCR_CSV = os.environ.get("DEPLOY_OCR_CSV", "")
_SUMMARIES_CSV = os.environ.get("DEPLOY_SUMMARIES_CSV", "")

_summarizer: Summarizer | None = None
_dataset: DataStore | None = None
_gpu_lock = threading.Lock()  # generation is not concurrency-safe on a single GPU


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _summarizer, _dataset
    source = resolve_source(_ADAPTER_DIR, _MERGED_MODEL or None)
    where = source.get("merged_model_dir") or source.get("adapter_dir")
    print(f"Loading model from: {where}")
    _summarizer = Summarizer(max_new_tokens=DEFAULT_MAX_NEW_TOKENS, **source)
    _summarizer.print_summary()  # print + inspect weights on deploy

    print(f"Loading dataset: OCR={_OCR_CSV or '(not set - dataset browser will be empty)'}")
    _dataset = DataStore(Path(_OCR_CSV or "."), Path(_SUMMARIES_CSV or "."))
    print(f"Dataset ready: {_dataset.count()} training pairs (OCR + reference summary)")
    print("Model ready. Front-end at /  |  API at /api/*")
    yield
    print("Shutting down.")


app = FastAPI(title="OCR → Summary (LLaVA LoRA)", version="1.0", lifespan=lifespan)


class SummarizeRequest(BaseModel):
    ocr_text: str = Field(..., description="Raw OCR text of one document page")
    max_new_tokens: int | None = Field(None, description="Override max generated tokens")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "device": _summarizer.device if _summarizer else "loading"}


@app.get("/api/model-info")
def model_info() -> dict:
    if _summarizer is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    return _summarizer.model_info()


@app.get("/api/weights")
def weights(filter: str = "q_proj", limit: int = 12) -> dict:
    """Inspect deployed weights: per-tensor shape/dtype/mean/std/norm."""
    if _summarizer is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    return {"filter": filter, "limit": limit, "tensors": _summarizer.inspect_weights(limit=limit, name_filter=filter)}


@app.get("/api/rows")
def rows(limit: int = 0, offset: int = 0) -> dict:
    """Lightweight list of training pairs (OCR preview + summary preview)."""
    if _dataset is None:
        return JSONResponse({"error": "dataset not loaded"}, status_code=503)
    return {"count": _dataset.count(), "offset": offset, "rows": _dataset.list(limit=limit, offset=offset)}


@app.get("/api/row/{idx}")
def row(idx: int) -> dict:
    """Full OCR text + reference summary for one training pair."""
    if _dataset is None:
        return JSONResponse({"error": "dataset not loaded"}, status_code=503)
    item = _dataset.get(idx)
    if item is None:
        return JSONResponse({"error": f"row {idx} out of range"}, status_code=404)
    return item


@app.get("/api/image/{idx}")
def image(idx: int):
    """Serve the source PNG that was OCR'd for this row."""
    if _dataset is None:
        return JSONResponse({"error": "dataset not loaded"}, status_code=503)
    path = _dataset.resolve_image_path(idx)
    if path is None:
        return JSONResponse({"error": f"image for row {idx} not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/summarize")
def summarize(req: SummarizeRequest) -> dict:
    if _summarizer is None:
        return JSONResponse({"error": "model not loaded"}, status_code=503)
    text = (req.ocr_text or "").strip()
    if not text:
        return JSONResponse({"error": "ocr_text is empty"}, status_code=400)

    t0 = time.time()
    with _gpu_lock:
        if req.max_new_tokens:
            prev = _summarizer.max_new_tokens
            _summarizer.max_new_tokens = req.max_new_tokens
            try:
                summary = _summarizer.summarize(text)
            finally:
                _summarizer.max_new_tokens = prev
        else:
            summary = _summarizer.summarize(text)
    return {"summary": summary, "elapsed_ms": round((time.time() - t0) * 1000, 1), "input_chars": len(text)}


# Mount static assets last so it doesn't shadow the API routes.
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the OCR -> summary model with FastAPI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

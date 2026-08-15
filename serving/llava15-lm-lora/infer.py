"""Standalone real-world inference for the OCR -> summary LoRA adapter.

This module is deliberately independent of the training code: deployment should
not import anything from ../../fine-tuning/llava15-lm-lora/. It loads the base LLaVA 1.5
language model, attaches the trained LoRA adapter (or loads an already-merged
model), and turns raw OCR text into a one-paragraph summary.

Usage (run from this folder):

    # one-off
    .venv\\Scripts\\python.exe infer.py --text "EONFIDENTIAt (newline) FM AMEMBASSY ..."

    # from a file (best for long/noisy pages)
    .venv\\Scripts\\python.exe infer.py --text-file page.txt

    # interactive: paste one OCR page per line, blank line / 'quit' to exit
    .venv\\Scripts\\python.exe infer.py

    # read a single page from stdin
    echo "OCR text..." | .venv\\Scripts\\python.exe infer.py --text-file -
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent
# serving/llava15-lm-lora/ -> repo root -> fine-tuning/llava15-lm-lora/
# (the training pipeline this serving folder serves; a top-level sibling of
# serving/, not a parent - see the repo README for why they're split like this).
_TRAINING_DIR = _DEPLOY_DIR.parent.parent / "fine-tuning" / "llava15-lm-lora"

# Reuse the training HF cache so the base model is not re-downloaded.
os.environ.setdefault("HF_HOME", str(_TRAINING_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ---------------------------------------------------------------------------
# These MUST stay identical to training (../../fine-tuning/llava15-lm-lora/train_llava15_lora.py).
# If the instruction wording or truncation differs from training, quality drops.
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "Summarize this scanned document page in one concise paragraph. "
    "Focus on key entities, dates, events, and any UAP-related content if present.\n\n"
    "OCR text:\n"
)
DEFAULT_ADAPTER = _TRAINING_DIR / "runs" / "llava15_lora" / "final_adapter"
DEFAULT_MERGED = _DEPLOY_DIR / "merged_model"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 220


def resolve_source(adapter_dir: Path | str = DEFAULT_ADAPTER, merged_model: Path | str | None = None) -> dict:
    """Decide which artifact to load. A fused/merged model is preferred for
    production (faster, no PEFT at runtime); fall back to the LoRA adapter."""
    if merged_model:
        return {"merged_model_dir": Path(merged_model)}
    if DEFAULT_MERGED.exists() and any(DEFAULT_MERGED.iterdir()):
        return {"merged_model_dir": DEFAULT_MERGED}
    return {"adapter_dir": Path(adapter_dir)}


def truncate_ocr_ids(ids: list[int], budget: int) -> list[int]:
    """Fit OCR token ids into `budget`, keeping the head and tail of the page."""
    if budget <= 0:
        return []
    if len(ids) <= budget:
        return ids
    head = int(budget * 0.75)
    tail = budget - head
    if tail <= 0:
        return ids[:budget]
    return ids[:head] + ids[-tail:]


def ensure_base_model_cached(base_id: str) -> str:
    """Ensure the public base model is present in the local Hugging Face cache,
    downloading it on first run if missing. The cache dir is HF_HOME (defaults
    to ../../fine-tuning/llava15-lm-lora/hf_cache, git-ignored in both a dev checkout and a
    distributed build - it is resolved relative to this file, so it works in
    either layout).

    Deliberately does NOT pass `cache_dir=` to snapshot_download: doing so makes
    huggingface_hub treat that path as the raw hub-cache root, bypassing its
    normal `<HF_HOME>/hub` layout - which silently creates a second, differently
    laid out copy of the same ~14 GB model alongside the one `from_pretrained()`
    already manages, instead of reusing it. Leaving cache_dir unset makes this
    function resolve the cache exactly the way `from_pretrained(base_id, ...)`
    does later in this same function (both read HF_HOME/HF_HUB_CACHE from the
    environment), so a model already cached by either call is recognized by
    the other with no re-download and no duplicate storage.

    This is a no-op (fast, no network transfer) once the model is cached -
    huggingface_hub checks file hashes/etags before deciding what to fetch. Only
    the base model goes through this path; the trained LoRA adapter is never
    downloaded - it is a local artifact that must already exist on disk (copied
    in, or produced by ../../fine-tuning/llava15-lm-lora/train_llava15_lora.py on this machine).
    """
    from huggingface_hub import snapshot_download

    print(f"Checking base model cache: {base_id} (HF_HOME={os.environ.get('HF_HOME')})")
    try:
        local_path = snapshot_download(repo_id=base_id)
    except Exception as exc:  # network/space/auth failures - fail with a clear message
        raise RuntimeError(
            f"Failed to fetch base model '{base_id}' from Hugging Face Hub.\n"
            f"This is a one-time ~14 GB download into {os.environ.get('HF_HOME')}.\n"
            f"Check internet access and available disk space, then retry.\n"
            f"Underlying error: {exc}"
        ) from exc
    print(f"Base model ready at: {local_path}")
    return local_path


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
    return "llava-hf/llava-1.5-7b-hf"


class Summarizer:
    """Load once, summarize many. Importable as a library.

    >>> s = Summarizer()
    >>> s.summarize("EONFIDENTIAt (newline) FM AMEMBASSY MOSCOW ...")
    'This classified report ...'
    """

    def __init__(
        self,
        adapter_dir: Path | str = DEFAULT_ADAPTER,
        base_model_id: str = "",
        merged_model_dir: Path | str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        device: str | None = None,
    ) -> None:
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if merged_model_dir:
            # Standalone, already-merged model (no PEFT needed at runtime).
            merged = Path(merged_model_dir).resolve()
            self.processor = AutoProcessor.from_pretrained(merged)
            model = LlavaForConditionalGeneration.from_pretrained(merged, torch_dtype=dtype)
        else:
            from peft import PeftModel

            adapter = Path(adapter_dir).resolve()
            if not adapter.exists():
                raise FileNotFoundError(
                    f"Adapter not found: {adapter}\n"
                    "../../fine-tuning/llava15-lm-lora/runs/ is gitignored - copy the trained adapter there "
                    "(or pass --adapter-dir pointing at it)."
                )
            base_id = read_base_model_id(adapter, base_model_id)
            ensure_base_model_cached(base_id)
            self.processor = AutoProcessor.from_pretrained(adapter)
            model = LlavaForConditionalGeneration.from_pretrained(base_id, torch_dtype=dtype)
            model = PeftModel.from_pretrained(model, str(adapter))

        self.tokenizer = self.processor.tokenizer
        self.model = model.to(self.device)
        self.model.eval()

        # Fixed wrapper; only the OCR body is re-tokenized per call.
        self.head_ids = self.tokenizer(f"USER: {INSTRUCTION}", add_special_tokens=True).input_ids
        self.suffix_ids = self.tokenizer(" ASSISTANT: ", add_special_tokens=False).input_ids

    def _build_input_ids(self, ocr_text: str) -> list[int]:
        budget = self.max_length - len(self.head_ids) - len(self.suffix_ids) - self.max_new_tokens
        ocr_ids = self.tokenizer(ocr_text, add_special_tokens=False).input_ids
        ocr_ids = truncate_ocr_ids(ocr_ids, budget)
        return self.head_ids + ocr_ids + self.suffix_ids

    def summarize(self, ocr_text: str) -> str:
        ocr_text = (ocr_text or "").strip()
        if not ocr_text:
            return ""
        ids = self._build_input_ids(ocr_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # -- introspection ------------------------------------------------------

    def model_info(self) -> dict:
        params = list(self.model.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        has_lora = any("lora_" in n for n, _ in self.model.named_parameters())
        vram_gb = None
        if self.device == "cuda":
            vram_gb = round(torch.cuda.memory_reserved() / 1e9, 2)
        return {
            "model_class": self.model.__class__.__name__,
            "mode": "adapter (LoRA attached)" if has_lora else "fused / standalone",
            "device": self.device,
            "dtype": str(params[0].dtype) if params else "n/a",
            "total_params": total,
            "total_params_billions": round(total / 1e9, 3),
            "trainable_params": trainable,
            "vram_reserved_gb": vram_gb,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
        }

    def inspect_weights(self, limit: int = 12, name_filter: str = "") -> list[dict]:
        """Return per-tensor stats (shape/dtype/mean/std/norm) for inspection."""
        rows: list[dict] = []
        for name, p in self.model.named_parameters():
            if name_filter and name_filter not in name:
                continue
            t = p.detach()
            tf = t.float()
            rows.append({
                "name": name,
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "numel": t.numel(),
                "mean": round(tf.mean().item(), 6),
                "std": round(tf.std().item(), 6),
                "l2_norm": round(tf.norm().item(), 4),
                "requires_grad": bool(p.requires_grad),
            })
            if len(rows) >= limit:
                break
        return rows

    def print_summary(self) -> None:
        info = self.model_info()
        print("=" * 64)
        print("  MODEL SUMMARY")
        for k, v in info.items():
            print(f"    {k:>22}: {v}")
        print("  SAMPLE WEIGHTS (language-model attention q_proj):")
        sample = self.inspect_weights(limit=4, name_filter="q_proj")
        for row in sample:
            print(f"    - {row['name']}")
            print(f"        shape={row['shape']} dtype={row['dtype']} "
                  f"mean={row['mean']} std={row['std']} l2={row['l2_norm']}")
        print("=" * 64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize OCR text with the trained LoRA adapter")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER, help="Trained LoRA adapter directory")
    p.add_argument("--merged-model", type=Path, default=None, help="Use a standalone merged model dir (skips PEFT)")
    p.add_argument("--base-model", default="", help="Base model id (auto-detected from adapter config when omitted)")
    p.add_argument("--text", default="", help="OCR text to summarize")
    p.add_argument("--text-file", default="", help="Read OCR text from this file ('-' for stdin)")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Input token budget (match training)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max generated tokens")
    p.add_argument("--inspect", action="store_true", help="Print model summary and sample weights, then continue")
    return p.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file == "-":
        return sys.stdin.read().strip()
    if args.text_file:
        return Path(args.text_file).resolve().read_text(encoding="utf-8").strip()
    return args.text.strip()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.adapter_dir, args.merged_model)
    summarizer = Summarizer(
        base_model_id=args.base_model,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        **source,
    )
    print(f"Loaded on: {summarizer.device}")
    if args.inspect:
        summarizer.print_summary()

    text = _load_text(args)
    if text:
        print("\n=== Summary ===\n")
        print(summarizer.summarize(text))
        print()
        return 0

    # Interactive mode: one OCR page per line (their OCR uses literal "(newline)").
    print("\nInteractive mode. Paste one OCR page and press Enter. Blank line or 'quit' to exit.\n")
    while True:
        try:
            line = input("OCR> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() in {"quit", "exit"}:
            break
        print("\n=== Summary ===\n")
        print(summarizer.summarize(line))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

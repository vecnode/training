"""Merge the LoRA adapter into the base model for standalone deployment.

After merging you get a single self-contained model directory that loads without
PEFT and without the LoRA math at runtime (slightly faster load/inference). The
trade-off: the merged model is the full ~14 GB fp16 LLaVA, versus the ~20 MB
adapter. Use this only if you want a portable, dependency-light artifact.

Run (from this folder):

    .venv\Scripts\python.exe merge_adapter.py --out merged_model

Then serve/infer with it:

    .venv\Scripts\python.exe infer.py --merged-model merged_model --text-file page.txt
    .venv\Scripts\python.exe app.py --merged-model merged_model
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_DEPLOY_DIR = Path(__file__).resolve().parent
# serving/llava15-lora/ -> repo root -> fine-tuning/llava15-lora/
_TRAINING_DIR = _DEPLOY_DIR.parent.parent / "fine-tuning" / "llava15-lora"
os.environ.setdefault("HF_HOME", str(_TRAINING_DIR / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

from infer import DEFAULT_ADAPTER, read_base_model_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--base-model", default="", help="Base model id (auto-detected when omitted)")
    p.add_argument("--out", type=Path, default=_DEPLOY_DIR / "merged_model", help="Output directory")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapter = args.adapter_dir.resolve()
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")

    base_id = read_base_model_id(adapter, args.base_model)
    out = args.out.resolve()
    print(f"Base model : {base_id}")
    print(f"Adapter    : {adapter}")
    print(f"Output     : {out}")

    print("Loading base model (fp16)...")
    model = LlavaForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float16)
    print("Attaching adapter and merging...")
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    AutoProcessor.from_pretrained(adapter).save_pretrained(out)
    print(f"Merged model saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

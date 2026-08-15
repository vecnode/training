# Fine-tuning Workspace

Adapter/LoRA fine-tuning pipelines. Each subfolder is an independent `uv`
project (own `pyproject.toml`/`.venv`) — see its README for setup and data.

| Folder | Framework | Base model |
|---|---|---|
| [`vicuna-7b-lora/`](vicuna-7b-lora/README.md) | `transformers` + `peft` | `lmsys/vicuna-7b-v1.5`, loaded directly (no LLaVA checkpoint, no vision encoder — see its README) |
| [`axolotl-ocr-summary/`](axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) | `Qwen/Qwen2.5-3B-Instruct` (config-driven LoRA/QLoRA) |

## Current focus

`vicuna-7b-lora/` trains on plain text/summary pairs — currently a local
CNN/DailyMail Parquet dump, not OCR output. Loads Vicuna-7B directly rather
than via a LLaVA checkpoint, since only the language model is ever used — a
future `llava15-full-lora/` sibling trained on image+text pairs would be
the first real VLM fine-tune in this repo, and the first place a LLaVA
dependency would actually belong. See
[`vicuna-7b-lora/README.md`](vicuna-7b-lora/README.md) for the dataset
format and commands. `axolotl-ocr-summary/` is unchanged.


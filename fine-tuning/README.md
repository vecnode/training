# Fine-tuning Workspace

Adapter/LoRA fine-tuning pipelines. Each subfolder is an independent `uv`
project (own `pyproject.toml`/`.venv`) — see its README for setup and data.

| Folder | Framework | Base model |
|---|---|---|
| [`llava15-lora/`](llava15-lora/README.md) | `transformers` + `peft` | `llava-hf/llava-1.5-7b-hf` (LoRA on language backbone, text-only) |
| [`axolotl-ocr-summary/`](axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) | `Qwen/Qwen2.5-3B-Instruct` (config-driven LoRA/QLoRA) |

## Current focus

`llava15-lora/` trains on plain text/summary pairs — currently a local
CNN/DailyMail Parquet dump, not OCR output. See
[`llava15-lora/README.md`](llava15-lora/README.md) for the dataset format
and commands. `axolotl-ocr-summary/` is unchanged.


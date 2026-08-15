# Fine-tuning Workspace

Adapter/LoRA fine-tuning pipelines. Each subfolder is an independent `uv`
project (own `pyproject.toml`/`.venv`) — see its README for setup and data.

| Folder | Framework | Base model |
|---|---|---|
| [`llava15-lm-lora/`](llava15-lm-lora/README.md) | `transformers` + `peft` | `llava-hf/llava-1.5-7b-hf`'s language-model backbone only — text-only, no vision encoder, not a VLM fine-tune (see its README) |
| [`axolotl-ocr-summary/`](axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) | `Qwen/Qwen2.5-3B-Instruct` (config-driven LoRA/QLoRA) |

## Current focus

`llava15-lm-lora/` trains on plain text/summary pairs — currently a local
CNN/DailyMail Parquet dump, not OCR output. Named `-lm-` because it only
fine-tunes LLaVA's language-model backbone (no images, no vision encoder) —
a future `llava15-full-lora/` sibling trained on image+text pairs would be
the first real VLM fine-tune in this repo. See
[`llava15-lm-lora/README.md`](llava15-lm-lora/README.md) for the dataset
format and commands. `axolotl-ocr-summary/` is unchanged.


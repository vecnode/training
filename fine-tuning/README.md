# Fine-tuning Workspace

Adapter/LoRA fine-tuning pipelines. Each subfolder is an independent `uv`
project (own `pyproject.toml`/`.venv`) — see its README for setup and data.

| Folder | Framework | Base model |
|---|---|---|
| [`llava15-lora/`](llava15-lora/README.md) | `transformers` + `peft` | `llava-hf/llava-1.5-7b-hf` (LoRA on language backbone, text-only) |
| [`axolotl-ocr-summary/`](axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) | `Qwen/Qwen2.5-3B-Instruct` (config-driven LoRA/QLoRA) |

## Current focus

`llava15-lora/` is being wired up to train against a local CNN/DailyMail
dump instead of (or alongside) the pre-training repo's OCR CSVs — see
[`llava15-lora/README.md`](llava15-lora/README.md#cnn-dailymail-source-current-focus)
for the dataset format and commands. `axolotl-ocr-summary/` is unchanged.


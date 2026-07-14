# fine-tuning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Fine-tuning and serving workspace, scaling from a single RTX 3090 (24GB) up to
multi-GPU.

## Pipelines

| Folder | Task | Framework |
|---|---|---|
| [`axolotl-ocr-summary/`](axolotl-ocr-summary/README.md) | OCR text → summary text, LoRA/QLoRA | [Axolotl](https://axolotl.ai/) |
| [`llava15-lora/`](llava15-lora/README.md) | OCR text → summary text, LLaVA 1.5 7B LoRA (text-only) | `transformers` + `peft` |

## Serving

| Folder | Serves | Framework |
|---|---|---|
| [`deploy/llava15-lora/`](deploy/llava15-lora/README.md) | `llava15-lora/`'s trained adapter | FastAPI |

`deploy/` holds one subfolder per pipeline that has a serving story, each with
its own venv and dependencies — independently deployable, and new pipelines
add a new `deploy/<pipeline>/` folder rather than touching existing ones.

## License

Licensed under the [MIT License](./LICENSE).

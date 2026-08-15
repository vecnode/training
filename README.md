# Training Workspace

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Model Training, Pre-Training, Fine-Tuning and Serving workspace, scaling from a single RTX 3090 (24GB) up to multi-GPU.

## Repository

- [training](./training/)
    - Ongoing
- [fine-tuning](./)
    - Ongoing
- [pre-training](./pre-training/)
    - Convert a PDF dataset to image dataset
    - OCR PNG pages with [Surya OCR](https://github.com/datalab-to/surya)
    - Summarize OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Describe page layout/structure image-grounded with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Generate synthetic QA pairs from OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))

## Datasets

Examples:

- [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)
- [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)


## uv Commands

One `uv` on your PATH drives every pipeline from the repo root — no `cd`
needed. Each folder below is its own `uv` project (own `pyproject.toml`,
own `.venv`, own pinned deps); `--directory <folder>` points `uv run` at
that project's environment *and* working directory in one flag, since some
scripts default to paths relative to where they're run from.

```sh
# --- pre-training: PDF corpus -> OCR/summary/layout/QA CSVs ---
uv run --directory pre-training python scripts/convert_pdf_to_png.py --help
uv run --directory pre-training python scripts/ocr_detection_png.py --help
uv run --directory pre-training python scripts/summarize_ocr_gemma.py --help

# --- fine-tuning/llava15-lora: LLaVA 1.5 7B LoRA (transformers + peft) ---
uv run --directory fine-tuning/llava15-lora python build_llava15_dataset.py --ocr-csv <path> --summaries-csv <path>
uv run --directory fine-tuning/llava15-lora python train_llava15_lora.py --num-epochs 1 --output-dir runs/llava15_lora
uv run --directory fine-tuning/llava15-lora python generate_llava15_lora.py --adapter-dir runs/llava15_lora/final_adapter --ocr-text "..."

# --- fine-tuning/axolotl-ocr-summary: Axolotl LoRA/QLoRA (Qwen2.5) ---
uv run --directory fine-tuning/axolotl-ocr-summary python scripts/prepare_dataset.py --input DATASET/your_file.csv --text-col text --summary-col summary --val-split 0.1
uv run --directory fine-tuning/axolotl-ocr-summary python -m axolotl.cli.train configs/qlora-3090-24gb.yml

# --- serving/llava15-lora: FastAPI inference for the trained adapter ---
uv run --directory serving/llava15-lora python app.py --port 8008
```

`uv run` bootstraps `.venv` and syncs deps on first call, so no separate
install step is required — `uv_setup.bat` / `uv_bootstrap.bat` in each folder
do the same thing and remain for double-click use on Windows. Swap
`--directory` for the pipeline you're touching; nothing here shares a venv
across folders.

## License

Licensed under the [MIT License](./LICENSE).

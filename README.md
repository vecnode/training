# Training Workspace

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Model Training, Pre-Training, Fine-Tuning and Serving workspace, scaling from a single RTX 3090 (24GB) up to multi-GPU.

## Repository

- [training](./training/)
    - Ongoing
- [fine-tuning](./fine-tuning/)
    - Fine-tune [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset
    - Fine-tune [Vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) on [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset
- [pre-training](./pre-training/)
    - OCR PNG pages with [Surya OCR](https://github.com/datalab-to/surya)
    - Summarize OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Describe page layout/structure image-grounded with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
    - Generate synthetic QA pairs from OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))

## Datasets

Text Datasets:

- [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail)
    - text/summary (312k rows)
    - The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization. 

<!--
- [EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)
- [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
-->

## Commands

```sh
# --- pre-training: PDF corpus -> OCR/summary/layout/QA CSVs ---
pre-training\exec_1.bat

# --- fine-tuning/vicuna-7b-lora: Vicuna-7B LoRA (transformers + peft) ---
# text summarization LoRA, loads lmsys/vicuna-7b-v1.5 directly
uv run --directory fine-tuning/vicuna-7b-lora python build_vicuna7b_dataset.py --cnn-dailymail-dir "C:\path\to\cnn_dailymail\3.0.0" --max-samples 2000
uv run --directory fine-tuning/vicuna-7b-lora python train_vicuna7b_lora.py --num-epochs 2 --output-dir runs/vicuna7b_lora
uv run --directory fine-tuning/vicuna-7b-lora python generate_vicuna7b_lora.py --adapter-dir runs/vicuna7b_lora/final_adapter --jsonl-eval data/vicuna7b_train.jsonl --num-samples 5
```

## License

Licensed under the [MIT License](./LICENSE).

# Training Workspace

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

Model Training, Pre-Training, Fine-Tuning and Serving workspace, scaling from a single RTX 3090 (24GB) up to multi-GPU.

## Repository

- [training](./training/)
    - Train Logistic Regression on [uci.edu/adult](https://archive.ics.uci.edu/dataset/2/adult) dataset
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
- [uci.edu/adult](https://archive.ics.uci.edu/dataset/2/adult)
    -  Predict whether annual income of an individual exceeds $50K/yr based on census data. Also known as "Census Income" dataset. 
    - DOI: 10.24432/C5XW20

<!--
- [EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)
- [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum)
- [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
-->

## Commands

```sh
# --- training/adult-income-logreg: logistic regression from scratch (raw numpy) ---
# no scikit-learn/pandas - sigmoid, cross-entropy loss, and gradient descent written by hand
uv run --directory training/adult-income-logreg python build_income_dataset.py --data-dir "C:\path\to\adult" --output-dir data
uv run --directory training/adult-income-logreg python train_logreg.py --data-path data/adult_income.npz --num-epochs 300 --output-dir runs/adult_logreg
uv run --directory training/adult-income-logreg python evaluate_logreg.py --data-path data/adult_income.npz --weights-path runs/adult_logreg/logreg_weights.npz

# --- pre-training: PDF corpus -> OCR/summary/layout/QA CSVs ---
pre-training\exec_1.bat

# --- fine-tuning/vicuna-7b-lora: Vicuna-7B LoRA (transformers + peft) ---
# text summarization LoRA, loads lmsys/vicuna-7b-v1.5 directly
uv run --directory fine-tuning/vicuna-7b-lora python build_vicuna7b_dataset.py --cnn-dailymail-dir "C:\path\to\cnn_dailymail\3.0.0" --max-samples 2000
uv run --directory fine-tuning/vicuna-7b-lora python train_vicuna7b_lora.py --num-epochs 2 --output-dir runs/vicuna7b_lora
uv run --directory fine-tuning/vicuna-7b-lora python generate_vicuna7b_lora.py --adapter-dir runs/vicuna7b_lora/final_adapter --jsonl-eval data/vicuna7b_train.jsonl --num-samples 5

# --- fine-tuning/qwen25-3b-lora: Qwen2.5-3B LoRA (transformers + peft) ---
# same pattern as vicuna-7b-lora, ChatML prompt format
uv run --directory fine-tuning/qwen25-3b-lora python build_qwen3b_dataset.py --cnn-dailymail-dir "C:\path\to\cnn_dailymail\3.0.0" --max-samples 2000
uv run --directory fine-tuning/qwen25-3b-lora python train_qwen3b_lora.py --num-epochs 2 --output-dir runs/qwen3b_lora
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --jsonl-eval data/qwen3b_train.jsonl --num-samples 5
```

## License

Licensed under the [MIT License](./LICENSE).

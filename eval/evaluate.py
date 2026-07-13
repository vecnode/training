"""Generate summaries with a trained adapter and score them against the
held-out DATASET_JSONL/val.jsonl split.

Loads the base model named in the Axolotl config plus the LoRA/QLoRA adapter
next to it (peft), runs generation over every validation record, and reports
ROUGE-L / BLEU. Predictions are dumped to a CSV for manual inspection.

Usage:
    uv run python eval/evaluate.py \
        --config configs/qlora-3090-24gb.yml \
        --adapter-dir output/qlora-3090-24gb \
        --val-file DATASET_JSONL/val.jsonl
"""

import argparse
import json
from pathlib import Path

import evaluate as hf_evaluate
import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Axolotl config used for training")
    parser.add_argument("--adapter-dir", required=True, type=Path, help="Trained adapter directory")
    parser.add_argument("--val-file", default=Path("DATASET_JSONL/val.jsonl"), type=Path)
    parser.add_argument("--out-dir", default=Path("output/eval"), type=Path)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--limit", default=None, type=int, help="Only evaluate the first N records")
    return parser.parse_args()


def load_records(path: Path, limit: int | None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return records[:limit] if limit else records


def format_prompt(record: dict) -> str:
    return f"{record['instruction']}\n\n{record['input']}\n\n### Response:\n"


def main() -> None:
    args = parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_model_name = config["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    records = load_records(args.val_file, args.limit)
    predictions, references = [], []

    for record in records:
        prompt = format_prompt(record)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predictions.append(generated.strip())
        references.append(record["output"].strip())

    rouge = hf_evaluate.load("rouge")
    bleu = hf_evaluate.load("sacrebleu")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])

    metrics = {
        "rougeL": rouge_scores["rougeL"],
        "rouge1": rouge_scores["rouge1"],
        "bleu": bleu_score["score"],
        "n_records": len(records),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    df = pd.DataFrame(
        {
            "input": [r["input"] for r in records],
            "reference": references,
            "prediction": predictions,
        }
    )
    df.to_csv(args.out_dir / "predictions.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Predictions written to {args.out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()

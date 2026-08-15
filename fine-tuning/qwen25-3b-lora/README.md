# qwen25-3b-lora

LoRA fine-tuning of **`Qwen/Qwen2.5-3B-Instruct`** on text/summary pairs.
Same pattern as [`../vicuna-7b-lora/`](../vicuna-7b-lora/README.md) — same
CNN/DailyMail dataset, same `transformers`+`peft` training loop, same
reconstruction-test workflow — swapped to a smaller, faster model. See
[Differences from vicuna-7b-lora](#differences-from-vicuna-7b-lora) below for
the two places the code actually had to change, not just the model name.

**Task:** given an article's raw text, produce a new one-paragraph summary.
Loss is computed only on the summary, and long input is truncated by tokens
(head + tail) so the summary is always preserved in the training budget.

**Data:** CNN/DailyMail article/summary pairs, downloaded locally as Parquet.
This repo does not download CNN/DailyMail itself; it only trains on the files
once you've fetched them. All downloads, cache, checkpoints, and adapters
stay inside this folder.

## Differences from `vicuna-7b-lora`

Everything else (training loop, checkpoint/resume logic, reconstruction
test, dataset builder, CLI shape) is line-for-line the same pattern. Two
things are genuinely different, both verified against the real model/tokenizer
before writing this, not assumed:

1. **Chat format.** Qwen2.5-Instruct was fine-tuned on ChatML, not Vicuna's
   `USER: ... ASSISTANT: ` format. Confirmed via
   `Qwen/Qwen2.5-3B-Instruct`'s `tokenizer_config.json`: `eos_token` is
   `<|im_end|>`, and the chat template wraps each turn as
   `<|im_start|>{role}\n...<|im_end|>`. The collator/generator here build
   `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{instruction}{text}<|im_end|>\n<|im_start|>assistant\n`
   instead.
2. **No `protobuf`/`sentencepiece` dependency.** Unlike
   `lmsys/vicuna-7b-v1.5` (raw SentencePiece tokenizer, needs `protobuf` to
   convert to a fast tokenizer), Qwen2.5-3B-Instruct ships a ready-to-use
   `tokenizer.json` (byte-level BPE) — confirmed by checking the model repo's
   file listing. One less thing to install.

**Same as `vicuna-7b-lora`, unchanged:** LoRA `target_modules=["q_proj",
"v_proj"]` — `Qwen2ForCausalLM` uses the same separate Q/K/V/O linear-layer
naming as Llama/Vicuna (confirmed via `peft`'s own default LoRA
target-module table, which lists `q_proj`/`v_proj` for `qwen2`, identical to
`llama`). Not every small instruct model is this safe a swap — `Phi-3.5-mini`,
for example, fuses Q/K/V into a single `qkv_proj` layer and would need
different `target_modules` to LoRA correctly; that one wasn't built here.

## Dataset — CNN/DailyMail

[`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail)
(config `3.0.0`), a public, no-setup-required news-summarization dataset.

**Where the dataset lives:** downloaded locally to
`C:\Users\luisarandas\Desktop\cnn_dailymail\3.0.0\` (outside this repo, and
outside the root folder — not committed, not moved). Point `--cnn-dailymail-dir`
at wherever you keep it; nothing in this pipeline assumes a fixed location.

**Dataset format.** Hugging Face ships CNN/DailyMail as Parquet, split into
train/validation/test shards:

```
cnn_dailymail/3.0.0/
├── train-00000-of-00003.parquet   \
├── train-00001-of-00003.parquet    } 287,113 rows total, ~772 MB
├── train-00002-of-00003.parquet   /
├── validation-00000-of-00001.parquet   13,368 rows, ~35 MB
└── test-00000-of-00001.parquet         11,490 rows, ~30 MB
```

Total: **311,971 rows, ~799 MB on disk** (compressed Parquet). `article`
(avg ~3,950 chars) → `text`, `highlights` (avg ~260 chars) → `summary`.

**Why `--max-samples` instead of the full 287k-row train split:** a smaller
model than Vicuna-7B means faster steps, but the same time/size trade-off
applies — pick a sample count you're willing to spend the wall-clock on.
Being ~2.3x smaller than Vicuna-7B (3B vs 7B params), expect meaningfully
faster steps here; a 2,000-sample/2-epoch run that took Vicuna-7B ~1 hour
should take noticeably less on this model, so a bigger `--max-samples` is
more affordable in the same time budget.

## 1) Install deps (once)

```bash
uv_setup.bat
```

Installs project dependencies (`transformers`, `peft`, `accelerate`,
`pandas`, `pyarrow`) through `uv sync`.

## 2) Build JSONL dataset

```bash
uv run --directory fine-tuning/qwen25-3b-lora python build_qwen3b_dataset.py --cnn-dailymail-dir "C:\Users\luisarandas\Desktop\cnn_dailymail\3.0.0" --cnn-dailymail-split train --max-samples 2000
```

Output is `data/qwen3b_train.jsonl`. Training consumes the `text` and
`summary` fields (`prompt`/`source`/`id` are kept for reference but ignored
by the trainer).

## 3) Smoke test (must pass before a full run)

```bash
uv run --directory fine-tuning/qwen25-3b-lora python train_qwen3b_lora.py --max-samples 256 --num-epochs 1
```

The trainer's default `--instruction` already matches the prompt
`build_qwen3b_dataset.py` used to build the JSONL, so nothing extra is
needed for CNN/DailyMail.

Expected: `trainable params` line shows a non-zero count (confirms LoRA
actually attached to `q_proj`/`v_proj`), train loss decreases, `eval_loss`
is numeric (not `nan`).

## 4) Full training with checkpoints

```bash
uv run --directory fine-tuning/qwen25-3b-lora python train_qwen3b_lora.py --num-epochs 2 --output-dir runs/qwen3b_lora
```

**Resume and train more epochs.** `--resume-from-checkpoint last` picks up the
newest `checkpoint-*` in the output dir; `--extra-epochs N` adds `N` epochs
*on top of* the epoch already reached:

```bash
uv run --directory fine-tuning/qwen25-3b-lora python train_qwen3b_lora.py --output-dir runs/qwen3b_lora --resume-from-checkpoint last --extra-epochs 2
```

Checkpoint files are written under `runs/qwen3b_lora/`, including
`latest_checkpoint.txt`, `resume_command.txt`, and `final_adapter/`.

## 5) Reconstruction test — verify quality, not just loss

Print N held-out examples (input / reference summary / generated summary
side by side). Replicates `train_qwen3b_lora.py`'s train/val split (same
`--seed`/`--val-ratio`), so these are genuinely unseen examples:

```bash
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --jsonl-eval data/qwen3b_train.jsonl --num-samples 5
```

Read the printed pairs, don't just look at the token-F1 number — it's a
rough word-overlap proxy; a coherent, accurate, on-topic summary with a
*low* F1 usually just means the model paraphrased instead of reusing
reference wording.

## 6) Generate a summary from new source text

```bash
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --text "LONDON, England (Reuters) -- ... your raw article text here ..."
```

From a file (best for long articles):

```bash
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --text-file my_article.txt
```

## 7) Batch evaluate against reference summaries (optional)

```bash
uv run --directory fine-tuning/qwen25-3b-lora python generate_qwen3b_lora.py --adapter-dir runs/qwen3b_lora/final_adapter --source-csv path\to\articles.csv --reference-csv path\to\references.csv --out-csv runs/qwen3b_lora/generated.csv --max-rows 200
```

## Serving

No dedicated `serving/qwen25-3b-lora/` folder yet — `serving/vicuna-7b-lora/`
is Vicuna-specific (its `infer.py` loads `AutoModelForCausalLM` against a
Vicuna-shaped adapter's config, but the ChatML wrapper differs, same as the
trainer above). Add a sibling `serving/qwen25-3b-lora/` following the same
pattern if/when this adapter needs production inference.

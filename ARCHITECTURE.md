# ARCHITECTURE.md

Technical map of this repository: what each stage does, how data flows
between them, and what's planned next. `AGENTS.md` is the conventions/how-to-run
guide; this file is the "what is this and why is it shaped this way" guide.

## Overview

This repository is a staged model-training workspace, split by pipeline stage
rather than by model:

```
pre-training/   PDF corpus  -> OCR text, summaries, layout, synthetic QA (CSVs)
fine-tuning/    OCR/summary CSVs (or any text/summary pairs) -> trained LoRA adapters
serving/        trained adapters -> inference (FastAPI)
training/       reserved for from-scratch / non-LoRA training (not yet built)
```

Each leaf folder (`pre-training/`, `fine-tuning/<pipeline>/`,
`serving/<pipeline>/`) is an independent `uv` project: its own
`pyproject.toml`, its own `.venv`, its own pinned dependency set (in
particular, its own CUDA torch build). Nothing is shared at runtime between
folders — a pipeline can be deleted or reworked without touching its
siblings. One `uv` binary at the root drives all of them via
`uv run --directory <folder> ...` (see the root `README.md` for the full
command list); there is deliberately no root-level Python project or shared
virtualenv, since the folders pin conflicting dependency versions (e.g.
different torch builds) that a single shared resolution would fight.

## Stage 1 — `pre-training/`

Turns a PDF corpus into training data. Local, GPU-first, Surya OCR +
Gemma 3 (`unsloth/gemma-3-4b-it`, an ungated mirror — no `HF_TOKEN` needed).
Five steps (`exec_1.bat` … `exec_5.bat`, or `main.bat` for an interactive
menu): PDF → PNG pages → OCR CSV → summary CSV / layout CSV / synthetic-QA
CSV, all written per-run to `outputs/[timestamp]_[dataset]/`.

**Status:** out of scope for current work — left as-is.

## Stage 2 — `fine-tuning/`

Two independent example pipelines, both currently framed as
**OCR text → summary text**, both LoRA-based, both sized for a single
RTX 3090 (24GB):

| Pipeline | Framework | Base model | Data shape |
|---|---|---|---|
| [`fine-tuning/llava15-lora/`](fine-tuning/llava15-lora/README.md) | `transformers` + `peft` (manual `Trainer` loop) | `llava-hf/llava-1.5-7b-hf` — LoRA on `q_proj`/`v_proj` of the language backbone only. Vision tower is not exercised; this is a text-only fine-tune of a multimodal model. | JSONL with `ocr_text` / `summary` fields |
| [`fine-tuning/axolotl-ocr-summary/`](fine-tuning/axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) (config-driven) | `Qwen/Qwen2.5-3B-Instruct` (full LoRA) or 7-8B (QLoRA) — no vision component | Alpaca-shape JSONL: `{"instruction", "input", "output"}` |

Both currently expect their input as a **CSV with a source-text column and a
target-summary column**, produced by `pre-training/`'s OCR + summarize steps
(`llava15-lora/build_llava15_dataset.py` reads the OCR/SUMMARIES CSV pair
directly; `axolotl-ocr-summary/scripts/prepare_dataset.py` reads any CSV with
`--text-col`/`--summary-col` flags). Neither pipeline is hard-wired to
`pre-training/`'s output specifically — any CSV with the right two columns
works, which is what makes the CNN/DailyMail integration below a data-prep
problem, not a code-architecture problem.

## Stage 3 — `serving/`

One `serving/<pipeline>/` folder per fine-tuning pipeline that has a serving
story. Currently: [`serving/llava15-lora/`](serving/llava15-lora/README.md),
a FastAPI service (`app.py`) that loads the base LLaVA model + trained
adapter (or a fused/merged model) once and serves a JSON API plus a
dataset-browser front-end. Deliberately decoupled from
`fine-tuning/llava15-lora/` — it only reads the trained adapter directory
(`../../fine-tuning/llava15-lora/runs/llava15_lora/final_adapter`), never
imports its training code.

## Stage 4 — `training/`

Reserved for from-scratch / non-LoRA training of other models, as distinct
from adapting an existing checkpoint (`fine-tuning/`). Not yet designed.

## Datasets

None of the fine-tuning pipelines ship data — `DATASET/`, `data/`, `runs/`,
`output/` etc. are all git-ignored, drop-zone folders. Today's expected
source is `pre-training/`'s own OCR/SUMMARIES CSV output, but the actual
requirement is narrower: any two-column (source-text, target-summary) table.

Example small, permissively-licensed public datasets that fit that shape are
listed in the root `README.md`'s **Datasets** section.

## Next steps — wiring in CNN/DailyMail

Goal: exercise both fine-tuning pipelines against
[`abisee/cnn_dailymail`](https://huggingface.co/datasets/abisee/cnn_dailymail)
(`article` → `highlights`) as a stand-in for real OCR/summary data, without
requiring the dataset to live inside the repo.

Planned, not yet implemented (dataset isn't downloaded yet, so this is
un-tested):

1. **Keep the dataset off the root and off git.** The user downloads/caches
   `cnn_dailymail` wherever they like (e.g. via `huggingface_hub` /
   `datasets.load_dataset`, or a plain CSV export) and points each pipeline
   at it with an existing or new CLI flag — no new folder convention needed,
   this already matches how `DATASET/` and `data/` work today (git-ignored
   drop zones, path passed on the command line).
2. **`fine-tuning/axolotl-ocr-summary/`** — `scripts/prepare_dataset.py`
   already accepts an arbitrary CSV + column names, so once CNN/DailyMail is
   exported to CSV (`article`, `highlights` columns), it's a straight:
   ```sh
   uv run --directory fine-tuning/axolotl-ocr-summary python scripts/prepare_dataset.py \
     --input <path-to-cnn_dailymail>.csv --text-col article --summary-col highlights --val-split 0.1
   ```
   No code change required for this pipeline — CSV-in is already the
   contract.
3. **`fine-tuning/llava15-lora/`** — `build_llava15_dataset.py` currently
   hard-assumes the pre-training repo's two-CSV (`-OCR.csv` /
   `-SUMMARIES.csv`, joined by `full_path`) shape. CNN/DailyMail is a single
   table with both columns together, so this pipeline needs either (a) a
   small new `--single-csv --text-col --summary-col` mode added to
   `build_llava15_dataset.py`, or (b) a separate
   `build_llava15_dataset_from_csv.py` that emits the same
   `data/llava15_train.jsonl` (`ocr_text`/`summary` fields) shape. Option (a)
   is preferred — one script, one JSONL contract, less to keep in sync.
4. **Smoke-test sizing.** CNN/DailyMail articles are longer and cleaner than
   scanned-OCR text — worth re-checking the head+tail token-truncation
   budget in both trainers against real CNN/DailyMail article lengths before
   trusting loss curves, since it was tuned against noisy OCR text.
5. **Do this only after the dataset is actually downloaded and inspected** —
   column names/config (`"3.0.0"` is the standard CNN/DailyMail config) and
   row count should be confirmed against the real files before touching
   `build_llava15_dataset.py`, rather than guessing the schema now.

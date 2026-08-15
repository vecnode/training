# ARCHITECTURE.md

Technical map of this repository: what each stage does, how data flows
between them, and what's planned next. `AGENTS.md` is the conventions/how-to-run
guide; this file is the "what is this and why is it shaped this way" guide.

## Overview

This repository is a staged model-training workspace, split by pipeline stage
rather than by model:

```
pre-training/   PDF corpus  -> OCR text, summaries, layout, synthetic QA (CSVs)
fine-tuning/    text/summary pairs -> trained LoRA adapters
serving/        trained adapters -> inference (FastAPI)
training/       reserved for from-scratch / non-LoRA training (not yet built)
```

Each leaf folder (`pre-training/`, `fine-tuning/<pipeline>/`,
`serving/<pipeline>/`) is an independent `uv` project: its own
`pyproject.toml`, `uv.lock`, `.python-version`, and pinned dependency set (in
particular, its own CUDA torch build). Nothing is shared at runtime between
folders — a pipeline can be deleted or reworked without touching its
siblings. One `uv` binary at the root drives all of them via
`uv run --directory <folder> ...` (see the root `README.md`'s **uv Commands**
section for the current, verified list); there is deliberately no root-level
Python project or shared virtualenv, since the folders pin conflicting
dependency versions (e.g. different torch builds) that a single shared
resolution would fight.

Repo-wide hygiene is intentionally centralized rather than per-folder: **one**
`.gitignore` at the root (unanchored patterns match every project's drop-zone
folders at any depth), and **one** `AGENTS.md`/`CLAUDE.md` pair at the root
covering conventions for the whole repo. No subfolder should have its own
copy of any of these.

### Why `.python-version` matters here

Every project pins `.python-version` to `3.12`. Without it, `uv run` picks
the newest CPython it can find (e.g. `3.14`), and some pinned dependencies —
`pillow==10.4.0` in particular — have no prebuilt wheel for that new a
version yet, so `uv` falls back to a from-source build that fails on Windows
(missing zlib headers). Pinning `3.12` (already installed and known to have
wheels for every pinned dependency across all four projects) is what makes
`uv run --directory <folder> ...` work reproducibly from a clean checkout.
This was verified by reproducing the failure and fixing it during this pass —
see the **Verified working** list below.

## Stage 1 — `pre-training/`

Turns a PDF corpus into training data. Local, GPU-first, Surya OCR +
Gemma 3 (`unsloth/gemma-3-4b-it`, an ungated mirror — no `HF_TOKEN` needed).
Five steps (`exec_1.bat` … `exec_5.bat`, or `main.bat` for an interactive
menu): PDF → PNG pages → OCR CSV → summary CSV / layout CSV / synthetic-QA
CSV, all written per-run to `outputs/[timestamp]_[dataset]/`.

Step 1 (PDF → PNG) is **not** a standalone Python entry point — it's
`scripts/convert_pdf_to_png.ps1`, which shells out to `poppler`
(`pdftoppm`/`pdfinfo` on `PATH`) and a small `compress_png_max.py` helper.
Run it via `exec_1.bat` or the `.ps1` directly, not `uv run ... python
scripts/convert_pdf_to_png.py` — that file doesn't exist (an earlier version
of this doc incorrectly assumed it did; fixed here). Steps 2–5
(`ocr_detection_png.py`, `summarize_ocr_gemma.py`, `describe_layout_gemma.py`,
`generate_qa_gemma.py`) are genuine `argparse` Python scripts and do run via
`uv run --directory pre-training python scripts/<name>.py`.

**Status:** out of scope for active work — left as-is, beyond the
`.gitignore`/`.python-version` consolidation described above.

## Stage 2 — `fine-tuning/`

Two independent example pipelines, both LoRA-based, both sized for a single
RTX 3090 (24GB):

| Pipeline | Framework | Base model | Data shape |
|---|---|---|---|
| [`fine-tuning/llava15-lora/`](fine-tuning/llava15-lora/README.md) | `transformers` + `peft` (manual `Trainer` loop) | `llava-hf/llava-1.5-7b-hf` — LoRA on `q_proj`/`v_proj` of the language backbone only. Vision tower is not exercised; this is a text-only fine-tune of a multimodal model. | JSONL with `text` / `summary` fields |
| [`fine-tuning/axolotl-ocr-summary/`](fine-tuning/axolotl-ocr-summary/README.md) | [Axolotl](https://axolotl.ai/) (config-driven) | `Qwen/Qwen2.5-3B-Instruct` (full LoRA) or 7-8B (QLoRA) — no vision component | Alpaca-shape JSONL: `{"instruction", "input", "output"}` |

**`llava15-lora/` is a generic text-summarization LoRA, not OCR-specific** —
its interface, data source, and default prompt were all cleaned up this pass
to reflect that:

- JSONL field is `text` (was `ocr_text`); `build_llava15_dataset.py` and
  `generate_llava15_lora.py`'s flags are `--source-csv`/`--text`/`--text-file`
  (were `--ocr-csv`/`--ocr-text`/`--ocr-text-file`).
- `build_llava15_dataset.py` **only** builds from a CNN/DailyMail Parquet
  dump now (`--cnn-dailymail-dir`, required) — the earlier dual-source mode
  that also read pre-training's image-linked OCR/SUMMARIES CSV pair
  (`normalize_image_key`/`resolve_image_path`/`load_summaries`) was removed
  entirely, not just renamed, since it's not needed for this pipeline's
  current use (`generate_llava15_lora.py`'s `--source-csv` batch-eval mode
  still accepts any generic CSV with a `text` column, unrelated to that
  removed ingestion path).
- `train_llava15_lora.py`/`generate_llava15_lora.py`'s `DEFAULT_INSTRUCTION`
  is now the CNN/DailyMail news-article wording (was "Summarize this scanned
  document page... UAP-related content") — since that's the only source this
  pipeline builds from, `--instruction` no longer needs to be passed
  explicitly for the common case.

`axolotl-ocr-summary/scripts/prepare_dataset.py` is unrelated to this and
still reads any CSV via `--input --text-col --summary-col` (Windows note
below) — it was not touched.

**Platform note — `axolotl-ocr-summary/` is Linux/WSL-only for now.**
`axolotl[deepspeed]` pulls in `triton`, which publishes no Windows wheels;
`uv run --directory fine-tuning/axolotl-ocr-summary ...` fails at
environment-resolution time on native Windows (verified this pass). This is
a real constraint of that dependency stack, not a bug — not something to
patch around inside that project without being asked.

### CNN/DailyMail — wired in and verified

Downloaded locally to `C:\Users\luisarandas\Desktop\cnn_dailymail\3.0.0\`
(outside the repo, outside the root folder, gitignored regardless). Measured
against the actual files:

| Split | Rows | Size |
|---|---|---|
| train (3 shards) | 287,113 | ~772 MB |
| validation | 13,368 | ~35 MB |
| test | 11,490 | ~30 MB |
| **Total** | **311,971** | **~799 MB** |

`article` (avg ~3,950 chars) → `text`, `highlights` (avg ~260 chars) →
`summary`. `build_llava15_dataset.py --cnn-dailymail-dir ... --max-samples
2000` was run end-to-end against the real files and produces valid JSONL
records; full details and commands in
[`fine-tuning/llava15-lora/README.md`](fine-tuning/llava15-lora/README.md).

## Stage 3 — `serving/`

One `serving/<pipeline>/` folder per fine-tuning pipeline that has a serving
story. Currently: [`serving/llava15-lora/`](serving/llava15-lora/README.md),
a FastAPI service (`app.py`) that loads the base LLaVA model + trained
adapter (or a fused/merged model) once and serves a JSON API plus a
dataset-browser front-end. Deliberately decoupled from
`fine-tuning/llava15-lora/` — it only reads the trained adapter directory
(`../../fine-tuning/llava15-lora/runs/llava15_lora/final_adapter`), never
imports its training code. `uv run --directory serving/llava15-lora python
app.py --help` verified working.

## Stage 4 — `training/`

Reserved for from-scratch / non-LoRA training of other models, as distinct
from adapting an existing checkpoint (`fine-tuning/`). Not yet designed.

## Datasets

None of the fine-tuning pipelines ship data — `DATASET/`, `data/`, `runs/`,
`output/` etc. are all git-ignored, drop-zone folders (via the single root
`.gitignore`). `llava15-lora/` trains on CNN/DailyMail; `axolotl-ocr-summary/`
accepts any matching CSV. Example small, permissively-licensed public
datasets are listed in the root `README.md`'s **Datasets** section.

## Verified working (this pass)

`uv run --directory <folder> python <script> --help` was actually executed,
not assumed, for:

- `pre-training` — `scripts/ocr_detection_png.py`, `scripts/summarize_ocr_gemma.py`
- `fine-tuning/llava15-lora` — `build_llava15_dataset.py` (including a real
  5-row CNN/DailyMail smoke test), `train_llava15_lora.py`,
  `generate_llava15_lora.py`
- `serving/llava15-lora` — `app.py` (also fixed a `SyntaxWarning` from
  unescaped backslashes in three docstrings: `app.py`, `merge_adapter.py`,
  `inspect_weights.py`)

Confirmed **not** working on native Windows, by design of the dependency,
not a bug here:

- `fine-tuning/axolotl-ocr-summary` — `triton` has no Windows wheel (see
  platform note above). Works under WSL/Linux; not attempted here.

Not executed this pass (no PDFs/poppler set up in this environment):

- `pre-training/exec_1.bat` / `scripts/convert_pdf_to_png.ps1`

## Next steps

- `fine-tuning/llava15-lora` on CNN/DailyMail is ready to actually train —
  next action is a real run (`train_llava15_lora.py --max-samples <N>`, no
  `--instruction` override needed since the default now matches), not more
  plumbing.
- `axolotl-ocr-summary`'s Windows/WSL split should be decided explicitly
  (document as WSL-only vs. investigate a triton-free deepspeed config)
  before it's a blocker for anyone following the root README on Windows.
- `training/` (from-scratch, non-LoRA) is still undesigned.

# Dataset Pre-Training Workspace

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Local, GPU-first pipeline that turns a PDF corpus into training data.

- Convert a PDF dataset
- OCR PNG pages with [Surya OCR](https://github.com/datalab-to/surya)
- Summarize OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
- Describe page layout/structure image-grounded with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))
- Generate synthetic QA pairs from OCR text with ([unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it))

## Folder structure

- `DATASET/`: Input PDFs (example: `Release_1/`).
- `outputs/[timestamp]_[dataset]/`: everything produced from one PDF dataset,
  self-contained in a single folder:
  - `[slug]-page-[n].png` — the converted pages.
  - `[timestamp]_[dataset]-OCR.csv` — OCR text per page (Surya OCR).
  - `[timestamp]_[dataset]-SUMMARIES.csv` — per-page summary (local Gemma 3).
  - `[timestamp]_[dataset]-LAYOUT.csv` — per-page layout description, image-grounded (local Gemma 3).
  - `[timestamp]_[dataset]-QA.csv` — synthetic QA pairs per page, one row per pair (local Gemma 3).

Keeping the PNGs and their CSVs in the same timestamped folder means each
`outputs/[timestamp]_[dataset]/` is a complete, portable unit for that run, and
re-running a step against the same folder resumes instead of colliding with a
different run of a similarly-named dataset.

`outputs/` is where all generated PNGs and CSVs go, and none of it is
committed to git — only `outputs/README.md` is tracked; everything else in
that folder is git-ignored.


## Run Files

```bat
uv_setup.bat     :: create/sync local venv, install CUDA torch, validate CUDA
exec_1.bat       :: Step 1 - Convert PDF dataset to PNG pages (resumable)
exec_2.bat       :: Step 2 - OCR PNG pages with Surya OCR (resumable)
exec_3.bat       :: Step 3 - Summarize OCR with local Gemma 3 (resumable)
exec_4.bat       :: Step 4 - Describe page layout from PNGs with local Gemma 3 (resumable)
exec_5.bat       :: Step 5 - Generate synthetic QA pairs with local Gemma 3 (resumable)
main.bat         :: interactive menu covering all pipeline steps
```

Each `exec_N.bat` is a standalone, double-clickable entry point for one
pipeline step: it bootstraps the env via `uv_setup.bat`, then runs the
matching script under `scripts/`. `main.bat` still covers the full menu,
including steps that don't have an `exec_N.bat` yet.

All operational batch and Python scripts are now under `scripts/`.


## License

Licensed under the [MIT License](./LICENSE)
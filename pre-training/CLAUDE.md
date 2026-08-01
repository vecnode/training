# CLAUDE.md

Guidance for Claude Code in this repository. The full agent/contributor guide is
in `AGENTS.md` — read it first.

@AGENTS.md

## Claude-specific quick reference

- This repo is a **local GPU pipeline** that turns a PDF corpus into training
  data: PDF → PNG pages → OCR text → per-page summary / layout description /
  synthetic QA. It does not train or serve a model itself — that lives in the
  separate [`fine-tuning`](https://github.com/vecnode/fine-tuning) repo, which
  trains on the CSVs this repo produces.
- **Env:** uv only, GPU-first. `uv_setup.bat` sets up `.venv` with CUDA torch;
  each pipeline step also has a standalone `exec_N.bat` at the project root
  (`exec_1.bat` = convert PDFs, `exec_2.bat` = OCR, `exec_3.bat` = summarize,
  `exec_4.bat` = layout description (image-grounded), `exec_5.bat` = synthetic
  QA pairs — the last three all local Gemma 3) that bootstraps the env itself
  and can be double-clicked directly; `main.bat` runs the interactive menu for
  all steps.
- **Per-run outputs:** each `exec_1.bat` run creates `outputs/[timestamp]_[dataset]/`
  holding that run's PNGs plus `[timestamp]_[dataset]-OCR.csv` /
  `-SUMMARIES.csv` / `-LAYOUT.csv` / `-QA.csv` once `exec_2`-`exec_5` run
  against it — everything for one dataset run stays in one folder, and
  re-running a step against the same folder resumes instead of colliding with
  a different run.
- **`exec_3`/`exec_4`/`exec_5`'s default model is `unsloth/gemma-3-4b-it`** —
  an ungated mirror of `google/gemma-3-4b-it`'s weights, chosen specifically
  to avoid the Hugging Face license click-through / `HF_TOKEN` setup the
  official repo requires. `--model-id` can still point at
  `google/gemma-3-4b-it` if you've set that up, but it's not the default for
  a reason. `exec_4` (layout) is the only step that actually feeds it images —
  `exec_3`/`exec_5` use it text-only despite it being multimodal-capable.
- **Don't** let `uv sync` resolve CPU-only torch (respect the pinned CUDA index
  in `pyproject.toml`), or commit weights/CSVs/PDFs/PNGs (all git-ignored).

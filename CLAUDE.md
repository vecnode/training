# CLAUDE.md

Guidance for Claude Code in this repository. The full agent/contributor guide
is in `AGENTS.md` — read it first. One `AGENTS.md`/`CLAUDE.md` pair at the
root covers the whole repo; subfolders don't get their own.

@AGENTS.md

## Claude-specific quick reference

- Staged workspace: `pre-training/` (PDF corpus → OCR/summary/layout/QA CSVs,
  currently out of scope for active work) → `fine-tuning/` (LoRA adapters,
  current focus) → `serving/` (inference) → `training/` (reserved,
  from-scratch training, not yet built). Full map in `ARCHITECTURE.md`.
- **Env:** `uv` only, one binary at the root drives every pipeline via
  `uv run --directory <folder> <command>` — see `README.md`'s uv Commands
  section for the exact list. Each leaf folder pins its own `.python-version`
  (currently `3.12` everywhere — newer CPython builds don't have prebuilt
  wheels yet for some pinned deps like `pillow`, which breaks `uv run` with a
  source-build failure) and its own CUDA torch build; never introduce a
  shared root Python environment.
- **`fine-tuning/llava15-lora/`** trains a LoRA adapter on
  `llava-hf/llava-1.5-7b-hf`'s language backbone only (vision tower unused) —
  it's a generic text-summarization fine-tune, not OCR-specific. It builds
  its JSONL from a local CNN/DailyMail Parquet dump only
  (`build_llava15_dataset.py --cnn-dailymail-dir`, required) — the earlier
  mode that also read pre-training's image-linked OCR/SUMMARIES CSV pair was
  removed entirely, not just renamed. CLI flags and the JSONL field are named
  generically (`--source-csv`, `--text`, `--text-file`, JSONL field `text`)
  rather than `ocr_*`, on purpose — don't reintroduce `ocr_*` naming or bring
  back the removed CSV-pair ingestion path without being asked. The default
  `--instruction` now matches the CNN/DailyMail prompt, so it doesn't need to
  be passed explicitly for the common case; it's still overridable (both
  trainer and generator, must match between the two) for other wording.
- **`fine-tuning/axolotl-ocr-summary/`** currently only resolves its `uv`
  environment on Linux/WSL — `axolotl[deepspeed]` pulls in `triton`, which
  has no Windows wheels. Don't try to "fix" this by editing that project's
  `pyproject.toml` without being asked; it's a real platform constraint, not
  a misconfiguration.
- **Never commit data or weights** — one root `.gitignore` covers every
  pipeline's drop-zone folders (`DATASET/`, `data/`, `runs/`, `output/`,
  `outputs/`, `.cache/`, `hf_cache/`, `merged_model/`, model/checkpoint
  binaries). Don't add per-folder `.gitignore` files.

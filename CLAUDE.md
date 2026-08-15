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
  it's a generic text-summarization fine-tune, not OCR-specific. Its CLI
  flags and JSONL field are named generically (`--source-csv`, `--text`,
  `--text-file`, JSONL field `text`) rather than `ocr_*`, on purpose — don't
  reintroduce `ocr_*` naming into this pipeline's general-purpose interface.
  Two interchangeable data sources: pre-training's OCR/SUMMARIES CSVs, or a
  local CNN/DailyMail parquet dump via
  `build_llava15_dataset.py --cnn-dailymail-dir`. The instruction wrapper is
  a CLI flag (`--instruction`, both trainer and generator) — it must match
  between training and generation, and the OCR-flavored default is wrong for
  non-OCR sources like CNN/DailyMail.
- **`fine-tuning/axolotl-ocr-summary/`** currently only resolves its `uv`
  environment on Linux/WSL — `axolotl[deepspeed]` pulls in `triton`, which
  has no Windows wheels. Don't try to "fix" this by editing that project's
  `pyproject.toml` without being asked; it's a real platform constraint, not
  a misconfiguration.
- **Never commit data or weights** — one root `.gitignore` covers every
  pipeline's drop-zone folders (`DATASET/`, `data/`, `runs/`, `output/`,
  `outputs/`, `.cache/`, `hf_cache/`, `merged_model/`, model/checkpoint
  binaries). Don't add per-folder `.gitignore` files.

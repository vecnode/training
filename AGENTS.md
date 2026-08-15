# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. See
`ARCHITECTURE.md` for the technical map of what each folder is and how they
relate; this file is conventions and how-to-run.

## What this repository is

A staged model-training workspace — `pre-training/` (data prep) →
`fine-tuning/` (LoRA adapters) → `serving/` (inference) → `training/`
(reserved, from-scratch training) — scaling from a single RTX 3090 (24GB) up
to multi-GPU. Each leaf folder is an independently deployable `uv` project;
there is no shared root Python environment. Full detail in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Environment & commands

**uv only**, one binary at the root drives every pipeline via
`uv run --directory <folder> <command>` — see the root
[`README.md`](README.md#uv-commands) for the current full list. Never `cd`
into a folder and run bare `python`; always go through `uv run` (or the
folder's `uv_setup.bat` / `uv_bootstrap.bat` for a one-time install) so the
correct pinned interpreter and CUDA torch build are used.

Each fine-tuning/serving folder pins its **own** torch/CUDA index in its
`pyproject.toml` — do not add a root-level `pyproject.toml` with shared
dependencies, and do not try to unify these into one `uv` workspace/lockfile.
The pins differ on purpose (different model classes, different CUDA
requirements) and a shared resolution would fight that.

Each project also pins its **own** `.python-version` (currently `3.12`
everywhere). Don't remove these or let a project fall back to "whatever
Python `uv` finds newest" — a too-new CPython (e.g. 3.14) can lack prebuilt
wheels for pinned deps like `pillow`, which makes `uv run` fail trying to
build from source instead of installing a wheel.

`fine-tuning/axolotl-ocr-summary/` currently only resolves on Linux/WSL:
`axolotl[deepspeed]` depends on `triton`, which ships no Windows wheels. This
is a real platform constraint of that dependency stack, not a bug to silently
patch around — flag it rather than reworking that project's pins unprompted.

## Conventions & guardrails

- **One root `.gitignore` covers the whole repo — don't add per-folder
  `.gitignore` files.** Its patterns are unanchored on purpose so they match
  at any depth (`DATASET/`, `data/`, `runs/`, `output/`, `outputs/`,
  `.cache/`, `hf_cache/`, `merged_model/`, `*.safetensors`, `*.pt`, `*.csv`
  etc. are ignored everywhere, not per-project). Only code, configs,
  `.python-version`, `uv.lock`, and README stubs in those drop-zone folders
  are versioned. Datasets are downloaded/pointed-to locally, never checked
  in. Likewise: one root `AGENTS.md`/`CLAUDE.md` pair for the whole repo,
  not one per pipeline folder.
- **`fine-tuning/llava15-lora/` builds its JSONL from CNN/DailyMail only**
  (`build_llava15_dataset.py --cnn-dailymail-dir`, required) — the earlier
  dual-source mode that also read pre-training's image-linked OCR/SUMMARIES
  CSV pair was removed entirely (not renamed) since it isn't needed for this
  pipeline's current use. Its CLI flags and JSONL field are named
  generically (`--source-csv` on the generator's batch-eval mode, `--text`,
  `--text-file`, JSONL field `text`), not `ocr_*` — don't reintroduce
  OCR-specific naming or resurrect the removed CSV-pair ingestion path
  without being asked. Its default instruction wrapper matches the
  CNN/DailyMail prompt, so `--instruction` doesn't need to be passed
  explicitly for the common case; it's still a CLI flag (on both the trainer
  and generator, must match between the two) for training on
  differently-worded source text.
- **`axolotl-ocr-summary/`'s data contract is unrelated and untouched** —
  it consumes any CSV via `--input --text-col --summary-col`.
- **Cross-folder references use full relative paths from repo root**, e.g.
  `serving/llava15-lora/` reaches its training counterpart via
  `../../fine-tuning/llava15-lora/`. When moving or renaming a pipeline
  folder, grep the whole repo for its old path (READMEs and code comments,
  not just imports — these pipelines don't import each other's code, but do
  reference each other's paths for the adapter/cache directories) before
  considering the move done.
- **`serving/<pipeline>/` never imports `fine-tuning/<pipeline>/` code** — it
  only reads that pipeline's trained output directory (adapter or merged
  model). Keep that boundary; it's what makes serving independently
  deployable.
- **Batch/PowerShell scripts** follow the existing style: bootstrap the env
  first (`uv_setup.bat`/`uv_bootstrap.bat`), resolve paths from the script's
  own location rather than assuming a cwd, `exit /b 1` on failure.
- **OCR-style input formatting is load-bearing where it appears** (e.g.
  `(newline)` markers, garbled spellings in `pre-training/`'s output) —
  that's the real training distribution downstream pipelines were tuned
  against. When wiring in a clean dataset like CNN/DailyMail instead, don't
  silently reformat it to look like OCR text; keep it as its own data source
  and be explicit about which fine-tuning run used which source.
- **Root `README.md` is user-owned** — only edit it when explicitly asked to.
  `ARCHITECTURE.md` and this file are the agent-maintained docs; keep them in
  sync with structural changes (new pipeline folder, moved folder, changed
  data contract) as part of the same change, not as a follow-up.
- **`pre-training/` is currently out of scope** for active work per repo
  owner — don't modify it unless asked.

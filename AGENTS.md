# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. See
`ARCHITECTURE.md` for the technical map of what each folder is and how they
relate; this file is conventions and how-to-run.

## What this repository is

A staged model-training workspace — `pre-training/` (data prep) →
`fine-tuning/` (LoRA adapters) → `serving/` (inference) → `training/`
(from-scratch training) — scaling from a single RTX 3090 (24GB) up
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

`fine-tuning/` currently holds `vicuna-7b-lora/` and `qwen25-3b-lora/`, both
same `transformers`+`peft` pattern. An earlier Axolotl-based
`axolotl-ocr-summary/` pipeline was removed by the repo owner — if a similar
pipeline reappears, note that `axolotl[deepspeed]` only resolves its `uv`
environment on Linux/WSL (`triton` ships no Windows wheels), a real platform
constraint, not something to patch around silently.

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
- **`fine-tuning/vicuna-7b-lora/` builds its JSONL from CNN/DailyMail only**
  (`build_vicuna7b_dataset.py --cnn-dailymail-dir`, required) — the earlier
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
- **Judge a trained `vicuna-7b-lora` adapter by `generate_vicuna7b_lora.py
  --jsonl-eval data/vicuna7b_train.jsonl --num-samples N`, not loss alone.**
  It replicates the trainer's held-out split and prints source/reference/
  generated triples with token-F1 — loss can plateau while the model is
  still producing good summaries (verified on a predecessor run: a flat
  ~1.0–1.2 loss plateau still produced coherent, on-topic, correctly-styled
  summaries).
- **`vicuna-7b-lora` loads `lmsys/vicuna-7b-v1.5` directly** via
  `AutoModelForCausalLM`/`AutoTokenizer` — not a LLaVA checkpoint, not
  `LlavaForConditionalGeneration`/`AutoProcessor`. Don't reintroduce a LLaVA
  dependency here; a planned `llava15-full-lora` sibling (image+text pairs,
  actually exercising the vision encoder/projector) is where that belongs —
  this repo's first real VLM fine-tune. `protobuf` is a required dependency
  in this pipeline specifically because `lmsys/vicuna-7b-v1.5` ships a raw
  SentencePiece tokenizer that needs it to convert to a fast tokenizer.
- **`fine-tuning/qwen25-3b-lora/` is `vicuna-7b-lora`'s sibling**, same
  pattern, `Qwen/Qwen2.5-3B-Instruct` instead. LoRA `target_modules` stay
  `["q_proj", "v_proj"]` (verified same as Vicuna via `peft`'s default LoRA
  target-module table for `qwen2`) but the prompt wrapper is ChatML
  (`<|im_start|>role\n...<|im_end|>`), not Vicuna's `USER:/ASSISTANT:` —
  verified against the model's actual `tokenizer_config.json`
  (`eos_token="<|im_end|>"`) before writing the code, not assumed. Before
  cloning this pattern to a new base model, check its actual attention
  module names first — `microsoft/Phi-3.5-mini-instruct`, for example, fuses
  Q/K/V into a single `qkv_proj` linear layer (confirmed by reading
  `Phi3Attention`'s source), so `target_modules=["q_proj","v_proj"]` would
  silently attach to nothing on that model.
- **`training/imdb-sentiment-cnn/`** trains a Text CNN (Kim 2014) **from
  scratch** on the Large Movie Review Dataset — binary sentiment
  classification, 25k train / 25k test, judged by accuracy on the held-out
  25k test split. Hand-written philosophy like the rest of `training/`: no
  torchtext/transformers/nltk, no `DataLoader` (numpy-permutation batching),
  and **no GloVe/pretrained embeddings by design** — strictly IMDB-only
  data, random-init trainable embeddings; don't silently add pretrained
  vectors. `build_imdb_dataset.py` verifies exactly 12,500 review files per
  split and refuses to build on a partial extraction (the dataset was once
  caught mid-extraction with `train/pos` still filling up). Data lives at
  `E:\datasets\aclImdb_v1` (a nested `aclImdb/` subfolder is also accepted).
  Verified real run: **89.2%** test acc, 20 epochs in ~31 s on the RTX 3090;
  dropout 0.5, best checkpoint by val acc (peaks ~epoch 3, then the model
  overfits fast — train acc → 100%); a dropout-0.7 variant scored worse and
  was discarded. The 50k unlabeled reviews are deliberately unused — a
  future AWD-LSTM / transformer+MLM pipeline is where they belong.
- **`training/flow-matching-mnist/`** trains a flow-matching /
  rectified-flow generative model **from scratch** on MNIST — the
  contemporary counterpart to `training/mnist-vae` (same dataset, same
  `data/mnist.npz` contract, same hand-written zlib PNG writer, so the two
  sample grids are directly comparable). Hand-written philosophy like the
  rest of `training/`: no `diffusers`/`torchcfm`/`torchdiffeq`/
  `torchvision`, no `DataLoader` (numpy-permutation batching) — the UNet
  velocity field, sinusoidal time embedding, EMA, and Euler/Heun ODE
  samplers are all written out. The objective is
  `mse(v(x_t, t), x1 - (1-sigma_min)*x0)` on the conditional-OT path
  `x_t = (1-(1-sigma_min)*t)*x0 + t*x1` (Lipman et al.
  [2210.02747](https://arxiv.org/abs/2210.02747)); at the default
  `--sigma-min 0.0` that is exactly rectified flow (Liu et al.
  [2209.03003](https://arxiv.org/abs/2209.03003)). **There is no noise
  schedule and no ELBO here on purpose** — don't add betas/`alpha_bar`, a
  variance head, or loss reweighting; that turns it back into a DDPM.
  1,175,841 params at `--base-channels 32`; data at
  `E:\datasets\mnist-dataset` (shared with `mnist-kmeans`/`mnist-vae`).
  **The evaluator's round-trip MAE/PSNR sweep measures ODE discretization
  error, not sample quality** — a near-zero velocity field round-trips
  almost perfectly since the identity is its own inverse, and a 2-epoch
  smoke run really did beat the converged model on it. Judge samples by
  `samples_grid.png` plus the nearest-neighbour memorization check. There
  is deliberately **no FID** (it needs a pretrained Inception, against this
  folder's from-scratch rule) — don't add a substitute score.
- **Cross-folder references use full relative paths from repo root**, e.g.
  `fine-tuning/vicuna-7b-lora/README.md` reaches a sibling pipeline via
  `../../<stage>/<pipeline>/`. When moving, renaming or removing a pipeline
  folder, grep the whole repo for its old path (READMEs and code comments,
  not just imports — these pipelines don't import each other's code, but do
  reference each other's paths for the adapter/cache directories) before
  considering the move done.
- **`serving/` currently holds no pipelines.** `serving/vicuna-7b-lora/` (a
  FastAPI service over the Vicuna adapter) was removed by the repo owner,
  the same way `fine-tuning/axolotl-ocr-summary/` was. If a serving folder
  returns, keep the boundary the old one had: **`serving/<pipeline>/` never
  imports `fine-tuning/<pipeline>/` code** — it only reads that pipeline's
  trained output directory (adapter or merged model), which is what makes
  serving independently deployable.
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

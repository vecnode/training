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
training/       raw datasets -> from-scratch / non-LoRA trained models
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

Two example pipelines, same `transformers`+`peft` pattern, both LoRA-based,
both sized for a single RTX 3090 (24GB). (A third, Axolotl-based
`axolotl-ocr-summary/` pipeline existed earlier but was removed by the repo
owner — it only resolved its `uv` environment on Linux/WSL, never natively
on Windows, since `axolotl[deepspeed]` depends on `triton`.)

| Pipeline | Framework | Base model | Data shape |
|---|---|---|---|
| [`fine-tuning/vicuna-7b-lora/`](fine-tuning/vicuna-7b-lora/README.md) | `transformers` + `peft` (manual `Trainer` loop) | `lmsys/vicuna-7b-v1.5` — LoRA on `q_proj`/`v_proj`, loaded directly via `AutoModelForCausalLM`/`AutoTokenizer`. No LLaVA checkpoint, no vision encoder, no multimodal projector anywhere in the dependency graph. | JSONL with `text` / `summary` fields |
| [`fine-tuning/qwen25-3b-lora/`](fine-tuning/qwen25-3b-lora/README.md) | `transformers` + `peft` (same pattern as `vicuna-7b-lora/`) | `Qwen/Qwen2.5-3B-Instruct` — LoRA on `q_proj`/`v_proj` (same target modules as Vicuna; `Qwen2ForCausalLM` uses the same separate Q/K/V/O naming, confirmed via `peft`'s own default LoRA target-module table). ChatML prompt format instead of Vicuna's `USER:/ASSISTANT:` (verified against the tokenizer's `chat_template`/`eos_token`). | JSONL with `text` / `summary` fields |

**Naming/loading history: `vicuna-7b-lora`, previously `llava15-lm-lora`,
originally `llava15-lora`.** Two renames, each fixing a real overstatement:

1. `llava15-lora` → `llava15-lm-lora`: the pipeline only ever LoRA'd the
   language-model backbone (`q_proj`/`v_proj`) of `llava-hf/llava-1.5-7b-hf`
   — never the vision encoder or multimodal projector, never an image. The
   `-lora` name alone overstated that as a VLM fine-tune.
2. `llava15-lm-lora` → `vicuna-7b-lora`: even loading LLaVA's checkpoint at
   all was unnecessary once the vision half was never used — it still
   downloaded the full ~14 GB multimodal weights via
   `LlavaForConditionalGeneration`/`AutoProcessor` to get to a submodule that
   is, in substance, Vicuna-7B. This pass switched to loading
   `lmsys/vicuna-7b-v1.5` **directly** via `AutoModelForCausalLM` +
   `AutoTokenizer` — ~13 GB instead of ~14 GB, no vision-related code path in
   the dependency graph at all, same LoRA config/target modules. Trade-off:
   `lmsys/vicuna-7b-v1.5` is the checkpoint LLaVA 1.5 was later
   visually-instruction-tuned *from*, not the LLaVA-tuned weights themselves
   — different starting point, not directly comparable to the old
   `llava15-lm-lora` run's results, but a cleaner, smaller, honestly-named
   base for a language-model-only LoRA. `serving/llava15-lora` was renamed
   to `serving/vicuna-7b-lora` in step, with matching model-loading changes
   (functionally required: a Vicuna-7B-trained adapter's parameter names
   don't match `LlavaForConditionalGeneration`'s `language_model.*` prefix,
   so serving would fail to load it otherwise) — that serving folder has
   since been removed, see Stage 3. A planned `llava15-full-lora`
   sibling, trained on image+text pairs and actually exercising the vision
   encoder/projector, remains the natural first real VLM fine-tune in this
   repo — that one *should* load the full LLaVA checkpoint. Full reasoning in
   [`fine-tuning/vicuna-7b-lora/README.md`](fine-tuning/vicuna-7b-lora/README.md#why-vicuna-7b-directly-not-via-llava).

**`vicuna-7b-lora/` is a generic text-summarization LoRA, not OCR-specific** —
its interface, data source, and default prompt were all cleaned up this pass
to reflect that:

- JSONL field is `text` (was `ocr_text`); `build_vicuna7b_dataset.py` and
  `generate_vicuna7b_lora.py`'s flags are `--source-csv`/`--text`/`--text-file`
  (were `--ocr-csv`/`--ocr-text`/`--ocr-text-file`).
- `build_vicuna7b_dataset.py` **only** builds from a CNN/DailyMail Parquet
  dump now (`--cnn-dailymail-dir`, required) — the earlier dual-source mode
  that also read pre-training's image-linked OCR/SUMMARIES CSV pair
  (`normalize_image_key`/`resolve_image_path`/`load_summaries`) was removed
  entirely, not just renamed, since it's not needed for this pipeline's
  current use (`generate_vicuna7b_lora.py`'s `--source-csv` batch-eval mode
  still accepts any generic CSV with a `text` column, unrelated to that
  removed ingestion path).
- `train_vicuna7b_lora.py`/`generate_vicuna7b_lora.py`'s `DEFAULT_INSTRUCTION`
  is now the CNN/DailyMail news-article wording (was "Summarize this scanned
  document page... UAP-related content") — since that's the only source this
  pipeline builds from, `--instruction` no longer needs to be passed
  explicitly for the common case.

**`qwen25-3b-lora/` is a near-clone of `vicuna-7b-lora/`** — same dataset
builder logic, same trainer/generator structure, same CLI shape. Two
verified differences (not assumed): the ChatML prompt wrapper (see table
above), and no `protobuf`/`sentencepiece` dependency needed (Qwen2.5-3B-Instruct
ships a ready `tokenizer.json`, unlike Vicuna's raw SentencePiece tokenizer).
No `serving/qwen25-3b-lora/` — `serving/` holds no pipelines at all now (see
Stage 3), so a serving folder would have to be written from scratch if this
adapter goes to production. It could not have been shared with the removed
`serving/vicuna-7b-lora/` in any case: that one was Vicuna-specific, and the
ChatML wrapper differs.

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
`summary`. `build_vicuna7b_dataset.py --cnn-dailymail-dir ... --max-samples
2000` was run end-to-end against the real files and produces valid JSONL
records; full details and commands in
[`fine-tuning/vicuna-7b-lora/README.md`](fine-tuning/vicuna-7b-lora/README.md).

### First real training run (superseded) and the reconstruction-test tool

Before the switch to loading Vicuna-7B directly, a 2,000-sample run on the
old `llava15-lm-lora` pipeline (1,800 train / 200 val, 1 epoch, 450 steps,
~31 min on a single RTX 3090) showed loss dropping 1.66 → ~1.12 in the first
~50 steps then plateauing in a ~1.0–1.2 band, with `eval_loss` (1.11)
tracking train loss closely (no overfitting) — a normal curve for a
rank-16, 2-projection adapter on a small dataset, not evidence of a broken
run. That adapter and its data/hf_cache were deleted as part of this pass's
switch to `lmsys/vicuna-7b-v1.5` (different base weights, not compatible
with the old adapter) — the numbers above are illustrative of the expected
curve shape, not a claim about the current pipeline's untrained state.

What carries forward: loss alone doesn't say whether summaries are actually
good, so `generate_vicuna7b_lora.py` has a `--jsonl-eval <path>
--num-samples N` reconstruction-test mode (added the same pass as the old
run above) — it replicates the trainer's train/val split (same seed/ratio)
and prints genuinely held-out source/reference/generated triples with
token-F1, instead of requiring a manual `--text` string. Run against the old
adapter it produced coherent, on-topic, correctly-bulleted CNN/DailyMail
summaries (avg token-F1 0.357 across 5 samples) despite the plateaued loss —
this is the tool to use to judge the next real run on the current pipeline,
not the loss curve. See the
[pipeline README](fine-tuning/vicuna-7b-lora/README.md#5-reconstruction-test--verify-quality-not-just-loss).

## Stage 3 — `serving/`

One `serving/<pipeline>/` folder per fine-tuning pipeline that has a serving
story. **Currently empty.** `serving/vicuna-7b-lora/` — a FastAPI service
(`app.py`) that loaded the base Vicuna-7B model plus the trained adapter (or
a fused/merged model) once and served a JSON API plus a dataset-browser
front-end — was removed by the repo owner, the same way
`fine-tuning/axolotl-ocr-summary/` was.

The design it demonstrated is still the intended shape for this stage, and
worth restating for whatever returns here: a serving folder is deliberately
decoupled from its fine-tuning counterpart, reading only that pipeline's
trained output directory (for Vicuna that was
`../../fine-tuning/vicuna-7b-lora/runs/vicuna7b_lora/final_adapter`) and
never importing its training code. That boundary is what makes a serving
folder independently deployable.

## Stage 4 — `training/`

From-scratch / non-LoRA training of other models, as distinct from
adapting an existing checkpoint (`fine-tuning/`). Six pipelines so far,
each an independent `uv` project and each writing out by hand whatever the
usual library would hide.

[`training/adult-income-logreg/`](training/adult-income-logreg/README.md) is
the first pipeline here: logistic regression on the UCI
[Adult / Census Income](https://archive.ics.uci.edu/dataset/2/adult)
dataset, implemented with raw numpy rather than scikit-learn — the sigmoid,
binary cross-entropy loss, gradient derivation, and gradient-descent update
loop are all written out by hand in `train_logreg.py` so the math stays
visible, and `build_income_dataset.py` parses the raw CSV files and does
one-hot/z-score encoding without pandas. Own `uv` project like every other
pipeline folder, but no torch/CUDA dependency at all — just `numpy`.
Verified end-to-end against the real dataset: 30,162 train / 15,060 test
rows after dropping `"?"` rows (matches the cleaned-variant counts in
`adult.names` exactly), 300 epochs of batch gradient descent, 84.6% test
accuracy (in line with the 84–86% published for tree-based methods on this
same cleaned split — a from-scratch linear model landing close to that is
the expected sanity-check result, not a target to beat).

[`training/cifar10-vqvae/`](training/cifar10-vqvae/README.md) is the next
pipeline here: a VQ-VAE
([van den Oord et al., 2017](https://arxiv.org/abs/1711.00937)) trained
from scratch on CIFAR-10 (raw python-format pickles at
`E:\datasets\cifar-10-python`, parsed by hand with stdlib `pickle` — no
torchvision/keras, no Pillow). It is the **in-place successor of the
former `cifar10-vae`** (renamed/converted): a plain VAE's blur comes from
Gaussian-posterior averaging in the ELBO, and VQ-VAE removes that
mechanism — an 8x8 grid of D-dim encoder vectors is replaced by its
nearest neighbors in a learned 512x64 codebook (straight-through
estimator, EMA codebook updates, commitment loss), and reconstruction is
driven purely by MSE. Same hand-written philosophy as the whole `training/`
folder: encoder/quantizer/decoder all plain `torch.nn`, torch only for
tensor ops/autograd/GPU, numpy-permutation batching, no VQ-VAE library,
no `DataLoader`. ~0.74M params (codebook included), ~40–60 min for 100
epochs on a single RTX 3090. Reconstruction-only by design (encode ->
quantize -> decode a real image back; no learned prior over the discrete
codes, so no sampling) — the discrete code grid is the substrate for a
later learned prior (the planned cascade). Its evaluator computes the same
metric suite used to judge the predecessor VAE (MAE/PSNR/SSIM/
high-frequency retention) plus a codebook-usage check, so the VQ-VAE vs
VAE comparison is reproducible; measured numbers in the pipeline README's
"Verified runs".

[`training/imdb-sentiment-cnn/`](training/imdb-sentiment-cnn/README.md)
is the text-classification pipeline here: a Text CNN
([Kim, 2014](https://arxiv.org/abs/1408.5882)) trained **from scratch** on
the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
(25k train / 25k test binary sentiment; raw review `.txt` files at
`E:\datasets\aclImdb_v1`, parsed by hand — no torchtext/datasets/nltk).
Same hand-written philosophy as the whole `training/` folder: a
randomly-initialized **trainable** embedding (**no GloVe** — strictly
IMDB-only data by design), three parallel 1D convs (widths 3/4/5 × 128
filters) + ReLU + 1-max-pool per filter, concat, dropout 0.5, linear → 2;
torch only for tensor ops/autograd/GPU, numpy-permutation batching, no
`DataLoader`. ~6.6M params (6.4M in the embedding), **20 epochs in ~31 s**
on a single RTX 3090, best checkpoint by val acc (peaks at epoch 3 before
the model overfits — train acc → 100%). Measured on the held-out 25k test
split: **89.2% accuracy** (neg 88.96% / pos 89.43%) — well above Kim's
published CNN-rand (82.7%), which the README attributes to full-length
reviews plus val-based early stopping. A dropout-0.7 variant scored 87.9%
on test and was discarded. Full numbers in the pipeline README's
"Verified runs".

[`training/flow-matching-mnist/`](training/flow-matching-mnist/README.md)
is the newest pipeline here, and the first *generative* model family in
this repo that is not an autoencoder: flow matching / rectified flow
trained **from scratch** on MNIST. The model learns a velocity field
`v(x,t)` whose ODE transports `N(0,I)` at `t=0` into the data at `t=1`;
training regresses that velocity on straight-line conditional paths with
plain MSE — `x_t = (1-(1-sigma_min)*t)*x0 + t*x1`, target `x1 -
(1-sigma_min)*x0` — which is Lipman et al.'s conditional-OT path
([2210.02747](https://arxiv.org/abs/2210.02747)) and, at the default
`--sigma-min 0.0`, exactly the rectified flow of Liu et al.
([2209.03003](https://arxiv.org/abs/2209.03003)). **There is no noise
schedule, no variance parameterization, and no ELBO** — that absence is the
point, and it is the difference between this and a DDPM. Same hand-written
philosophy as the rest of `training/`: the UNet velocity field (three
resolutions, one 7x7 self-attention block, sinusoidal time embedding as a
per-channel bias), the EMA, and the Euler/Heun ODE samplers are all plain
`torch.nn`; no `diffusers`/`torchcfm`/`torchdiffeq`/`torchvision`, no
`DataLoader`, numpy-permutation batching. ~1.18M params, **40 epochs in
~5.3 min** on a single RTX 3090.

It is the deliberate counterpart to [`training/mnist-vae`](training/mnist-vae/README.md):
same dataset, same `data/mnist.npz` contract, same hand-written zlib PNG
writer, so the two prior-sample grids are directly comparable — the flow
model's are visibly sharper, the VAE's blur being the Gaussian-posterior
averaging that `cifar10-vqvae` also exists to remove. That comparison is
model-family vs model-family, **not** a controlled ablation: 1,175,841
params of UNet against 370,945 of plain conv encoder/decoder, and the two
runs cannot separate objective from capacity. The honest cost difference
they do show: the VAE generates in one forward pass, the flow model needs
20–50 network evaluations.

Judging it needed care, and two of this pipeline's guardrails came from
getting it wrong first. (1) The evaluator's round-trip MAE/PSNR sweep
measures **ODE discretization error, not sample quality** — the ODE is
time-reversible, so a real digit can be integrated back to noise and
forward again, but a near-zero velocity field round-trips perfectly since
the identity is its own inverse; a 2-epoch smoke run really did post a
better round-trip PSNR than the converged model. The sweep's actual use is
choosing `--num-steps`, and it shows Heun winning per *network evaluation*,
not just per step (10 Heun steps beat 50 Euler steps by ~2 dB at the same
100 evals). (2) The nearest-neighbour memorization check reports a distance
that is meaningless without a scale, so real held-out test digits are
measured against the training set the same way as a control. There is
deliberately **no FID** — it would require a pretrained Inception network,
against this folder's from-scratch rule, and a substitute number would be
worse than none. Measured figures in the pipeline README's "Verified runs".


## Datasets

None of the fine-tuning pipelines ship data — `DATASET/`, `data/`, `runs/`,
`output/` etc. are all git-ignored, drop-zone folders (via the single root
`.gitignore`). Both `vicuna-7b-lora/` and `qwen25-3b-lora/` train on
CNN/DailyMail. Example small, permissively-licensed public datasets are
listed in the root `README.md`'s **Datasets** section.

### `vicuna-7b-lora`'s real 2-epoch run (repo owner's machine)

2,000-sample JSONL (1,800 train / 200 val), 2 epochs, 900 steps, ~61 min on
a single RTX 3090. Train loss 1.76 → 0.99, eval_loss essentially flat across
epochs (1.100 → 1.094 — a mild overfitting signal in isolation, train loss
kept falling while eval_loss didn't). What matters: reconstruction-test avg
token-F1 rose from 0.357 (an earlier 1-epoch/1,800-sample run on the
predecessor `llava15-lm-lora` pipeline) to **0.467** on this run, and the
generated summaries reproduced exact figures from source text correctly
(e.g. "383-41" and "70-26" vote counts). Confirms the earlier lesson again:
eval_loss plateauing is not itself a stop signal — the reconstruction test
is what actually shows whether a further epoch helped.

## Verified working (this pass)

`uv run --directory <folder> python <script> --help`, and further real
executions where noted, actually run, not assumed:

- `training/cifar10-vqvae` (successor of the former `cifar10-vae`) — real
  runs on the RTX 3090 against the actual downloaded CIFAR-10 python
  pickles (`E:\datasets\cifar-10-python`): `build_cifar10_dataset.py`
  wrote `data/cifar10.npz` (50k train / 10k test); `train_vqvae.py` and
  `evaluate_vqvae.py` run end-to-end (100 epochs, ~10–15 min, codebook
  perplexity ~404/512, 512/512 codes fired on test — no collapse).
  Measured reconstruction on the held-out test set: PSNR **25.2 dB**, SSIM
  **0.884**, MAE 0.042, high-frequency detail kept **73.8%** — vs the
  predecessor VAE's best (PSNR 21.5 dB, SSIM 0.742, HF 51.4%), i.e. the
  discrete-codebook family removes the plain-VAE blur mechanism. Full
  numbers in the pipeline README's "Verified runs".
- `training/flow-matching-mnist` — real run on the RTX 3090 against the
  actual MNIST IDX files at `E:\datasets\mnist-dataset`:
  `build_mnist_dataset.py` wrote `data/mnist.npz` (60k train / 10k test);
  `train_flow.py` trained 1,175,841 params for 40 epochs in **317.5 s**
  (val velocity MSE 0.2263 → **0.1704**, best at epoch 38 — train and val
  track each other throughout, no overfitting); `evaluate_flow.py` scored
  **0.1687** velocity MSE on the held-out 10k and wrote all three PNGs.
  Reproduced in a second independent run of the same commands (312.1 s,
  best val 0.1705, test 0.1687) — expect the third decimal to move.
  Samples are clean, readable digits with a handful of malformed glyphs per
  64. The ~0.17 loss floor is expected, not a defect: `u = x1 - x0` is
  irreducibly random given `(x_t, t)`, so the MSE floors at that
  conditional variance and can never reach zero — judge the samples, the
  same lesson `fine-tuning/vicuna-7b-lora` taught about loss plateaus.
  Memorization check: generated samples sit *farther* from the training set
  (mean L2 4.029, min 1.858) than real unseen test digits do (3.611 /
  1.169).
- `fine-tuning/qwen25-3b-lora` — `build_qwen3b_dataset.py`,
  `train_qwen3b_lora.py`, `generate_qwen3b_lora.py`, plus a real 40-sample
  smoke train against the actual downloaded `Qwen/Qwen2.5-3B-Instruct`
  weights, confirming `trainable params > 0` (LoRA genuinely attached to
  `q_proj`/`v_proj`) rather than trusting `peft`'s target-module table alone.
- `fine-tuning/vicuna-7b-lora` — real 2,000-sample/2-epoch training run (see
  above), executed by the repo owner, not just a smoke test.
Not executed this pass (no PDFs/poppler set up in this environment):

- `pre-training/exec_1.bat` / `scripts/convert_pdf_to_png.ps1`

## Next steps

- `fine-tuning/vicuna-7b-lora` has a verified-good real run (see above) —
  reasonable next moves are more samples (the eval_loss plateau suggests
  more epochs on this same 1,800-row set has limited further upside),
  judged by the reconstruction test, not loss alone.
- `fine-tuning/qwen25-3b-lora` is smoke-tested but not yet trained for real
  — same next step as Vicuna's first run: build a few-thousand-sample JSONL,
  train, then judge with `--jsonl-eval`.
- `training/` has a verified real `cifar10-vqvae` run (see above) — the
  planned next rung is stage 2 of its cascade: a learned prior over the
  discrete code grid (PixelCNN/transformer over code indices, or a latent
  DDPM), latent-diffusion style.
- `training/imdb-sentiment-cnn` is verified at **89.2%** test acc with
  random embeddings in ~31 s — natural next rungs: a GloVe variant
  (~+1–3 pts expected), or the bigger from-scratch projects that use the
  50k unlabeled reviews (AWD-LSTM LM-pretrain + fine-tune, or a small
  transformer with MLM pretraining, both ~91% territory).
- `training/flow-matching-mnist` is verified at ~5.3 min for 40 epochs and
  was still improving when it stopped — natural next rungs: more epochs or
  a wider `--base-channels`; class-conditioning plus classifier-free
  guidance (the smallest real upgrade, and what makes samples steerable); a
  second rectification pass (re-train on the model's own noise/sample
  pairs) to straighten the paths for 1–4-step sampling, which is the whole
  reason rectified flow is used in production; or the same objective on
  CIFAR-10 next to `cifar10-vqvae`, where the VAE-blur comparison has more
  room to show itself than at 28x28.
- `fine-tuning/llava15-full-lora` (planned, not started): the first real VLM
  fine-tune in this repo — image+text pairs, vision encoder/projector
  actually in the training graph, unlike `vicuna-7b-lora`/`qwen25-3b-lora`.
- A `phi35-mini-lora` sibling (discussed, not started) would need
  `target_modules=["qkv_proj"]` instead of `["q_proj", "v_proj"]` — Phi-3
  fuses Q/K/V into one linear layer (confirmed by reading
  `Phi3Attention`'s source), so the `vicuna-7b-lora`/`qwen25-3b-lora`
  target-module config would silently attach to nothing on that model.

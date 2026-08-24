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
  caught mid-extraction with `train/pos` still filling up); a nested
  `aclImdb/` subfolder is also accepted.
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
  1,175,841 params at `--base-channels 32`; same MNIST drop as
  `mnist-kmeans`/`mnist-vae`.
  **The evaluator's round-trip MAE/PSNR sweep measures ODE discretization
  error, not sample quality** — a near-zero velocity field round-trips
  almost perfectly since the identity is its own inverse, and a 2-epoch
  smoke run really did beat the converged model on it. Judge samples by
  `samples_grid.png` plus the nearest-neighbour memorization check. There
  is deliberately **no FID** (it needs a pretrained Inception, against this
  folder's from-scratch rule) — don't add a substitute score.
- **`training/rvq-audio-codec/`** trains a neural audio codec with
  residual vector quantization **from scratch** on LJSpeech — the
  EnCodec/SoundStream/DAC architecture, and the **first audio pipeline in
  this repo**. Hand-written philosophy like the rest of `training/`: the
  RIFF/WAVE parser *and* writer, the SEANet conv encoder/decoder, the RVQ
  (factorized 8-dim lookup, cosine distance, EMA updates, dead-code
  revival), the mel filterbank, the multi-scale STFT discriminator and
  SI-SDR are all written out — no `encodec`/`descript-audio-codec`/
  `audiocraft`, no `torchaudio`/`librosa`/`soundfile`/`scipy`, no
  `DataLoader` (numpy-permutation batching over utterance indices, one
  random crop each). 7,338,658 params plus a 2,112,582-param discriminator
  used only in training. `--data-dir` points at an extracted LJSpeech-1.1
  (a nested `LJSpeech-1.1/` subfolder is also accepted);
  `build_ljspeech_dataset.py`
  verifies exactly 13,100 wavs and refuses a partial extraction, and writes
  a memmapped `data/ljspeech_audio.i16` + `data/ljspeech_index.npz` rather
  than an `.npz` of samples — 3.8 GB of int16 becomes 7.6 GB as float32 and
  won't sit in RAM. Things not to silently undo:
  - **Trained at LJSpeech's native 22,050 Hz, no resampler.** Frame rate is
    68.9 Hz and the bitrate 5.51 kbps — don't "fix" these to EnCodec's
    published 24 kHz / 75 Hz / 6 kbps figures, and don't add a resampler
    without being asked.
  - **Quantizer dropout is load-bearing**, not a regularizer: it is what
    lets one trained model serve the whole 1→8 codebook ladder. Removing it
    means eight separate runs for the same demo.
  - **The discriminator is staged behind `--adv-start-step`** on purpose;
    `--lambda-adv 0` is the reconstruction-only A/B. Don't make it
    unconditional.
  - **The dead-code cutoff is a fraction of uniform codebook usage, not an
    absolute count.** The absolute 2.0 that EnCodec and
    `vector-quantize-pytorch` use is a trap at this batch size (32 × 69 =
    2,208 vectors over 1,024 entries → uniform usage is 2.16), and the
    first smoke run really did report 1,023 of 1,024 entries revived per
    codebook. After the fix, 0.
  - **The discriminator is ~8x the cost of the whole rest of the step**
    (7.75 steps/s reconstruction-only vs 0.95 fp32 with it, RTX 3090, batch
    32). `--disc-bf16 1` (default) runs **only the critic** under bf16
    autocast → 2.01 steps/s, 10.8 GiB instead of 16.8, no architecture
    change. Don't extend that autocast over the generator: the codebook
    lookup and EMA updates must stay fp32, since bf16 EMA statistics stop
    accumulating small updates — the exact thing dead-code revival exists
    to detect. `cudnn.benchmark` and TF32 were measured and do nothing here.
  - **The EMA weights are only better once converged.** After n steps they
    still carry `ema_decay**n` of the random init — the 802-step smoke
    checkpoint at decay 0.999 is 45% initialization and scored 9.02 mel
    against the live weights' 7.42. A full run leaves that behind
    (`0.999**24060 = 4e-11`), so `--use-ema 1` stays the default, but
    `evaluate_codec.py` computes the share from the checkpoint and warns
    above 1%. Don't remove that warning or the `global_step`/`ema_decay`
    fields it reads.
  - **SI-SDR is a weak proxy for a GAN-trained codec** — the adversarial
    loss trades waveform/phase alignment for perceptual realism, so a
    better-sounding model can score worse. Judge by the emitted
    `original_NN.wav` / `recon_nq{8,4,2,1}_NN.wav` pairs and the
    per-codebook usage table. There is deliberately **no ViSQOL/PESQ/
    NISQA** (external binary or pretrained network), the same rule that
    keeps FID out of `flow-matching-mnist` — don't add a substitute score.
  It is the deliberate successor of `training/cifar10-vqvae` (one codebook
  → eight; full-latent L2 lookup → 8-dim factorized cosine lookup), and its
  collapse-mitigation findings are meant to transfer back there — the
  60-epoch run ended with **all 1,024 entries of all eight codebooks in
  use**, the deepest codebook carrying the highest perplexity of the stack
  (903.6 vs the first's 792.0). **Don't re-run training to "improve" the
  numbers without being asked**: the run cost 2 h 54 min and the repo owner
  has said it is finished. Its measured results are pinned in the pipeline
  README's "Verified runs" and in `ARCHITECTURE.md`; treat them as the
  record.

- **`training/fashion-mnist-dcgan/`** trains a **DCGAN**
  ([Radford et al. 2015](https://arxiv.org/abs/1511.06434)) **from scratch**
  on Fashion-MNIST - the repo's **first GAN pipeline**. Hand-written
  philosophy like the rest of `training/`: the generator and discriminator
  conv nets, the `N(0, 0.02)` weight init, one-sided label smoothing and the
  balanced D/G update loop are all written out - no `torchvision`, no
  `kagglehub`/`pytorch-gan-metrics`, no `DataLoader` (numpy-permutation
  batching). `--data-dir` points at a folder of IDX ubyte files (the Kaggle
  CSVs are also accepted); `build_fashion_mnist_dataset.py` verifies exactly
  60,000/10,000 and refuses a partial extraction, writing
  `data/fashion_mnist.npz` (float32 `[0,1]` plus class names; the trainer
  rescales to `[-1,1]` for Tanh output). Things not to silently undo:
  - **28x28 does not divide cleanly down DCGAN's canonical 32x32 ladder** -
    three stride-2 convs take 28 -> 14 -> 7 -> 3, so the discriminator's
    last feature map is 3x3 (final 3x3 conv to a logit) and the generator
    starts from a 7x7 grid, not 4x4. The shapes in `train_dcgan.py` are the
    verified ones; "fixing" them to the paper's 32x32 figures breaks the
    tensors.
  - **A GAN is judged by its samples, not its loss.** D/G losses move
    adversarially and say little about quality, so `train_dcgan.py` saves a
    fixed-z sample grid every `--sample-every` epochs (collapse becomes
    visible across training) and `evaluate_dcgan.py` emits `samples_grid.png`
    (hand-written zlib PNG writer, same as `mnist-vae`/`flow-matching`), the
    nearest-neighbour memorization check (L2 to the closest training image
    vs a real-image control) and a pairwise-diversity probe. There is
    deliberately **no FID/IS** - both need a pretrained Inception, the same
    rule that keeps FID out of `flow-matching-mnist` and ViSQOL out of
    `rvq-audio-codec`.
  New pipeline - no verified-run numbers yet; pin them in the pipeline
  README's "Verified runs" once it has been run on the RTX 3090.

- **`training/vit-cifar10/`** trains a **Vision Transformer**
  ([Dosovitskiy et al. 2021](https://arxiv.org/abs/2010.11929), pre-LN /
  norm-first layout as popularized by DeiT) **from scratch** on CIFAR-10 -
  the repo's **first attention-based vision model and its first from-scratch
  transformer of any kind**. Hand-written philosophy like the rest of
  `training/`: the patch embedding, learned CLS token + positional
  embeddings, the pre-LN transformer blocks, and the multi-head
  self-attention (QKV projections, scaled dot-product, output projection)
  are all plain `torch.nn` - no `transformers`/`timm`/`torchvision`, no
  `DataLoader` (numpy-permutation batching). `build_cifar10_dataset.py` is
  the same stdlib-pickle parser / same `data/cifar10.npz` contract as
  `training/cifar10-vqvae`. Things not to silently undo:
  - **Flip+crop augmentation is plain torch ops, applied per batch in the
    training loop** (`torch.flip`, 4px zero-pad + random crop, per-channel
    normalize with the hardcoded CIFAR-10 train statistics). `--no-augment`
    is the documented A/B (measured 66.8% top-1 with aug at 60 epochs; the
    no-aug leg is still unmeasured), not a debug flag to remove.
  - **The LR schedule is linear-warmup-then-cosine, not plain cosine.**
    ViTs train unstably from scratch without the warmup; the rest of
    `training/`'s plain `CosineAnnealingLR` is deliberately not reused
    here. Don't "simplify" it back.
  - **AdamW with weight decay 0.05, not plain Adam** - the ViT default,
    unlike the other torch trainers in this folder. Same reason.
  Defaults are ~10.7M params (`--dim 384 --depth 6 --heads 6 --mlp-ratio
  4`), ~23 min for 60 epochs fp32 on the RTX 3090; best checkpoint by val
  acc on a 10% holdout, the 10k test split stays unseen until
  `evaluate_vit.py`. The evaluator reports test top-1/top-5, per-class
  accuracy + confusion matrix, and writes `predictions_grid.png` (a
  hand-written zlib RGB PNG - green border = correct, red = wrong - no
  imaging library). There is deliberately **no pretrained-feature score**,
  the same rule that keeps FID out of `flow-matching-mnist` and ViSQOL out
  of `rvq-audio-codec`. The patch-embed/block stack is the planned encoder
  for a future I-JEPA-style self-supervised pipeline. Verified real run on
  the RTX 3090 (repo owner): **66.82% test top-1 / 97.32% top-5** at the
  60-epoch defaults in ~23 min (10,695,562 params, best val 67.50% at
  epoch 60, frog 81.4% / cat 44.7% per-class). That is below the ~80-86%
  figure this section's earlier draft expected - too optimistic for
  flip+crop-only at 60 epochs; the measured 66.8% is the record, and the
  correction is documented in the pipeline README.

- **`training/mae-cifar100/`** trains a **Masked Autoencoder**
  ([He et al. 2022](https://arxiv.org/abs/2111.06377)) **from scratch** on
  CIFAR-100 - the repo's **first representation-learning (self-supervised)
  pipeline**, and the mask-reconstruct sibling of the I-JEPA-style rung
  planned in `ARCHITECTURE.md`. Hand-written philosophy like the rest of
  `training/`: the patch embedding, positional embeddings, pre-LN
  transformer blocks, hand-written multi-head self-attention, the random
  masking, and the lightweight decoder are all plain `torch.nn` - no
  `transformers`/`timm`/`torchvision`, no `DataLoader` (numpy-permutation
  batching). The encoder is **vit-cifar10's patch-embed/block stack copied
  in by hand** (pipelines never import each other's code), defaulting to
  patch 2 -> 256 patches (64 visible at 75% masking; `--patch-size 4`
  gives the sibling's literal 64-patch config). `build_cifar100_dataset.py`
  is the same stdlib-pickle style as `vit-cifar10`'s builder, but CIFAR-100
  ships one 50k `train` file + one 10k `test` file (plus `meta`), verified
  by exact count. Things not to silently undo:
  - **Masking is per-sample fixed-count (a random permutation keeping the
    first 75%-complement), not a per-patch Bernoulli** - the paper's
    scheme, so every image is masked at exactly `--mask-ratio`.
  - **The loss is MSE on the masked patches only, with per-patch-normalized
    targets** (subtract patch mean / divide patch std - the MAE trick that
    stops the decoder collapsing to patch means). `--no-patch-norm` is the
    documented A/B, not a flag to remove.
  - **Pretraining does not normalize pixels** (flip+crop only, raw [0,1]):
    the reconstruction targets ARE the pixels. Normalization belongs to the
    linear probe, with the hardcoded CIFAR-100 train statistics.
  - **The decoder is pretraining-only** (~1M of the ~11.8M params);
    `linear_probe.py` discards it and reads frozen encoder features
    (mean-pooled patch tokens - MAE has no CLS token).
  - **The judge is a hand-written linear probe** (a linear head trained
    from scratch on the model's own frozen features, SGD momentum + cosine
    per the paper) - not FID/Inception, the same no-pretrained-features
    rule as everywhere else in `training/`. Probe features are precomputed
    offline (normalize-only inputs, no probe-time flip+crop - documented).
  - **Pos-embed / mask-token are trunc-normal 0.02 initialized** (the
    paper's init) - unlike vit-cifar10, whose pos embed is left at zero.
  Defaults: 11,766,540 params (10,750,848 encoder / 1,015,692 decoder),
  ~35 s/epoch on the RTX 3090 (smoke-measured) -> ~36 min for 60
  epochs; best checkpoint by a deterministic full-image val reconstruction
  MSE, the 10k test split stays unseen until `linear_probe.py`, which
  reports test top-1/top-5, coarse (20 superclass) top-1, per-class +
  100x100 confusion matrix, `test_metrics.txt` and a hand-written zlib
  `probe_grid.png`. Verified real run on the RTX 3090 (repo owner):
  **25.56% test top-1 / 53.41% top-5** (coarse 37.89%) at the 60-epoch
  defaults in ~36 min (2,166 s) - recorded on the **final epoch-60
  checkpoint**, which probes better than the best-val epoch-34 one
  (24.65% / 53.15% / 36.80%): the val recon curve bottomed at epoch 34
  while the masked-MSE kept improving, so the late features are the
  better representation (a finding - don't assume the best-val checkpoint
  holds the best features; the comparison is deterministic and was
  reproduced). oak_tree 69.0% / bowl 1.0% per-class. That is below the
  ~30-45% figure this section's earlier draft expected - too optimistic
  for 60 epochs / 45k images without probe-time augmentation; the measured
  25.56% is the record, and the correction is documented in the pipeline
  README.

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

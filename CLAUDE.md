# CLAUDE.md

Guidance for Claude Code in this repository. The full agent/contributor guide
is in `AGENTS.md` — read it first. One `AGENTS.md`/`CLAUDE.md` pair at the
root covers the whole repo; subfolders don't get their own.

@AGENTS.md

## Claude-specific quick reference

- Staged workspace: `pre-training/` (PDF corpus → OCR/summary/layout/QA CSVs,
  currently out of scope for active work) → `fine-tuning/` (LoRA adapters,
  current focus) → `serving/` (inference) → `training/` (from-scratch
  training). Full map in `ARCHITECTURE.md`.
- **Env:** `uv` only, one binary at the root drives every pipeline via
  `uv run --directory <folder> <command>` — see `README.md`'s uv Commands
  section for the exact list. Each leaf folder pins its own `.python-version`
  (currently `3.12` everywhere — newer CPython builds don't have prebuilt
  wheels yet for some pinned deps like `pillow`, which breaks `uv run` with a
  source-build failure) and its own CUDA torch build; never introduce a
  shared root Python environment.
- **`fine-tuning/vicuna-7b-lora/`** trains a LoRA adapter on plain
  `lmsys/vicuna-7b-v1.5`, loaded directly via `AutoModelForCausalLM`/
  `AutoTokenizer` — not a LLaVA checkpoint, not `LlavaForConditionalGeneration`/
  `AutoProcessor`. Previously named `llava15-lm-lora` (loaded the full ~14 GB
  `llava-hf/llava-1.5-7b-hf` checkpoint and only LoRA'd its language
  submodule) and before that `llava15-lora`; both names overstated what was
  happening. `protobuf` is a required dependency specifically because
  `lmsys/vicuna-7b-v1.5` ships a raw SentencePiece tokenizer that needs it to
  convert to a fast tokenizer — omitting it breaks `AutoTokenizer.from_pretrained`.
  It's a generic text-summarization fine-tune, not OCR-specific. It builds
  its JSONL from a local CNN/DailyMail Parquet dump only
  (`build_vicuna7b_dataset.py --cnn-dailymail-dir`, required) — an earlier
  mode that also read pre-training's image-linked OCR/SUMMARIES CSV pair was
  removed entirely, not just renamed. CLI flags and the JSONL field are named
  generically (`--source-csv`, `--text`, `--text-file`, JSONL field `text`)
  rather than `ocr_*`, on purpose — don't reintroduce `ocr_*` naming or bring
  back the removed CSV-pair ingestion path without being asked. The default
  `--instruction` matches the CNN/DailyMail prompt, so it doesn't need to
  be passed explicitly for the common case; it's still overridable (both
  trainer and generator, must match between the two) for other wording. A
  planned `llava15-full-lora` sibling (image+text pairs) is where a real
  LLaVA/vision dependency belongs in this repo — not here.
- **`fine-tuning/qwen25-3b-lora/`** is `vicuna-7b-lora`'s sibling, same
  `transformers`+`peft` pattern, `Qwen/Qwen2.5-3B-Instruct` instead. LoRA
  `target_modules` stay `["q_proj", "v_proj"]` (verified same naming as
  Vicuna via `peft`'s default LoRA target-module table for `qwen2`), but the
  prompt wrapper is ChatML (`<|im_start|>role\n...<|im_end|>`), not Vicuna's
  `USER:/ASSISTANT:` — verified against the model's real
  `tokenizer_config.json` before writing the code. No `protobuf`/
  `sentencepiece` needed here (Qwen ships a ready `tokenizer.json`). Before
  cloning this pattern to another base model, verify its actual attention
  module names first, don't assume `q_proj`/`v_proj` — e.g.
  `microsoft/Phi-3.5-mini-instruct` fuses Q/K/V into one `qkv_proj` layer
  (confirmed by reading `Phi3Attention`'s source) and would need
  `target_modules=["qkv_proj"]` instead, or LoRA silently attaches to
  nothing.
- **`training/imdb-sentiment-cnn/`** — Text CNN (Kim 2014) trained **from
  scratch** on the Large Movie Review Dataset (binary sentiment, 25k
  train / 25k test): hand-written torch (no torchtext/transformers, no
  `DataLoader`, numpy-permutation batching) and **no GloVe/pretrained
  embeddings by design** — strictly IMDB-only, random-init trainable
  embeddings. `build_imdb_dataset.py` verifies exactly 12,500 files per
  split (refuses a partial extraction; the dataset was once caught
  mid-extraction) and accepts a nested `aclImdb/` subfolder; data at
  a nested `aclImdb/` subfolder. Verified real run: **89.2%** test acc, 20
  epochs in ~31 s on the RTX 3090; dropout 0.5, best checkpoint by val acc
  (peaks ~epoch 3, then fast overfitting — train acc → 100%). Details in
  `ARCHITECTURE.md` Stage 4.
- **`training/flow-matching-mnist/`** — flow matching / rectified flow
  trained **from scratch** on MNIST, the contemporary counterpart to
  `training/mnist-vae` (same `data/mnist.npz` contract, same hand-written
  PNG writer, comparable sample grids). Hand-written torch: no
  `diffusers`/`torchcfm`/`torchdiffeq`/`torchvision`, no `DataLoader`; the
  UNet velocity field, sinusoidal time embedding, EMA, and Euler/Heun ODE
  samplers are all written out. Objective is plain MSE against the
  conditional-OT path's velocity; `--sigma-min 0.0` (default) makes it
  exactly rectified flow. **No noise schedule, no ELBO, by design** — don't
  add betas/`alpha_bar`, a variance head, or loss reweighting. 1,175,841
  params at `--base-channels 32`; same MNIST drop as `mnist-kmeans`/`mnist-vae`.
  Careful with the evaluator's round-trip sweep: it measures **ODE
  discretization error, not quality** (an untrained model scores
  near-perfectly on it). No FID by design — it would need a pretrained
  Inception. Details in `ARCHITECTURE.md` Stage 4.
- **`training/rvq-audio-codec/`** — neural audio codec with residual
  vector quantization (EnCodec/SoundStream/DAC architecture) trained
  **from scratch** on LJSpeech; the repo's **first audio pipeline** and the
  successor of `training/cifar10-vqvae` (one codebook → eight, each
  quantizing the previous residual). Hand-written torch: RIFF/WAVE parser
  and writer, SEANet conv encoder/decoder, RVQ, mel filterbank,
  multi-scale STFT discriminator, SI-SDR — no `encodec`/`descript-audio-
  codec`/`audiocraft`, no `torchaudio`/`librosa`/`soundfile`/`scipy`, no
  `DataLoader`. 7,338,658 params (+2,112,582 discriminator, training only);
  `--data-dir` points at an extracted LJSpeech-1.1 (13,100 wavs, 23.92 h,
  verified count refuses a partial extraction), stored as a memmapped
  `data/ljspeech_audio.i16` + index rather than an `.npz`. **Native 22,050
  Hz, no resampler** → 68.9 frames/s, 5.51 kbps at 8×1024 — don't "fix"
  these to EnCodec's 24 kHz / 75 Hz / 6 kbps. Quantizer dropout is
  load-bearing (one model serves the whole 1→8 ladder); the discriminator
  is staged behind `--adv-start-step` and `--lambda-adv 0` is the
  recon-only A/B. The dead-code cutoff is a **fraction of uniform codebook
  usage**, not the absolute 2.0 of EnCodec/`vector-quantize-pytorch` — at
  32×69 vectors over 1,024 entries uniform usage is only 2.16, and the
  first smoke run revived 1,023/1,024 entries per codebook before the fix.
  The discriminator is **~8x the cost of the rest of the step**, so
  `--disc-bf16 1` (default) autocasts **only the critic** to bf16 (0.95 →
  2.01 steps/s, 16.8 → 10.8 GiB); never extend that autocast over the
  generator — codebook lookup and EMA updates must stay fp32.
  **SI-SDR is a weak proxy for a GAN-trained codec**; judge by the emitted
  wav pairs and the per-codebook usage table. EMA weights are only better
  once converged — `evaluate_codec.py` warns when `ema_decay**global_step`
  still exceeds 1% (the 802-step smoke checkpoint was 45% init and scored
  worse than the live weights); keep that warning and the checkpoint fields
  it reads. No FID-equivalent
  (ViSQOL/PESQ/NISQA all need an external binary or pretrained network) by
  design. **Trained for real**: 60 epochs / 24,060 steps in 2 h 54 min, val
  mel 7.313 → 3.075, a monotone 1→8 ladder (5.51 kbps: +1.81 dB SI-SDR,
  3.083 mel) and **100% codebook usage on all eight** with the deepest
  codebook the highest-perplexity of the stack. It sounds like a working
  codec with an audible metallic edge, not a transparent one. Don't re-run
  training to chase better numbers unless asked — those results are the
  record. Details in `ARCHITECTURE.md` Stage 4.
- **`training/fashion-mnist-dcgan/`** — DCGAN
  ([Radford et al. 2015](https://arxiv.org/abs/1511.06434)) trained **from
  scratch** on Fashion-MNIST; the repo's **first GAN pipeline**. Hand-written
  torch: generator (`z ∈ ℝ¹⁰⁰ → 7×7×256 → 14×14×128 → 28×28×1`, Tanh),
  discriminator (strided convs 28→14→7→3→1, LeakyReLU(0.2), BN everywhere
  except the input layer), `N(0, 0.02)` init, one-sided label smoothing,
  Adam 2e-4 / betas (0.5, 0.999) on both nets, numpy-permutation batching,
  hand-written zlib PNG sample grids — no `torchvision`, no `DataLoader`.
  **28×28 breaks DCGAN's canonical 32×32 ladder**: D ends at a 3×3 feature
  map, G starts from 7×7 — the shapes in `train_dcgan.py` are the verified
  ones, don't "fix" them to 32×32. Build accepts a folder of IDX ubyte
  files (preferred) or the Kaggle CSVs, verifies 60,000/10,000, writes
  `data/fashion_mnist.npz` in [0,1] (trainer rescales to [-1,1]). Judge by
  `samples_grid.png` + the fixed-z grids saved during training, the
  nearest-neighbour memorization check and a pairwise-diversity probe —
  **no FID/IS by design** (pretrained Inception), same rule as FID in
  `flow-matching-mnist`. New pipeline — no verified-run numbers yet.
  Details in `ARCHITECTURE.md` Stage 4.
- **An Axolotl-based `fine-tuning/axolotl-ocr-summary/` pipeline existed
  earlier and was removed by the repo owner.** If something similar returns,
  note `axolotl[deepspeed]` only resolves its `uv` environment on Linux/WSL
  (`triton` has no Windows wheels) — a real platform constraint, not
  something to silently patch around.
- **Never commit data or weights** — one root `.gitignore` covers every
  pipeline's drop-zone folders (`DATASET/`, `data/`, `runs/`, `output/`,
  `outputs/`, `.cache/`, `hf_cache/`, `merged_model/`, model/checkpoint
  binaries). Don't add per-folder `.gitignore` files.

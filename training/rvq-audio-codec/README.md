# rvq-audio-codec

A neural audio codec with **residual vector quantization**, trained **from
scratch** on LJSpeech — the EnCodec / SoundStream / DAC architecture, with
no neural-codec library (no `encodec`, no `descript-audio-codec`, no
`audiocraft`) and no audio/DSP library either (no `torchaudio`, no
`librosa`, no `soundfile`, no `scipy`). The RIFF/WAV parser and writer, the
SEANet-style conv encoder and decoder, the residual vector quantizer with
its EMA codebook updates and dead-code re-initialization, the mel
filterbank, the multi-scale STFT discriminator, and the SI-SDR metric are
all written out by hand in `build_ljspeech_dataset.py` / `train_codec.py` /
`evaluate_codec.py`; torch supplies tensor ops, autograd, conv/STFT
primitives and GPU execution — the same role it plays in
`training/cifar10-vqvae` and `training/flow-matching-mnist`.

This is a `training/` pipeline (from-scratch, non-LoRA), an independent `uv`
project like every other pipeline folder in this repo — own
`pyproject.toml` (pinned to the CUDA 12.8 torch build), own
`.python-version`, no shared root environment.

**Why this architecture.** It is the layer every modern audio LM sits on:
VALL-E, MusicGen, Moshi and Kyutai all generate *codec tokens*, not
waveforms. And it is the direct generalization of the quantization already
in `training/cifar10-vqvae` — that folder has one codebook, this one stacks
eight, each quantizing the residual the previous one left behind. This is
also the first audio pipeline in the repo.

## Method

An autoencoder compresses a waveform to a low frame-rate sequence of latent
vectors, quantizes each one against a stack of codebooks, and decodes it
back. The compression is the whole point: what would travel over a wire is
the integer code indices, so the bitrate is exactly

```
bitrate = frame_rate x num_quantizers x log2(codebook_size)
```

### Why residual, and not one big codebook

A single codebook of `K` entries buys `log2(K)` bits per frame. Buying more
bits by growing `K` does not scale — 80 bits/frame would need `2^80`
entries, and every one of them would have to be searched. So the codebooks
are **stacked**, each quantizing what the previous one failed to capture:

```
residual = z
for q in quantizers:
    code_q    = nearest_entry(residual, q.codebook)
    quantized += code_q
    residual  -= code_q
```

`N` codebooks of `K` entries give `N*log2(K)` bits per frame while storing
`N*K` vectors instead of `K^N`. Eight codebooks of 1,024 entries is 80
bits/frame from 8,192 stored vectors.

### Codebook collapse, and the four things that stop it

Collapse — most entries never being selected, so the extra codebooks buy
nothing — is the failure mode that makes or breaks RVQ. Each mitigation is
behind a flag so it can be ablated, and every finding transfers straight
back to `training/cifar10-vqvae`:

| mitigation | flag | why it works |
|---|---|---|
| factorized low-dim codes | `--code-dim 8` | the nearest-neighbour lookup runs in an 8-dim projected space, not the 128-dim latent. The encoder only has to match 8 numbers to reach an entry, so far more entries stay reachable. (Descript's DAC finding.) |
| L2-normalized lookup | `--vq-l2-normalize 1` | assignment by cosine distance on unit-norm vectors, so an entry that merely drifted to a large norm cannot capture everything |
| EMA codebook updates | `--vq-mode ema` | entries are moving averages of the vectors that chose them, with no gradient — the same update `training/cifar10-vqvae` uses. `--vq-mode loss` switches to the classic VQ-VAE codebook gradient term for comparison |
| dead-code re-initialization | `--dead-code-threshold 0.05` | entries nobody is using are resampled from the live batch. Without it a code that goes unused once tends to stay unused forever: it is never the nearest neighbour, so it never moves |

Per-quantizer perplexity and unused-entry counts are printed every epoch.
That log is how you tell whether the 8th codebook is doing any work.

**One guardrail here came from getting it wrong first.** The dead-code
cutoff is expressed as a *fraction of uniform codebook usage*, not as the
absolute value of 2.0 that EnCodec and `vector-quantize-pytorch` use — and
the smoke run walked straight into why. One batch here is 32 x 69 = 2,208
vectors spread over 1,024 entries, so *uniform* usage is only 2.16 per
entry: an absolute cutoff of 2.0 condemns roughly half a perfectly healthy
codebook on every sweep and churns it forever. A genuinely dead entry
decays toward zero (x0.99 per step) and falls below a few percent of
uniform within a few hundred steps, so the relative cutoff catches it with
no false positives. The first smoke run reported 1,023 of 1,024 entries
"revived" per codebook; after the fix, 0.

### Quantizer dropout — one model, the whole bitrate ladder

`--quantizer-dropout 0.5` trains half of each batch at a random
`n_q` in `[1, N]` and the rest at full depth. That is what lets a **single**
trained model be evaluated at 1, 2, 4 and 8 codebooks. Without it the model
only ever sees all eight, and its output with four dropped is unusable —
you would have to train a separate model per bitrate to get the ladder.

### Losses

| term | weight | what it does |
|---|---|---|
| time-domain L1 | `--lambda-time 1.0` | keeps the waveform aligned at all |
| multi-scale mel L1 | `--lambda-mel 15.0` | seven window sizes (32 → 2048), log and linear magnitude. Short windows see transients, long ones see pitch; a codec judged at one resolution learns to cheat the other |
| commitment | `--lambda-commit 0.25` | pulls encoder outputs toward the entry they selected |
| adversarial (hinge) | `--lambda-adv 1.0` | multi-scale STFT discriminator |
| feature matching | `--lambda-feat 2.0` | L1 between the discriminator's intermediate activations on real and generated audio — in practice this carries most of the adversarial signal |

The reconstruction terms alone give a **muffled, over-smoothed** codec: L1
on magnitudes has no opinion about phase or fine texture, so the decoder
hedges. The multi-scale STFT discriminator is what fixes that. It is fed
the *complex* STFT (real and imaginary as two channels), not the magnitude,
because phase incoherence is exactly what makes a magnitude-trained codec
sound metallic and a magnitude-only critic cannot hear it.

The discriminator is **staged**: reconstruction only until
`--adv-start-step` (default 5,000, about 12 epochs at the default batch
size). A randomly-initialized generator fighting a randomly-initialized
discriminator is the classic way to collapse a codec in the first thousand
steps; 5,000 steps is long enough for the reconstruction to be sane and
short enough to leave most of a 60-epoch run adversarial. `--lambda-adv 0`
disables it entirely, which gives a clean reconstruction-only A/B.

## Model

```
Encoder                                            22,080 samples (1.0014 s)
  Conv1d(1 -> 32, k=7)
  stride 2:  2x ResidualUnit(32,  dil 1,3) -> Conv1d(32 -> 64,  k=4,  s=2)
  stride 4:  2x ResidualUnit(64,  dil 1,3) -> Conv1d(64 -> 128, k=8,  s=4)
  stride 5:  2x ResidualUnit(128, dil 1,3) -> Conv1d(128 -> 256, k=10, s=5)
  stride 8:  2x ResidualUnit(256, dil 1,3) -> Conv1d(256 -> 512, k=16, s=8)
  Conv1d(512 -> 128, k=7)                          69 frames x 128

RVQ  8 codebooks x 1024 entries, looked up in 8-dim factorized space
                                                   69 frames x 80 bits

Decoder  mirror of the encoder, ConvTranspose1d, final tanh
                                                   22,080 samples
```

ELU rather than ReLU throughout (waveforms are zero-centred; a hard zero
floor throws away the negative half of every sample), weight norm on every
convolution.

**7,338,658 parameters** at the defaults — 3,659,936 encoder, 3,660,162
decoder, 18,560 RVQ. The discriminator is a further 2,112,582 parameters
that exist only during training and are never used at inference.

A stride-`s` convolution with kernel `2s` needs a total padding of `s` to
map `L` samples onto exactly `L/s` frames. Two of the strides here are odd,
so that padding cannot be split evenly and `nn.Conv1d`'s symmetric
`padding=` argument cannot express it — it is applied explicitly with
`F.pad` and trimmed back off after each transposed convolution.

### Rate

LJSpeech is 22,050 Hz and is trained at that **native rate — no resampler
is written**. The stride stack (2, 4, 5, 8) gives a hop of 320 samples, so:

```
frame rate = 22050 / 320          = 68.9 Hz
bitrate    = 68.9 x 8 x log2(1024) = 5.51 kbps
```

EnCodec's published 75 Hz / 6 kbps is this same architecture at 24 kHz. The
numbers here are stated for what this actually runs at rather than quoting
figures from a rate this pipeline does not use.

### Cost, and where it actually goes

The discriminator is not a detail on the side — it is **8x the cost of
everything else in a training step**. Measured on an RTX 3090 at batch 32,
1-second crops:

| configuration | steps/s | peak VRAM | 60 epochs (24,060 steps) |
|---|---|---|---|
| reconstruction only (`--lambda-adv 0`) | 7.75 | 5.19 GiB | 0.9 h |
| fp32 discriminator | 0.95 | 16.78 GiB | 7.0 h |
| **bf16 discriminator** (default) | **2.01** | **10.80 GiB** | **3.3 h** |
| bf16 + `--disc-filters 16` | 2.92 | 7.96 GiB | 2.3 h |

Its spectrograms are far larger than the waveform they judge — 173 x 257
positions at the 512-point resolution against 22,080 samples, times three
resolutions, times three passes per step — and those odd conv shapes map
badly onto fp32 tensor cores.

So `--disc-bf16 1` is the default. **Only the critic** runs
reduced-precision: the generator, the codebook lookup and every EMA update
stay fp32, because bf16 EMA statistics would quietly stop accumulating
small updates, which is precisely the mechanism dead-code revival exists to
detect. bf16 rather than fp16 keeps fp32's exponent range, so no gradient
scaler is needed and the hinge loss cannot overflow. `torch.stft` has no
autocast policy, so it stays fp32 and only the conv stack is cast.
`--disc-bf16 0` forces fp32 throughout.

`--disc-filters 16` is faster still but is a real capacity cut (2.11M ->
0.53M discriminator parameters), so reach for bf16 first.

`cudnn.benchmark` and TF32 matmul were both measured and do nothing here
(0.90-0.95 steps/s, inside the noise) — the shapes are fixed but the
bottleneck is not algorithm selection.

## Dataset

[LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) — 13,100
single-speaker (Linda Johnson) LibriVox readings, 22,050 Hz / 16-bit / mono
PCM, **23.92 hours**. Point `--data-dir` at the extracted folder; a nested
`LJSpeech-1.1/` subfolder is also accepted. Not checked into this repo
(`data/` is gitignored via the root `.gitignore`). On the repo owner's
machine: `E:\datasets\LJSpeech-1.1`.

The transcripts in `metadata.csv` are deliberately **unused** — a codec is
trained by self-supervised reconstruction and never sees text.

`build_ljspeech_dataset.py` verifies exactly 13,100 wav files and refuses
to build on a partial extraction, the same guardrail
`training/imdb-sentiment-cnn` uses for its 12,500-file splits. It writes
`data/ljspeech_audio.i16` (every utterance concatenated as raw int16,
3.80 GB, opened with `np.memmap`) plus a small `data/ljspeech_index.npz` of
offsets/lengths/ids/split — **not** an `.npz` of samples the way the MNIST
and CIFAR builders do, because 3.8 GB of int16 becomes 7.6 GB as float32
and will not sit comfortably in RAM. Crops are converted to float one batch
at a time.

Measured on the real corpus: shortest utterance 1.11 s, longest 10.10 s,
mean 6.57 s. Every utterance clears the 1.0014 s training crop, so none are
dropped.

### Is one corpus enough?

For this pipeline, yes. Codec training is self-supervised reconstruction —
it needs audio, not labels — and every metric here is reference-based on
held-out audio from the same corpus, so no second dataset and no pretrained
network is involved.

The honest limitation: a codec trained only on LJSpeech is a
**single-speaker, single-recording-condition, speech-only** codec. It
overfits one voice and one LibriVox room, and will degrade on other
speakers, on music, and on noisy audio. LJSpeech also makes a
speaker-independent split impossible — the val split is held-out
*utterances* (262 of them), not held-out speakers. If generalization
becomes the goal, the natural additions are LibriTTS (585 h multi-speaker,
natively 24 kHz), VCTK (110 speakers), and MUSDB18 or FMA for music;
`--data-dir` points at a wav tree, so that is a small change.

## Commands

```sh
uv run --directory training/rvq-audio-codec python build_ljspeech_dataset.py --data-dir "E:\datasets\LJSpeech-1.1" --output-dir data
uv run --directory training/rvq-audio-codec python -u train_codec.py --data-dir data --num-quantizers 8 --codebook-size 1024 --num-epochs 60 --batch-size 32 --output-dir runs/ljspeech_codec
uv run --directory training/rvq-audio-codec python -u evaluate_codec.py --data-dir data --checkpoint-path runs/ljspeech_codec/codec_best.pt --output-dir runs/ljspeech_codec
```

`python -u` is worth keeping on the training command: without it Python
block-buffers stdout when redirected and a three-hour run shows nothing
until it finishes.

One-time environment bootstrap (creates `.venv`, installs the CUDA torch
wheel, verifies `torch.cuda.is_available()`):

```sh
training\rvq-audio-codec\uv_setup.bat
```

`train_codec.py` runs a hand-written AdamW training loop with two
optimizers (generator and discriminator), numpy-permutation batching over
utterance indices with one random crop each, no
`torch.utils.data.DataLoader`. It saves the best-val-mel checkpoint to
`runs/ljspeech_codec/codec_best.pt` plus a final checkpoint; both carry the
raw weights, the EMA weights, the discriminator, and the full model config.

Useful variations:

```sh
# reconstruction-only baseline (no GAN at all) - the A/B for "does the discriminator matter"
... python train_codec.py --lambda-adv 0 --lambda-feat 0 --output-dir runs/recon_only

# ablate the collapse mitigations one at a time
... python train_codec.py --code-dim 128 --output-dir runs/no_factorization
... python train_codec.py --vq-l2-normalize 0 --output-dir runs/no_l2norm
... python train_codec.py --dead-code-threshold 0 --output-dir runs/no_revival
... python train_codec.py --vq-mode loss --output-dir runs/codebook_loss
```

## How this is judged

**By listening.** `evaluate_codec.py` writes `original_NN.wav` next to
`recon_nq8_NN.wav` / `recon_nq4_NN.wav` / `recon_nq2_NN.wav` /
`recon_nq1_NN.wav` for a handful of held-out utterances. Play them in order
and the bitrate ladder is audible — that side-by-side is the evaluation the
printed numbers are only a proxy for.

The numbers it prints, at every rung of the ladder (all served by the same
model, thanks to quantizer dropout):

| metric | meaning |
|---|---|
| bitrate | `frame_rate x n_q x log2(codebook_size)`, in kbps. Exact, not measured — it is what the integer indices cost |
| SI-SDR | scale-invariant signal-to-distortion ratio, dB |
| mel distance | the multi-scale mel L1 the model was trained on, so it is directly comparable to the training log |
| log-STFT distance | L1 between log magnitude spectra at three resolutions — linear frequency bins, so it weights the top octaves the mel scale compresses away |
| codebook usage | % of each codebook's entries the held-out set touches, plus perplexity. **The collapse check**: if quantizer 8 uses 3% of its entries, the last codebook is decoration |

Plus `spectrogram_grid.png` — original above, reconstruction below, on the
same 80 dB grey scale — written with the same hand-written stdlib `zlib`
PNG encoder `training/flow-matching-mnist` uses.

Two honest warnings, in the same spirit as that folder's note that its
round-trip sweep measures ODE discretization error rather than sample
quality:

- **SI-SDR is a weak proxy for a GAN-trained codec.** The adversarial loss
  deliberately trades exact waveform/phase alignment for perceptual
  realism, so a model that sounds clearly better can post a *worse* SI-SDR
  than a reconstruction-only one. It is reported because it is the standard
  number and because it is meaningful for the recon-only baseline — not
  because higher means it sounds better.
- **There is deliberately no ViSQOL, PESQ or NISQA.** Each needs an external
  binary or a pretrained network, which is the same rule that keeps FID out
  of `training/flow-matching-mnist`. A hand-rolled substitute perceptual
  score would be worse than no score at all.
- **The EMA weights are only better once they have converged**, and on a
  short run they are much worse. After `n` steps they still carry
  `ema_decay**n` of the random initialization: at the default decay 0.999
  the 802-step smoke checkpoint is **45% initialization**, and it really did
  score 9.02 mel with a non-monotone ladder against the live weights' 7.42
  with a monotone one. A full 60-epoch run leaves that behind entirely
  (`0.999**24060 = 4e-11`), so `--use-ema 1` is the right default — but
  `evaluate_codec.py` now computes that share from the checkpoint and prints
  a warning when it exceeds 1%, which is exactly when you should compare
  against `--use-ema 0` before believing anything.

## Verified runs

<!-- filled in from real runs on the repo owner's RTX 3090; no invented numbers -->

**Build** — 13,100 utterances read through the hand-written RIFF parser,
all confirmed 22,050 Hz / 16-bit / mono PCM: 1,898,881,532 samples,
**23.92 h**, 3.80 GB. 12,838 train / 262 val. Shortest utterance 1.11 s,
longest 10.10 s, mean 6.57 s — nothing is dropped by the 1.0014 s crop.

**Model** — 7,338,658 parameters (3,659,936 encoder / 3,660,162 decoder /
18,560 RVQ), plus a 2,112,582-parameter discriminator used only in
training.

**I/O** — the hand-written WAV writer round-trips through the hand-written
parser at the 16-bit quantization floor (max abs error 5.32e-05 against a
floor of 3.05e-05), clipping clamps rather than wraps, SI-SDR is
scale-invariant (116.4 dB for x vs x, 122.5 dB for 2x vs x), and
`log_stft_distance(x, x)` is exactly 0.

**2-epoch smoke run** (802 steps, discriminator from step 0, fp32):

```
epoch 1/2  time=0.0421 mel=8.510 commit=38.69 adv=2.357 feat=0.377 d=0.932 | val mel=7.363
    codebook perplexity per q: 286 155 167 258 153 199 206 253
    unused entries per q:      437 576 507 447 578 508 526 495   revived: 3087
epoch 2/2  time=0.0434 mel=7.352 commit=79.70 adv=2.901 feat=0.290 d=0.565 | val mel=7.224
    codebook perplexity per q: 251 252 244 265 227 278 222 299
    unused entries per q:      375 404 447 410 468 389 444 364   revived: 788
```

No NaN, both optimizers stepping, val mel falling. The codebook columns are
the ones that matter and they are healthy: unused entries dropping across
every quantizer, revivals falling 3,087 -> 788, and **the 8th codebook is as
busy as the 1st** (perplexity 299 vs 251) — no collapse down the stack,
which is the whole risk with RVQ.

**bf16 discriminator, 1 epoch** (401 steps) — the same epoch in fp32 and in
bf16, to check the speedup costs nothing:

| | fp32 | bf16 critic |
|---|---|---|
| train mel | 8.510 | 8.516 |
| **val mel** | 7.363 | **7.287** |
| commitment | 38.69 | 33.55 |
| epoch wall time | 435.8 s | **212.6 s** |
| peak GPU memory | 16.78 GiB | **10.78 GiB** |

Indistinguishable on loss (the val difference is noise), half the time, two
thirds the memory. The reported peak of 10.78 GiB matches the benchmark's
10.80 GiB, so a 60-epoch run fits with ~13 GiB of headroom on a 24 GB card.

**Evaluator, end-to-end on that 2-epoch checkpoint** (16 held-out
utterances, 119.2 s) — run to prove the path works, *not* as a quality
result; a 2-epoch codec sounds like noise:

```
  n_q   kbps    SI-SDR      mel     log-STFT      <- live weights (--use-ema 0)
    1   0.69   -43.08 dB    7.713     1.7948
    2   1.38   -44.22 dB    7.533     1.7479
    4   2.76   -42.14 dB    7.425     1.6927
    8   5.51   -41.02 dB    7.422     1.6824
```

The ladder is already monotone on both mel and log-STFT after two epochs,
which is the shape it should have. All eleven artifacts wrote correctly
(four `recon_nq*` wavs per utterance, the originals, and
`spectrogram_grid.png`).

_Pending: the full 60-epoch run and its evaluation ladder._

## Compared with `training/cifar10-vqvae`

Same idea, one generation apart:

| | `cifar10-vqvae` | `rvq-audio-codec` |
|---|---|---|
| codebooks | 1 x 512 | 8 x 1024 |
| lookup space | the full 64-dim latent | 8-dim factorized projection |
| lookup metric | squared L2 | cosine (unit-norm) |
| collapse handling | EMA updates | EMA + factorization + L2 norm + dead-code revival |
| bits per position | 9 | 80 |
| reconstruction loss | MSE on pixels | time L1 + multi-scale mel L1 |
| adversarial | none | multi-scale STFT discriminator |

The single-codebook version is the honest starting point and still the
clearer place to read the straight-through estimator. What RVQ adds is the
bit budget: 9 bits per latent position is enough for a 32x32 thumbnail and
nowhere near enough for a waveform.

## Files

- `build_ljspeech_dataset.py` — hand-written RIFF/WAVE parser, memmappable int16 corpus + index, 13,100-file integrity check
- `train_codec.py` — SEANet encoder/decoder, residual vector quantizer (factorized, L2-normalized, EMA, dead-code revival), quantizer dropout, mel filterbank, multi-scale mel loss, multi-scale STFT discriminator, two-optimizer training loop
- `evaluate_codec.py` — bitrate ladder, SI-SDR, mel and log-STFT distances, codebook usage, hand-written WAV writer and zlib PNG output

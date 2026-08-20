"""
Evaluate a trained RVQ audio codec on the held-out LJSpeech utterances,
and - more importantly - write out the audio so it can actually be
listened to.

What this prints, for every rung of the codebook ladder (n_q = 1, 2, 4, 8
by default, all served by the *same* trained model thanks to quantizer
dropout):

  bitrate        frame_rate * n_q * log2(codebook_size), in kbps. Exact,
                 not measured - it is what the integer indices cost.
  SI-SDR         scale-invariant signal-to-distortion ratio, dB.
  mel distance   the multi-scale mel L1 the model was trained on, so the
                 eval number is directly comparable to the training log.
  log-STFT dist  L1 between log magnitude spectra at three resolutions.
  codebook use   % of each codebook's entries the val set actually touches,
                 plus perplexity. This is the collapse check: if quantizer
                 8 uses 3% of its entries, the last codebook is decoration.

Two honest warnings about these numbers, in the same spirit as
training/flow-matching-mnist's note that its round-trip sweep measures ODE
discretization error rather than sample quality:

  * SI-SDR is a weak proxy for a GAN-trained codec. The adversarial loss
    deliberately trades exact waveform/phase alignment for perceptual
    realism, so a model that sounds clearly better can post a *worse*
    SI-SDR than a reconstruction-only one. It is reported because it is
    the standard number and because it is meaningful for the recon-only
    baseline - not because a higher value means it sounds better.
  * There is deliberately **no ViSQOL, PESQ or NISQA**. Each needs an
    external binary or a pretrained network, which is the same rule that
    keeps FID out of training/flow-matching-mnist. A hand-rolled substitute
    perceptual score would be worse than no score at all.

The real evaluation is the wav files this writes: original_NN.wav next to
recon_nq8_NN.wav / recon_nq4_NN.wav / recon_nq2_NN.wav / recon_nq1_NN.wav.
Play them in order and the bitrate ladder is audible.

The WAV writer here is hand-written (a 44-byte RIFF header packed with
stdlib struct), deliberately mirroring the hand-written zlib PNG writer in
training/flow-matching-mnist's evaluator - no soundfile, no torchaudio.

Usage:
    uv run --directory training/rvq-audio-codec python evaluate_codec.py \
        --data-dir data \
        --checkpoint-path runs/ljspeech_codec/codec_best.pt \
        --output-dir runs/ljspeech_codec
"""

import argparse
import math
import os
import struct
import zlib

import numpy as np
import torch

from train_codec import (
    DISC_RESOLUTIONS,
    AudioCodec,
    MultiScaleMelLoss,
    load_corpus,
    mel_filterbank,
)


def write_wav(path, samples, sample_rate):
    """Write mono 16-bit PCM. The canonical 44-byte header: RIFF chunk,
    a 16-byte PCM `fmt ` chunk, then `data`. The inverse of the parser in
    build_ljspeech_dataset.py."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()

    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH",
        16,             # fmt chunk size
        1,              # PCM
        1,              # mono
        sample_rate,
        sample_rate * 2,  # byte rate = rate * channels * bytes/sample
        2,              # block align
        16,             # bits per sample
    )
    header += b"data" + struct.pack("<I", len(pcm))

    with open(path, "wb") as f:
        f.write(header + pcm)


def write_png(path, array_2d_uint8):
    """Single-channel 8-bit grayscale PNG: IHDR + one zlib-compressed IDAT
    (filter byte 0 per scanline) + IEND. Same minimal encoder as
    training/flow-matching-mnist's evaluator."""
    height, width = array_2d_uint8.shape

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    raw = b"".join(b"\x00" + array_2d_uint8[y].tobytes() for y in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=9))
    png += chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)


def si_sdr(estimate, reference, epsilon=1e-8):
    """Scale-invariant SDR in dB: project the estimate onto the reference,
    call that the signal and the rest distortion. Scale-invariant because
    the projection absorbs any overall gain difference - a codec that comes
    back 3 dB quiet is not thereby wrong."""
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    alpha = (estimate * reference).sum() / (reference.pow(2).sum() + epsilon)
    target = alpha * reference
    noise = estimate - target
    return 10.0 * torch.log10(
        (target.pow(2).sum() + epsilon) / (noise.pow(2).sum() + epsilon)
    )


def log_stft_distance(estimate, reference, resolutions=DISC_RESOLUTIONS):
    """Mean L1 between log magnitude spectrograms, averaged over
    resolutions. Complements the mel loss: linear frequency bins, so it
    weights the top octaves the mel scale compresses away."""
    total = 0.0
    for n_fft, hop in resolutions:
        window = torch.hann_window(n_fft, device=estimate.device)
        kwargs = dict(n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, center=True, pad_mode="reflect",
                      return_complex=True)
        mag_e = torch.stft(estimate, **kwargs).abs().clamp_min(1e-5)
        mag_r = torch.stft(reference, **kwargs).abs().clamp_min(1e-5)
        total += (mag_e.log() - mag_r.log()).abs().mean().item()
    return total / len(resolutions)


def log_mel_image(waveform, sample_rate, n_mels=80, n_fft=1024,
                  reference=None):
    """(mel_bands, frames) uint8 image of a log-mel spectrogram, low
    frequencies at the bottom. When `reference` is given, its dynamic range
    sets the grey scale, so an original and its reconstruction are shaded
    identically and can be compared by eye."""
    device = waveform.device
    filterbank = mel_filterbank(n_mels, n_fft, sample_rate).to(device)
    spec = torch.stft(
        waveform, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
        window=torch.hann_window(n_fft, device=device), center=True,
        pad_mode="reflect", return_complex=True,
    ).abs()
    # (1, n_mels, frames) -> (n_mels, frames): write_png takes a 2-D array.
    log_mel = (filterbank @ spec).clamp_min(1e-5).log10().squeeze(0) * 20.0

    source = log_mel if reference is None else reference
    high = source.max()
    low = high - 80.0                      # 80 dB of dynamic range shown
    scaled = ((log_mel - low) / (high - low)).clamp(0.0, 1.0)
    image = (scaled * 255.0).to(torch.uint8).cpu().numpy()
    return np.flipud(image), log_mel


def pad_to_hop(x, hop):
    """Pad a waveform up to a whole number of codec frames. The padding is
    trimmed back off the reconstruction, so it never reaches a metric."""
    remainder = x.shape[-1] % hop
    if remainder == 0:
        return x, 0
    pad = hop - remainder
    return torch.nn.functional.pad(x, (0, pad)), pad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-path",
                        default="runs/ljspeech_codec/codec_best.pt")
    parser.add_argument("--output-dir", default="runs/ljspeech_codec")
    parser.add_argument("--ladder", default="1,2,4,8",
                        help="Comma-separated codebook counts to evaluate. All are "
                             "served by the same model - that is what quantizer "
                             "dropout during training buys.")
    parser.add_argument("--num-utterances", type=int, default=64,
                        help="Held-out utterances to compute metrics over.")
    parser.add_argument("--num-wavs", type=int, default=4,
                        help="How many utterances to write out as wav files.")
    parser.add_argument("--use-ema", type=int, default=1,
                        help="1 = evaluate the EMA weights (recommended for a "
                             "GAN-trained model), 0 = the live weights.")
    parser.add_argument("--png-row-scale", type=int, default=2,
                        help="Vertical pixel repeat for the spectrogram png.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device,
                            weights_only=False)
    model = AudioCodec(
        base_channels=checkpoint["base_channels"],
        latent_dim=checkpoint["latent_dim"],
        code_dim=checkpoint["code_dim"],
        codebook_size=checkpoint["codebook_size"],
        num_quantizers=checkpoint["num_quantizers"],
        strides=tuple(checkpoint["strides"]),
        ema_decay=checkpoint.get("vq_ema_decay", 0.99),
        l2_normalize=checkpoint.get("vq_l2_normalize", True),
    ).to(device)

    state = checkpoint["model_state"]
    if args.use_ema and checkpoint.get("ema_state") is not None:
        # The EMA weights are the right ones to judge a GAN-trained model by
        # - but only once they have converged. After n steps they still carry
        # ema_decay**n of the random initialization, and on a short run that
        # is not a rounding error: a 2-epoch (802-step) smoke checkpoint at
        # decay 0.999 is 45% initialization, and really did score 9.02 mel
        # against the live weights' 7.42, with a non-monotone ladder on top.
        # A full run leaves this far behind (0.999**24060 = 4e-11), so the
        # warning fires exactly when it should.
        decay = checkpoint.get("ema_decay")
        steps = checkpoint.get("global_step")
        if decay and steps:
            init_share = decay ** steps
            if init_share > 0.01:
                print(f"WARNING: the EMA weights still carry "
                      f"{100.0 * init_share:.1f}% of the random initialization "
                      f"({steps} steps at decay {decay}). On a run this short "
                      f"they are worse than the live weights, not better - "
                      f"compare against --use-ema 0 before believing anything "
                      f"below.")
        print("Using EMA weights")
    model.load_state_dict(state)
    model.eval()

    sample_rate = checkpoint["sample_rate"]
    hop = int(np.prod(checkpoint["strides"]))
    frame_rate = sample_rate / hop
    codebook_size = checkpoint["codebook_size"]
    bits_per_code = math.log2(codebook_size)

    print(f"Checkpoint: epoch {checkpoint['epoch']}, "
          f"val mel {checkpoint['val_mel']:.3f}")
    print(f"{sample_rate} Hz, hop {hop} -> {frame_rate:.1f} frames/s, "
          f"{checkpoint['num_quantizers']} x {codebook_size} codebooks")

    audio, index = load_corpus(args.data_dir)
    offsets, lengths, split = index["offsets"], index["lengths"], index["split"]
    utterance_ids = index["utterance_ids"]
    val_idx = np.nonzero(split == 1)[0]
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(val_idx)[:args.num_utterances]

    mel_loss_fn = MultiScaleMelLoss(sample_rate).to(device)
    ladder = [int(n) for n in args.ladder.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Held-out waveforms, loaded once and reused for every rung.
    waveforms = []
    for i in chosen:
        start, length = int(offsets[i]), int(lengths[i])
        samples = np.asarray(audio[start:start + length], dtype=np.float32) / 32768.0
        waveforms.append(torch.from_numpy(samples).to(device))

    results = {}
    usage_totals = torch.zeros(checkpoint["num_quantizers"], codebook_size,
                               device=device)

    with torch.no_grad():
        for n_q in ladder:
            sdr_total, mel_total, stft_total = 0.0, 0.0, 0.0
            for waveform in waveforms:
                x = waveform.view(1, 1, -1)
                padded, pad = pad_to_hop(x, hop)
                x_hat = model.reconstruct(padded, n_quantizers=n_q)
                if pad:
                    x_hat = x_hat[..., :x.shape[-1]]

                sdr_total += si_sdr(x_hat.view(-1), x.view(-1)).item()
                mel_total += mel_loss_fn(x_hat.view(1, -1), x.view(1, -1)).item()
                stft_total += log_stft_distance(x_hat.view(-1), x.view(-1))

            n = len(waveforms)
            results[n_q] = {
                "kbps": frame_rate * n_q * bits_per_code / 1000.0,
                "si_sdr": sdr_total / n,
                "mel": mel_total / n,
                "stft": stft_total / n,
            }

        # Codebook usage over the whole held-out set, at full depth.
        for waveform in waveforms:
            padded, _ = pad_to_hop(waveform.view(1, 1, -1), hop)
            z = model.encoder(padded)
            _, _, _, indices, _ = model.quantizer(z, ema_enabled=False)
            for q, idx in enumerate(indices):
                usage_totals[q] += torch.bincount(
                    idx.reshape(-1), minlength=codebook_size
                ).float()

    print(f"\nHeld-out utterances: {len(waveforms)}  "
          f"({sum(w.shape[0] for w in waveforms) / sample_rate:.1f} s of audio)")
    print("\n  n_q   kbps    SI-SDR      mel     log-STFT")
    print("  " + "-" * 44)
    for n_q in ladder:
        r = results[n_q]
        print(f"  {n_q:3d}  {r['kbps']:5.2f}  {r['si_sdr']:7.2f} dB  "
              f"{r['mel']:7.3f}  {r['stft']:9.4f}")

    print(f"\nCodebook usage on the held-out set (of {codebook_size} entries):")
    print("    q   used      %   perplexity")
    print("  " + "-" * 34)
    for q in range(checkpoint["num_quantizers"]):
        usage = usage_totals[q]
        used = int((usage > 0).sum())
        p = usage / usage.sum().clamp_min(1)
        perplexity = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
        print(f"  {q + 1:3d}  {used:5d}  {100.0 * used / codebook_size:5.1f}  "
              f"{perplexity:11.1f}")

    # ---- audio and spectrogram artifacts ----
    print()
    with torch.no_grad():
        for k in range(min(args.num_wavs, len(waveforms))):
            waveform = waveforms[k]
            name = str(utterance_ids[chosen[k]])
            original_path = os.path.join(args.output_dir, f"original_{k:02d}.wav")
            write_wav(original_path, waveform.cpu().numpy(), sample_rate)
            print(f"wrote {original_path}  ({name}, "
                  f"{waveform.shape[0] / sample_rate:.2f} s)")

            padded, _ = pad_to_hop(waveform.view(1, 1, -1), hop)
            for n_q in ladder:
                x_hat = model.reconstruct(padded, n_quantizers=n_q)
                x_hat = x_hat[..., :waveform.shape[0]].view(-1)
                path = os.path.join(args.output_dir, f"recon_nq{n_q}_{k:02d}.wav")
                write_wav(path, x_hat.cpu().numpy(), sample_rate)
                print(f"wrote {path}  ({results[n_q]['kbps']:.2f} kbps)")

        # One original-vs-reconstruction log-mel pair, full depth.
        waveform = waveforms[0]
        padded, _ = pad_to_hop(waveform.view(1, 1, -1), hop)
        recon = model.reconstruct(padded, n_quantizers=max(ladder))
        recon = recon[..., :waveform.shape[0]].view(1, -1)

        original_image, original_log_mel = log_mel_image(
            waveform.view(1, -1), sample_rate
        )
        recon_image, _ = log_mel_image(
            recon, sample_rate, reference=original_log_mel
        )
        gap = np.zeros((4, original_image.shape[1]), dtype=np.uint8)
        grid = np.concatenate([original_image, gap, recon_image], axis=0)
        grid = np.repeat(grid, args.png_row_scale, axis=0)
        png_path = os.path.join(args.output_dir, "spectrogram_grid.png")
        write_png(png_path, grid)
        print(f"wrote {png_path}  (original on top, reconstruction below, "
              f"same 80 dB scale)")

    print("\nNow listen to the wavs - that is the evaluation the numbers "
          "above are only a proxy for.")


if __name__ == "__main__":
    main()

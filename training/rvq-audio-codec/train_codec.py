"""
A neural audio codec with residual vector quantization (RVQ), trained
**from scratch** on LJSpeech - the EnCodec / SoundStream / DAC
architecture, with no neural-codec library (no encodec, no
descript-audio-codec, no audiocraft) and no audio/DSP library either (no
torchaudio, no librosa, no soundfile, no scipy). The SEANet-style conv
encoder and decoder, the residual vector quantizer with its EMA codebook
updates and dead-code re-initialization, the mel filterbank, and the
multi-scale STFT discriminator are all written out below as plain torch
modules; torch supplies tensor ops, autograd, conv/STFT primitives and GPU
execution, the same role it plays in training/cifar10-vqvae and
training/flow-matching-mnist.

The method, in full:

  An autoencoder compresses a waveform to a low frame-rate sequence of
  latent vectors, quantizes each one against a stack of codebooks, and
  decodes it back. The compression is the point: what travels is the
  integer code indices, so the bitrate is exactly

      frame_rate x num_quantizers x log2(codebook_size)   bits/second

  RVQ is what makes that bitrate reachable. A single codebook of K entries
  buys log2(K) bits per frame, and buying more bits by growing K is
  hopeless - 80 bits/frame would need 2^80 entries. So instead N codebooks
  are stacked, each quantizing what the previous one *failed* to capture:

      residual = z
      for q in quantizers:
          code_q   = nearest_entry(residual, q.codebook)
          quantized += code_q
          residual  -= code_q

  N codebooks of K entries then give N*log2(K) bits/frame at N*K entries of
  storage instead of K^N. This is the direct generalization of the single
  codebook in training/cifar10-vqvae, and the reason every modern audio LM
  (VALL-E, MusicGen, Moshi) can treat audio as a short sequence of tokens.

  Codebook collapse - most entries never being selected, so the extra
  codebooks buy nothing - is the failure mode that makes or breaks this.
  Four standard mitigations are implemented, each behind a flag so they can
  be ablated (the findings transfer straight back to cifar10-vqvae):
    1. factorized low-dim codes  (--code-dim 8): the nearest-neighbour
       lookup happens in an 8-dim projected space, not the 128-dim latent
    2. L2-normalized lookup      (--vq-l2-normalize): assignment by cosine
       distance, so a few high-norm entries cannot win everything
    3. EMA codebook updates      (--vq-mode ema): entries are moving
       averages of the vectors that chose them, no gradient
    4. dead-code re-init         (--dead-code-threshold): entries nobody
       has used lately are resampled from the live batch
  Per-quantizer perplexity and dead-code counts are logged every epoch;
  that log is how you tell whether the 8th codebook is doing any work.

  Quantizer dropout (--quantizer-dropout) trains a random n_q in [1, N] on
  part of each batch, so ONE trained model serves the whole 1->8 codebook
  bitrate ladder rather than needing eight separate runs.

  The reconstruction losses are a time-domain L1 plus a multi-scale mel L1
  (seven window sizes from 32 to 2048). Those alone give a muffled,
  over-smoothed codec: L1 on magnitudes has no opinion about phase or fine
  texture, so the decoder hedges. The multi-scale STFT discriminator is
  what fixes that, and it is staged - reconstruction only until
  --adv-start-step, because a randomly-initialized generator fighting a
  randomly-initialized discriminator is the classic way to collapse a
  codec in the first thousand steps. --lambda-adv 0 disables it entirely
  for a clean reconstruction-only comparison.

LJSpeech is 22,050 Hz and is trained at that native rate - no resampler is
written. The stride stack (2, 4, 5, 8) gives a hop of 320, so the frame
rate is 22050/320 = 68.9 Hz and the default 8 x 1024 codebooks come to
68.9 * 8 * 10 = 5.5 kbps. (EnCodec's published 75 Hz / 6 kbps is the same
architecture at 24 kHz; the numbers here are stated for what this actually
runs at.)

Usage:
    uv run --directory training/rvq-audio-codec python train_codec.py \
        --data-dir data \
        --num-quantizers 8 \
        --codebook-size 1024 \
        --num-epochs 60 \
        --batch-size 32 \
        --output-dir runs/ljspeech_codec
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

# The stride stack, applied in order by the encoder and in reverse by the
# decoder. Their product is the hop: 2*4*5*8 = 320 samples per frame.
DEFAULT_STRIDES = (2, 4, 5, 8)

# Window sizes for the multi-scale mel reconstruction loss, with the number
# of mel bands used at each. Short windows see transients, long windows see
# pitch; a codec judged at only one resolution learns to cheat the other.
MEL_WINDOWS = (32, 64, 128, 256, 512, 1024, 2048)
MEL_BANDS = (5, 10, 20, 40, 80, 160, 320)

# Resolutions of the multi-scale STFT discriminator (n_fft, hop).
DISC_RESOLUTIONS = ((2048, 512), (1024, 256), (512, 128))


# ---------------------------------------------------------------------------
# Padding helpers
#
# A stride-s convolution with kernel 2s needs a total padding of s to map a
# length-L signal onto exactly L/s frames. s is odd for two of the strides
# here (5), so that padding cannot be split evenly and nn.Conv1d's symmetric
# `padding=` argument cannot express it. It is applied explicitly instead,
# and the matching transposed conv trims the same amounts back off.
# ---------------------------------------------------------------------------

def _pad_amounts(kernel, stride):
    total = kernel - stride
    return total - total // 2, total // 2


def pad_for_stride(x, kernel, stride):
    left, right = _pad_amounts(kernel, stride)
    return F.pad(x, (left, right))


def trim_after_transpose(x, kernel, stride):
    left, right = _pad_amounts(kernel, stride)
    return x[..., left:x.shape[-1] - right]


# ---------------------------------------------------------------------------
# Encoder / decoder (SEANet-style)
# ---------------------------------------------------------------------------

class ResidualUnit(nn.Module):
    """Two convolutions with a skip connection: a dilated 3-tap conv that
    widens the receptive field, then a 1-tap mixer. ELU rather than ReLU
    (waveforms are zero-centred and a hard zero floor throws away the
    negative half of every sample), weight norm on both convs."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation,
                      padding=dilation)
        )
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        h = self.conv1(F.elu(x))
        h = self.conv2(F.elu(h))
        return x + h


class EncoderBlock(nn.Module):
    """Two residual units at the current resolution, then a strided conv
    that halves/quarters/... the time axis and doubles the channels."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.units = nn.ModuleList(
            [ResidualUnit(in_channels, dilation=d) for d in (1, 3)]
        )
        self.stride = stride
        self.kernel = 2 * stride
        self.downsample = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel,
                      stride=stride)
        )

    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return self.downsample(pad_for_stride(F.elu(x), self.kernel, self.stride))


class DecoderBlock(nn.Module):
    """Mirror of EncoderBlock: a transposed conv that expands the time axis
    and halves the channels, then two residual units."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.kernel = 2 * stride
        self.upsample = weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=self.kernel,
                               stride=stride)
        )
        self.units = nn.ModuleList(
            [ResidualUnit(out_channels, dilation=d) for d in (1, 3)]
        )

    def forward(self, x):
        x = trim_after_transpose(self.upsample(F.elu(x)), self.kernel, self.stride)
        for unit in self.units:
            x = unit(x)
        return x


class Encoder(nn.Module):
    """waveform (B, 1, L) -> latents (B, latent_dim, L / prod(strides))."""

    def __init__(self, base_channels, latent_dim, strides):
        super().__init__()
        self.stem = weight_norm(nn.Conv1d(1, base_channels, kernel_size=7, padding=3))
        blocks = []
        channels = base_channels
        for stride in strides:
            blocks.append(EncoderBlock(channels, channels * 2, stride))
            channels *= 2
        self.blocks = nn.ModuleList(blocks)
        self.out = weight_norm(
            nn.Conv1d(channels, latent_dim, kernel_size=7, padding=3)
        )

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.out(F.elu(h))


class Decoder(nn.Module):
    """latents (B, latent_dim, T) -> waveform (B, 1, T * prod(strides))."""

    def __init__(self, base_channels, latent_dim, strides):
        super().__init__()
        channels = base_channels * (2 ** len(strides))
        self.stem = weight_norm(
            nn.Conv1d(latent_dim, channels, kernel_size=7, padding=3)
        )
        blocks = []
        for stride in reversed(strides):
            blocks.append(DecoderBlock(channels, channels // 2, stride))
            channels //= 2
        self.blocks = nn.ModuleList(blocks)
        self.out = weight_norm(nn.Conv1d(channels, 1, kernel_size=7, padding=3))

    def forward(self, z):
        h = self.stem(z)
        for block in self.blocks:
            h = block(h)
        # tanh bounds the output to [-1, 1], the same range int16 samples are
        # scaled into, so the decoder cannot "win" reconstruction loss by
        # emitting values no wav file could hold.
        return torch.tanh(self.out(F.elu(h)))


# ---------------------------------------------------------------------------
# Residual vector quantization
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """One codebook. Generalizes training/cifar10-vqvae's VectorQuantizer
    with the three collapse mitigations that a stack of these needs:

      * factorized codes - the lookup runs in a `code_dim`-dimensional
        projected space (8 by default) rather than the full latent width.
        Low-dimensional codes are far easier to keep alive: the encoder
        only has to match 8 numbers, not 128, so more entries stay
        reachable. (Descript's DAC finding.)
      * L2-normalized lookup - assignment by cosine distance instead of
        raw Euclidean, so entries that merely drifted to a large norm
        cannot capture everything. With EMA updates the codebook is
        re-normalized after each update, which keeps every entry on the
        unit sphere where the encoder outputs are being pushed.
      * dead-code re-initialization - entries whose EMA usage falls below a
        threshold are resampled from the current batch's encoder outputs.
        Without it, a code that goes unused once tends to stay unused
        forever: it is never the nearest neighbour, so it never moves.
    """

    def __init__(self, latent_dim, code_dim, codebook_size, ema_decay=0.99,
                 l2_normalize=True, epsilon=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.l2_normalize = l2_normalize
        self.epsilon = epsilon

        self.in_proj = weight_norm(nn.Conv1d(latent_dim, code_dim, kernel_size=1))
        self.out_proj = weight_norm(nn.Conv1d(code_dim, latent_dim, kernel_size=1))

        codebook = torch.randn(codebook_size, code_dim)
        if l2_normalize:
            codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_embedding_sum", codebook.clone())

    def lookup(self, flat):
        """flat: (N, code_dim) -> nearest codebook index per row."""
        if self.l2_normalize:
            # Both sides unit-norm, so ||a-b||^2 = 2 - 2 a.b: maximizing the
            # dot product is exactly minimizing the distance.
            scores = F.normalize(flat, dim=-1) @ F.normalize(self.codebook, dim=-1).t()
            return scores.argmax(1)
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            + self.codebook.pow(2).sum(1)
            - 2.0 * flat @ self.codebook.t()
        )
        return dist.argmin(1)

    @torch.no_grad()
    def ema_update(self, flat, indices):
        encodings = F.one_hot(indices, self.codebook_size).to(flat.dtype)
        cluster_size = encodings.sum(0)
        embedding_sum = encodings.t() @ flat

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1.0 - self.ema_decay
        )
        self.ema_embedding_sum.mul_(self.ema_decay).add_(
            embedding_sum, alpha=1.0 - self.ema_decay
        )
        # Sonnet-style Laplace smoothing of the cluster sizes, so an entry
        # that briefly wins nothing does not divide by ~0.
        n = self.ema_cluster_size.sum()
        smoothed = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.codebook_size * self.epsilon)
            * n
        )
        updated = self.ema_embedding_sum / smoothed.unsqueeze(1)
        if self.l2_normalize:
            updated = F.normalize(updated, dim=-1)
        self.codebook.copy_(updated)

    @torch.no_grad()
    def revive_dead_codes(self, flat, fraction):
        """Resample entries nobody is using from the batch's own encoder
        outputs. Returns how many were revived - a number worth watching in
        the training log.

        "Nobody is using" is deliberately *relative*: the cutoff is
        `fraction` of the usage a perfectly uniform codebook would show,
        which is `ema_cluster_size.sum() / codebook_size` (the sum is an
        EMA of the constant batch element count, so this tracks batch size
        and crop length automatically).

        The absolute threshold of 2.0 that EnCodec and
        vector-quantize-pytorch use is a trap at this scale, and the smoke
        run walked straight into it: one batch here is 32 x 69 = 2,208
        vectors over 1,024 entries, so *uniform* usage is only 2.16 per
        entry. An absolute cutoff of 2.0 therefore condemns roughly half a
        perfectly healthy codebook every sweep, permanently churning it.
        A genuinely dead entry decays toward zero (x0.99 per step), so it
        falls below a few percent of uniform within a few hundred steps and
        is caught by this cutoff without any false positives.
        """
        expected = self.ema_cluster_size.sum() / self.codebook_size
        threshold = fraction * expected
        dead = self.ema_cluster_size < threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        replacement = flat[pick]
        if self.l2_normalize:
            replacement = F.normalize(replacement, dim=-1)
        self.codebook[dead] = replacement
        self.ema_embedding_sum[dead] = replacement
        # Reset usage to uniform-expectation so a freshly revived code is
        # given a real window to earn assignments before being culled again.
        self.ema_cluster_size[dead] = expected
        return n_dead

    def forward(self, residual, ema_enabled):
        """residual: (B, latent_dim, T).

        Returns the quantized contribution (same shape, straight-through),
        per-sample commitment and codebook losses, the code indices, and
        this batch's one-hot usage histogram.
        """
        b, _, t = residual.shape
        z_e = self.in_proj(residual)                       # (B, code_dim, T)
        flat = z_e.permute(0, 2, 1).reshape(-1, self.code_dim)

        indices = self.lookup(flat)
        z_q_flat = F.embedding(indices, self.codebook)
        z_q = z_q_flat.view(b, t, self.code_dim).permute(0, 2, 1)

        if ema_enabled and self.training:
            self.ema_update(flat.detach(), indices)

        # Per-sample so quantizer dropout can mask them; mean over the code
        # dimension and time, matching the usual per-element MSE scale.
        commitment = (z_e - z_q.detach()).pow(2).mean(dim=(1, 2))
        # Only used in --vq-mode loss; with EMA the codebook is a moving
        # average and carries no gradient.
        codebook_loss = (z_q - z_e.detach()).pow(2).mean(dim=(1, 2))

        # Straight-through: the decoder sees the quantized value, the
        # encoder receives the gradient as if nothing had been quantized.
        z_q = z_e + (z_q - z_e).detach()
        return (
            self.out_proj(z_q),
            commitment,
            codebook_loss,
            indices.view(b, t),
            flat.detach(),
        )


class ResidualVectorQuantizer(nn.Module):
    """A stack of VectorQuantizers, each quantizing the previous one's
    residual. See the module docstring for why this beats one big codebook.
    """

    def __init__(self, latent_dim, code_dim, codebook_size, num_quantizers,
                 ema_decay=0.99, l2_normalize=True):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList([
            VectorQuantizer(latent_dim, code_dim, codebook_size, ema_decay,
                            l2_normalize)
            for _ in range(num_quantizers)
        ])

    def forward(self, z, n_quantizers=None, ema_enabled=True):
        """n_quantizers: either None (use all), an int (use the first n), or
        a per-sample (B,) tensor (quantizer dropout)."""
        b = z.shape[0]
        if n_quantizers is None:
            n_active = torch.full((b,), self.num_quantizers, device=z.device,
                                  dtype=torch.float32)
        elif torch.is_tensor(n_quantizers):
            n_active = n_quantizers.to(z.device).float()
        else:
            n_active = torch.full((b,), float(n_quantizers), device=z.device)

        quantized = torch.zeros_like(z)
        residual = z
        commitment = z.new_zeros(())
        codebook_loss = z.new_zeros(())
        indices_per_q = []
        usage_per_q = []

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= int(n_active.max().item()):
                break
            q_out, commit_i, cb_i, idx_i, _ = quantizer(residual, ema_enabled)

            # 1.0 for samples whose n_q reaches this codebook, else 0.0.
            mask = (n_active > i).to(z.dtype)
            quantized = quantized + q_out * mask.view(-1, 1, 1)
            # The residual is advanced unconditionally: a masked-out sample
            # still produces a meaningful residual for the deeper codebooks
            # to learn from, it just does not reach the decoder.
            residual = residual - q_out

            commitment = commitment + (commit_i * mask).mean()
            codebook_loss = codebook_loss + (cb_i * mask).mean()
            indices_per_q.append(idx_i)
            usage_per_q.append(
                torch.bincount(idx_i.reshape(-1), minlength=self.codebook_size)
            )

        return quantized, commitment, codebook_loss, indices_per_q, usage_per_q

    @torch.no_grad()
    def revive_dead_codes(self, z, fraction):
        """Re-run the stack purely to collect each codebook's own input
        distribution - quantizer q sees the residual left by q-1, so each
        one has to be resampled from its own inputs, not from z."""
        revived = []
        residual = z
        for quantizer in self.quantizers:
            q_out, _, _, _, flat = quantizer(residual, ema_enabled=False)
            revived.append(quantizer.revive_dead_codes(flat, fraction))
            residual = residual - q_out
        return revived


class AudioCodec(nn.Module):
    def __init__(self, base_channels=32, latent_dim=128, code_dim=8,
                 codebook_size=1024, num_quantizers=8, strides=DEFAULT_STRIDES,
                 ema_decay=0.99, l2_normalize=True):
        super().__init__()
        self.strides = tuple(strides)
        self.hop = int(np.prod(self.strides))
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.encoder = Encoder(base_channels, latent_dim, self.strides)
        self.quantizer = ResidualVectorQuantizer(
            latent_dim, code_dim, codebook_size, num_quantizers, ema_decay,
            l2_normalize,
        )
        self.decoder = Decoder(base_channels, latent_dim, self.strides)

    def forward(self, x, n_quantizers=None, ema_enabled=True):
        z = self.encoder(x)
        q, commitment, codebook_loss, indices, usage = self.quantizer(
            z, n_quantizers, ema_enabled
        )
        return self.decoder(q), commitment, codebook_loss, indices, usage, z

    def reconstruct(self, x, n_quantizers=None):
        """Deterministic encode -> quantize -> decode, for evaluation."""
        z = self.encoder(x)
        q, _, _, _, _ = self.quantizer(z, n_quantizers, ema_enabled=False)
        return self.decoder(q)


# ---------------------------------------------------------------------------
# Losses: multi-scale mel + multi-scale STFT discriminator
# ---------------------------------------------------------------------------

def hz_to_mel(hz):
    """HTK mel scale. Written out rather than imported - it is one line,
    and importing librosa for it would pull a whole DSP stack in."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_mels, n_fft, sample_rate, f_min=0.0, f_max=None):
    """Triangular mel filterbank, (n_mels, n_fft//2 + 1).

    Each filter rises linearly from its lower neighbour's centre to its own
    and falls to its upper neighbour's, with the centres placed evenly on
    the mel scale - which is why low frequencies get narrow filters and
    high ones get wide.
    """
    f_max = sample_rate / 2.0 if f_max is None else f_max
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])

    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lower, centre, upper = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        if centre > lower:
            rising = (freqs - lower) / (centre - lower)
            filters[i] += np.clip(rising, 0.0, None) * (freqs <= centre)
        if upper > centre:
            falling = (upper - freqs) / (upper - centre)
            filters[i] += np.clip(falling, 0.0, None) * (freqs > centre)
    return torch.from_numpy(np.clip(filters, 0.0, None))


class MultiScaleMelLoss(nn.Module):
    """L1 between the mel spectrograms of target and reconstruction, at
    seven window sizes at once.

    Both a log-magnitude and a linear-magnitude term are used: the log term
    is where quiet detail lives (and is what a listener notices), the
    linear term keeps loud partials from being ignored.
    """

    def __init__(self, sample_rate, windows=MEL_WINDOWS, bands=MEL_BANDS):
        super().__init__()
        self.windows = tuple(windows)
        for w, n_mels in zip(windows, bands):
            self.register_buffer(
                f"fb_{w}", mel_filterbank(n_mels, w, sample_rate), persistent=False
            )
            self.register_buffer(
                f"win_{w}", torch.hann_window(w), persistent=False
            )

    def mel(self, x, window):
        """x: (B, L) -> (B, n_mels, frames)."""
        spec = torch.stft(
            x,
            n_fft=window,
            hop_length=window // 4,
            win_length=window,
            window=getattr(self, f"win_{window}"),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = spec.abs()
        return getattr(self, f"fb_{window}") @ magnitude

    def forward(self, x_hat, x):
        loss = x.new_zeros(())
        for window in self.windows:
            mel_hat = self.mel(x_hat, window)
            mel_ref = self.mel(x, window)
            loss = loss + (mel_hat - mel_ref).abs().mean()
            loss = loss + (
                mel_hat.clamp_min(1e-5).log10() - mel_ref.clamp_min(1e-5).log10()
            ).abs().mean()
        return loss


class STFTDiscriminator(nn.Module):
    """One resolution of the discriminator: a small 2D conv stack over the
    complex spectrogram, with real and imaginary parts as two input
    channels. Feeding it the complex STFT rather than the magnitude is the
    point - phase incoherence is exactly what makes a magnitude-trained
    codec sound metallic, and a magnitude-only critic cannot hear it."""

    def __init__(self, n_fft, hop_length, filters=32, max_filters=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        convs = [weight_norm(nn.Conv2d(2, filters, (3, 9), padding=(1, 4)))]
        in_channels = filters
        for i, dilation in enumerate((1, 2, 4)):
            out_channels = min(filters * (i + 2), max_filters)
            convs.append(weight_norm(nn.Conv2d(
                in_channels, out_channels, (3, 9), stride=(1, 2),
                dilation=(1, dilation), padding=(1, 4 * dilation),
            )))
            in_channels = out_channels
        convs.append(weight_norm(
            nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1))
        ))
        self.convs = nn.ModuleList(convs)
        self.conv_post = weight_norm(
            nn.Conv2d(in_channels, 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x):
        """x: (B, L) -> (logits, [feature maps])."""
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        # (B, freq, time) complex -> (B, 2, time, freq): the conv stack
        # strides along the frequency axis and keeps time resolution.
        h = torch.view_as_real(spec).permute(0, 3, 2, 1)

        features = []
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.1)
            features.append(h)
        return self.conv_post(h), features


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, resolutions=DISC_RESOLUTIONS, filters=32):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [STFTDiscriminator(n_fft, hop, filters=filters)
             for n_fft, hop in resolutions]
        )

    def forward(self, x):
        logits, features = [], []
        for disc in self.discriminators:
            logit, feats = disc(x)
            logits.append(logit)
            features.append(feats)
        return logits, features


def discriminator_hinge_loss(real_logits, fake_logits):
    """Hinge loss: push real above +1 and fake below -1, and stop caring
    once they are there. Gentler than BCE, which keeps pushing an already
    confident discriminator and starves the generator of gradient."""
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


def generator_hinge_loss(fake_logits):
    return sum(F.relu(1.0 - f).mean() for f in fake_logits) / len(fake_logits)


def feature_matching_loss(real_features, fake_features):
    """L1 between the discriminator's intermediate activations on real and
    generated audio. In practice this carries most of the adversarial
    signal - it tells the generator *how* it differs, not just that it
    was caught."""
    loss, count = 0.0, 0
    for real_scale, fake_scale in zip(real_features, fake_features):
        for real, fake in zip(real_scale, fake_scale):
            loss = loss + (fake - real.detach()).abs().mean()
            count += 1
    return loss / count


# ---------------------------------------------------------------------------
# EMA of the generator weights (same class as training/flow-matching-mnist)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of the weights, kept alongside the live
    model and used for evaluation. GAN training never converges to a point
    - the generator orbits - so the averaged weights are reliably better
    than whatever the last step happened to land on."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return self.shadow


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_corpus(data_dir):
    """Open the memmapped int16 corpus written by build_ljspeech_dataset.py."""
    index = np.load(os.path.join(data_dir, "ljspeech_index.npz"))
    audio = np.memmap(
        os.path.join(data_dir, "ljspeech_audio.i16"), dtype="<i2", mode="r"
    )
    return audio, index


def eligible_indices(index, split_value, crop_samples):
    """Utterances in this split that are at least one crop long. Short ones
    are dropped rather than zero-padded: padding would teach the codec to
    reconstruct silence it will never see at inference."""
    mask = (index["split"] == split_value) & (index["lengths"] >= crop_samples)
    return np.nonzero(mask)[0]


def gather_crops(audio, offsets, lengths, batch_idx, crop_samples, rng):
    """One random crop per utterance, int16 -> float32 in [-1, 1).

    Plain numpy indexing into the memmap; no torch DataLoader, no worker
    processes - the batching stays visible, the same choice every other
    pipeline in training/ makes.
    """
    out = np.empty((len(batch_idx), crop_samples), dtype=np.float32)
    for j, i in enumerate(batch_idx):
        start = int(offsets[i]) + int(rng.integers(0, lengths[i] - crop_samples + 1))
        out[j] = audio[start:start + crop_samples]
    return out / 32768.0


def iterate_batches(indices, batch_size, rng, shuffle, drop_last):
    order = rng.permutation(indices) if shuffle else indices
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        if drop_last and len(batch) < batch_size:
            return
        yield batch


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def sample_n_quantizers(batch_size, num_quantizers, dropout_fraction, device):
    """Quantizer dropout: a `dropout_fraction` slice of the batch gets a
    random n_q in [1, N], the rest keeps all N.

    This is what makes one trained model serve every bitrate on the ladder.
    Without it the model only ever sees all N codebooks and its output with
    the last four dropped is unusable - you would have to train a separate
    model per bitrate."""
    n_active = torch.full((batch_size,), float(num_quantizers), device=device)
    if dropout_fraction > 0:
        n_dropped = int(batch_size * dropout_fraction)
        if n_dropped > 0:
            n_active[:n_dropped] = torch.randint(
                1, num_quantizers + 1, (n_dropped,), device=device
            ).float()
    return n_active


def discriminator_precision(enabled, device):
    """bf16 autocast for the discriminator only.

    The multi-scale STFT discriminator is by far the most expensive part of
    a step - its spectrograms are much larger than the waveform it judges
    (173 x 257 positions at the 512-point resolution, against 22,080
    samples), and the odd conv shapes map badly onto fp32 tensor cores.
    Measured on an RTX 3090 at batch 32: 0.95 -> 2.01 steps/s and
    16.8 -> 10.8 GiB peak, which is the difference between a 7-hour run and
    a 3.3-hour one.

    Only the critic runs reduced-precision. The generator, the codebook
    lookup and every EMA update stay fp32 - bf16 EMA statistics would
    quietly stop accumulating small updates, which is exactly the mechanism
    dead-code revival exists to detect. bf16 rather than fp16 because it
    keeps fp32's exponent range, so no gradient scaler is needed and the
    hinge loss cannot overflow.

    torch.stft has no autocast policy, so it keeps running in fp32 and only
    the conv stack is cast.
    """
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def codebook_stats(usage_per_q):
    """Per-codebook perplexity and the count of entries this batch never
    touched. Perplexity near 1 means collapsed; near codebook_size means
    every entry is pulling its weight."""
    stats = []
    for usage in usage_per_q:
        p = usage.float() / usage.sum().clamp_min(1)
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        stats.append((float(perplexity), int((usage == 0).sum())))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/ljspeech_codec")

    parser.add_argument("--base-channels", type=int, default=32,
                        help="Encoder stem width; channels double at every stride.")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--code-dim", type=int, default=8,
                        help="Factorized lookup dimension. Nearest-neighbour search "
                             "happens here, not in the full latent width - low-dim "
                             "codes are much harder to kill.")
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--num-quantizers", type=int, default=8,
                        help="Codebooks in the RVQ stack. Bitrate = frame_rate * "
                             "this * log2(codebook_size).")
    parser.add_argument("--vq-mode", choices=("ema", "loss"), default="ema",
                        help="ema: codebook entries are exponential moving averages "
                             "of the vectors that chose them (no gradient). loss: the "
                             "classic VQ-VAE codebook gradient term instead.")
    parser.add_argument("--vq-ema-decay", type=float, default=0.99)
    parser.add_argument("--vq-l2-normalize", type=int, default=1,
                        help="1 = assign by cosine distance on unit-norm vectors.")
    parser.add_argument("--dead-code-threshold", type=float, default=0.05,
                        help="Dead-code cutoff as a FRACTION of uniform codebook "
                             "usage (0.05 = an entry seeing under 5%% of its fair "
                             "share is resampled from the live batch). Relative "
                             "rather than absolute on purpose - see "
                             "VectorQuantizer.revive_dead_codes. 0 disables revival.")
    parser.add_argument("--dead-code-check-every", type=int, default=100,
                        help="Steps between dead-code sweeps.")
    parser.add_argument("--quantizer-dropout", type=float, default=0.5,
                        help="Fraction of each batch trained at a random n_q in "
                             "[1, N], so one model covers the whole bitrate ladder.")

    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-mel", type=float, default=15.0)
    parser.add_argument("--lambda-commit", type=float, default=0.25)
    parser.add_argument("--lambda-codebook", type=float, default=1.0,
                        help="Only used with --vq-mode loss.")
    parser.add_argument("--lambda-adv", type=float, default=1.0,
                        help="0 disables the discriminator entirely (pure "
                             "reconstruction run).")
    parser.add_argument("--lambda-feat", type=float, default=2.0)
    parser.add_argument("--disc-filters", type=int, default=32,
                        help="Discriminator stem width. 32 is EnCodec's. 16 is ~1.7x "
                             "faster again at a quarter of the discriminator's "
                             "parameters - a real capacity cut, so prefer --disc-bf16 "
                             "first.")
    parser.add_argument("--disc-bf16", type=int, default=1,
                        help="Run the discriminator (only) under bf16 autocast. The "
                             "generator, the quantizer and every EMA update stay "
                             "fp32. Measured on an RTX 3090: 0.95 -> 2.01 steps/s and "
                             "16.8 -> 10.8 GiB peak, with no architecture change. "
                             "0 forces fp32 everywhere.")
    parser.add_argument("--adv-start-step", type=int, default=5000,
                        help="Reconstruction-only warmup length. The discriminator "
                             "is neither trained nor applied before this step. 5000 "
                             "steps is ~12 epochs at the default batch size - long "
                             "enough for the reconstruction to be sane, short enough "
                             "to leave most of a 60-epoch run adversarial.")

    parser.add_argument("--crop-frames", type=int, default=69,
                        help="Training crop length in codec frames; the sample "
                             "count is this times the hop (320), so crops are always "
                             "a whole number of frames. 69 frames ~= 1.0 s.")
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.999996,
                        help="Per-step exponential decay on both optimizers.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay for the evaluation weights; 0 disables it.")
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    audio, index = load_corpus(args.data_dir)
    offsets, lengths = index["offsets"], index["lengths"]
    sample_rate = int(index["sample_rate"])

    strides = DEFAULT_STRIDES
    hop = int(np.prod(strides))
    crop_samples = args.crop_frames * hop
    frame_rate = sample_rate / hop
    bits_per_frame = args.num_quantizers * math.log2(args.codebook_size)

    train_idx = eligible_indices(index, 0, crop_samples)
    val_idx = eligible_indices(index, 1, crop_samples)
    print(f"Corpus: {sample_rate} Hz, hop {hop} -> {frame_rate:.1f} frames/s")
    print(f"Bitrate at {args.num_quantizers} x {args.codebook_size}: "
          f"{frame_rate * bits_per_frame / 1000.0:.2f} kbps")
    print(f"Train utterances: {len(train_idx)}  Val utterances: {len(val_idx)}  "
          f"(crop {crop_samples} samples = {crop_samples / sample_rate:.3f} s)")

    model = AudioCodec(
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        code_dim=args.code_dim,
        codebook_size=args.codebook_size,
        num_quantizers=args.num_quantizers,
        strides=strides,
        ema_decay=args.vq_ema_decay,
        l2_normalize=bool(args.vq_l2_normalize),
    ).to(device)
    discriminator = MultiScaleSTFTDiscriminator(filters=args.disc_filters).to(device)
    mel_loss_fn = MultiScaleMelLoss(sample_rate).to(device)

    n_gen = sum(p.numel() for p in model.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Codec parameters: {n_gen:,}  (discriminator: {n_disc:,}, "
          f"training-only, {'bf16' if args.disc_bf16 else 'fp32'})")

    opt_g = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                              betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.learning_rate,
                              betas=(0.8, 0.99))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=args.lr_decay)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=args.lr_decay)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    use_ema_codebook = args.vq_mode == "ema"
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "codec_best.pt")
    final_path = os.path.join(args.output_dir, "codec_final.pt")

    def checkpoint(epoch, val_mel):
        return {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "discriminator_state": discriminator.state_dict(),
            "base_channels": args.base_channels,
            "latent_dim": args.latent_dim,
            "code_dim": args.code_dim,
            "codebook_size": args.codebook_size,
            "num_quantizers": args.num_quantizers,
            "strides": list(strides),
            "sample_rate": sample_rate,
            "vq_mode": args.vq_mode,
            "vq_l2_normalize": bool(args.vq_l2_normalize),
            "vq_ema_decay": args.vq_ema_decay,
            "disc_filters": args.disc_filters,
            # Stored so evaluate_codec.py can tell whether the EMA weights
            # have actually converged: after n steps they still carry
            # ema_decay**n of the random initialization.
            "ema_decay": args.ema_decay,
            "global_step": global_step,
            "epoch": epoch,
            "val_mel": val_mel,
        }

    global_step = 0
    best_val_mel = float("inf")
    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        discriminator.train()
        sums = {"time": 0.0, "mel": 0.0, "commit": 0.0, "adv": 0.0, "feat": 0.0,
                "disc": 0.0}
        n_batches = 0
        revived_total = 0
        last_usage = None

        for batch_idx in iterate_batches(train_idx, args.batch_size, rng,
                                         shuffle=True, drop_last=True):
            x = torch.from_numpy(
                gather_crops(audio, offsets, lengths, batch_idx, crop_samples, rng)
            ).to(device).unsqueeze(1)                       # (B, 1, L)

            n_active = sample_n_quantizers(
                x.shape[0], args.num_quantizers, args.quantizer_dropout, device
            )
            x_hat, commitment, codebook_loss, _, usage, z = model(
                x, n_quantizers=n_active, ema_enabled=use_ema_codebook
            )
            adversarial_on = args.lambda_adv > 0 and global_step >= args.adv_start_step

            # ---- discriminator step (on the detached reconstruction) ----
            disc_loss_value = 0.0
            if adversarial_on:
                with discriminator_precision(args.disc_bf16, device):
                    real_logits, _ = discriminator(x.squeeze(1))
                    fake_logits, _ = discriminator(x_hat.detach().squeeze(1))
                    d_loss = discriminator_hinge_loss(real_logits, fake_logits)
                d_loss = d_loss.float()
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                               args.max_grad_norm)
                opt_d.step()
                sched_d.step()
                disc_loss_value = d_loss.item()

            # ---- generator step ----
            time_loss = (x_hat - x).abs().mean()
            mel_loss = mel_loss_fn(x_hat.squeeze(1), x.squeeze(1))
            g_loss = args.lambda_time * time_loss + args.lambda_mel * mel_loss
            g_loss = g_loss + args.lambda_commit * commitment
            if args.vq_mode == "loss":
                g_loss = g_loss + args.lambda_codebook * codebook_loss

            adv_value = feat_value = 0.0
            if adversarial_on:
                # The real branch is a constant target for feature matching -
                # no gradient ever flows back through it - so running it under
                # no_grad drops its activation graph. That matters: the
                # discriminator's spectrograms are much larger than the
                # waveform, and keeping this graph pushed peak memory to the
                # edge of the 3090's 24 GB at batch 32.
                with discriminator_precision(args.disc_bf16, device):
                    with torch.no_grad():
                        _, real_features = discriminator(x.squeeze(1))
                    fake_logits, fake_features = discriminator(x_hat.squeeze(1))
                    adv_loss = generator_hinge_loss(fake_logits)
                    feat_loss = feature_matching_loss(real_features, fake_features)
                # Back to fp32 before joining the reconstruction terms, so the
                # generator's own graph never sees a reduced-precision loss.
                adv_loss, feat_loss = adv_loss.float(), feat_loss.float()
                g_loss = g_loss + args.lambda_adv * adv_loss + args.lambda_feat * feat_loss
                adv_value, feat_value = adv_loss.item(), feat_loss.item()

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt_g.step()
            sched_g.step()
            if ema is not None:
                ema.update(model)

            # ---- dead-code revival ----
            if (args.dead_code_threshold > 0 and use_ema_codebook
                    and global_step % args.dead_code_check_every == 0):
                revived_total += sum(
                    model.quantizer.revive_dead_codes(
                        z.detach(), args.dead_code_threshold
                    )
                )

            sums["time"] += time_loss.item()
            sums["mel"] += mel_loss.item()
            sums["commit"] += commitment.item()
            sums["adv"] += adv_value
            sums["feat"] += feat_value
            sums["disc"] += disc_loss_value
            n_batches += 1
            global_step += 1
            last_usage = usage

        # ---- validation: mel distance at full rate, fixed crops ----
        model.eval()
        val_rng = np.random.default_rng(args.seed)
        val_mel_total, val_time_total, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for i, batch_idx in enumerate(iterate_batches(
                    val_idx, args.batch_size, val_rng, shuffle=False, drop_last=False)):
                if i >= args.val_batches:
                    break
                x = torch.from_numpy(
                    gather_crops(audio, offsets, lengths, batch_idx, crop_samples,
                                 val_rng)
                ).to(device).unsqueeze(1)
                x_hat = model.reconstruct(x)
                val_mel_total += mel_loss_fn(x_hat.squeeze(1), x.squeeze(1)).item()
                val_time_total += (x_hat - x).abs().mean().item()
                val_n += 1
        val_mel = val_mel_total / max(val_n, 1)
        val_time = val_time_total / max(val_n, 1)

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            stats = codebook_stats(last_usage)
            perplexities = " ".join(f"{p:.0f}" for p, _ in stats)
            dead = " ".join(f"{d}" for _, d in stats)
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  step {global_step}  "
                f"time={sums['time'] / n_batches:.4f} "
                f"mel={sums['mel'] / n_batches:.3f} "
                f"commit={sums['commit'] / n_batches:.4f} "
                f"adv={sums['adv'] / n_batches:.3f} "
                f"feat={sums['feat'] / n_batches:.3f} "
                f"d={sums['disc'] / n_batches:.3f} | "
                f"val: mel={val_mel:.3f} time={val_time:.4f} "
                f"[{time.time() - start_time:7.1f}s]"
            )
            print(f"    codebook perplexity per q: {perplexities}")
            print(f"    unused entries per q (of {args.codebook_size}): {dead}"
                  f"   revived this epoch: {revived_total}")

        if val_mel < best_val_mel:
            best_val_mel = val_mel
            torch.save(checkpoint(epoch, val_mel), best_path)

    torch.save(checkpoint(args.num_epochs, val_mel), final_path)
    print(f"\nTrained {args.num_epochs} epochs "
          f"({global_step} steps) in {time.time() - start_time:.1f}s")
    if device.type == "cuda":
        # Worth printing: the discriminator's spectrograms are far larger
        # than the waveform, so this run sits much closer to the card's
        # limit than the 7.3M parameter count suggests.
        print(f"Peak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"Saved best checkpoint (val mel={best_val_mel:.3f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_codec.py next - and listen to the wavs it writes, "
          "the numbers alone will not tell you how it sounds.")


if __name__ == "__main__":
    main()

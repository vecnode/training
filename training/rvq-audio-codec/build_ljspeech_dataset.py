"""
Turn the raw LJSpeech-1.1 wav files into a memory-mappable int16 corpus
plus an index, ready for neural-codec training.

LJSpeech is 13,100 single-speaker (Linda Johnson) LibriVox readings, 22,050
Hz / 16-bit / mono PCM, ~23.9 hours in total. The transcripts in
metadata.csv are deliberately unused: a codec is trained by
self-supervised reconstruction and never sees text.

Two things are done by hand here that a library would normally hide:

  1. The RIFF/WAVE parser. Every .wav is walked chunk by chunk with the
     stdlib struct module - no soundfile, no torchaudio, no librosa, not
     even the stdlib wave module. LJSpeech's files happen to be the simple
     44-byte-header case (fmt then data, nothing else), but the parser
     walks chunks properly anyway so a corpus carrying a LIST/INFO chunk
     still loads. The format fields are asserted rather than assumed:
     PCM (format 1), mono, 16-bit, and one consistent sample rate across
     the whole corpus.

  2. The storage layout. 3.8 GB of int16 becomes 7.6 GB as float32, which
     is more than is comfortable to hold in RAM, so this does *not* write
     an .npz the way the MNIST/CIFAR/IMDB builders in the sibling
     pipelines do. Instead every utterance is concatenated into one raw
     little-endian int16 file that train_codec.py opens with np.memmap,
     next to a small .npz index of offsets/lengths/ids/split. Crops are
     converted to float on the fly, one batch at a time.

The utterance count is verified (exactly 13,100) so a partial or
in-progress extraction fails loudly here instead of silently training on
half a corpus - the same guardrail as build_imdb_dataset.py's 12,500-file
check in training/imdb-sentiment-cnn.

Usage:
    uv run --directory training/rvq-audio-codec python build_ljspeech_dataset.py \
        --data-dir "C:\\path\\to\\LJSpeech-1.1" \
        --output-dir data
"""

import argparse
import os
import struct

import numpy as np

EXPECTED_WAV_COUNT = 13100


def resolve_wav_dir(data_dir):
    """The LJSpeech-1.1.tar.bz2 extracts to an LJSpeech-1.1/ subfolder
    containing wavs/. Accept either the folder that holds wavs/ directly,
    or one wrapping an LJSpeech-1.1 subfolder that does - the same
    tolerance build_imdb_dataset.py has for a nested aclImdb/."""
    direct = os.path.join(data_dir, "wavs")
    if os.path.isdir(direct):
        return direct
    nested = os.path.join(data_dir, "LJSpeech-1.1", "wavs")
    if os.path.isdir(nested):
        return nested
    raise FileNotFoundError(
        f"Could not find a wavs/ folder under {data_dir} (looked directly "
        f"and under an LJSpeech-1.1 subfolder)"
    )


def read_wav_pcm16(path):
    """Parse a RIFF/WAVE file and return (int16 samples, sample_rate).

    Layout: b"RIFF" + uint32 size + b"WAVE", then a sequence of chunks,
    each b"<id>" + uint32 size + <size bytes> (odd sizes are padded to
    even). Only the "fmt " and "data" chunks are needed; anything else
    (LIST, INFO, ...) is skipped over.
    """
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")

        sample_rate = None
        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                raise ValueError(f"{path}: ran out of chunks before a data chunk")
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)

            if chunk_id == b"fmt ":
                fmt = f.read(chunk_size)
                audio_format, channels, sample_rate, _byte_rate, _align, bits = \
                    struct.unpack("<HHIIHH", fmt[:16])
                if audio_format != 1:
                    raise ValueError(
                        f"{path}: audio format {audio_format} is not PCM (1)"
                    )
                if channels != 1:
                    raise ValueError(f"{path}: {channels} channels, expected mono")
                if bits != 16:
                    raise ValueError(f"{path}: {bits}-bit, expected 16-bit")
            elif chunk_id == b"data":
                if sample_rate is None:
                    raise ValueError(f"{path}: data chunk before fmt chunk")
                raw = f.read(chunk_size)
                if len(raw) < chunk_size:
                    raise ValueError(
                        f"{path}: data chunk claims {chunk_size} bytes, got {len(raw)} "
                        f"- truncated file"
                    )
                samples = np.frombuffer(raw, dtype="<i2")
                return samples, sample_rate
            else:
                f.seek(chunk_size + (chunk_size % 2), os.SEEK_CUR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=r"C:\path\to\LJSpeech-1.1",
        help="Directory containing the extracted LJSpeech wavs/ folder (an "
             "LJSpeech-1.1 subfolder is also accepted).",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--expected-wavs", type=int, default=EXPECTED_WAV_COUNT,
        help="Refuse to build unless exactly this many .wav files are present "
             "(guards against a partial extraction). Set to 0 to skip the "
             "check when pointing this at a different corpus.",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.02,
        help="Fraction of utterances held out for validation. LJSpeech is a "
             "single speaker, so this is a held-out-utterance split, not a "
             "held-out-speaker one - see the README.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wav_dir = resolve_wav_dir(args.data_dir)
    print(f"Using wavs at {wav_dir}")

    wav_names = sorted(n for n in os.listdir(wav_dir) if n.lower().endswith(".wav"))
    if args.expected_wavs and len(wav_names) != args.expected_wavs:
        raise RuntimeError(
            f"Expected {args.expected_wavs} .wav files under {wav_dir} but found "
            f"{len(wav_names)}. A partial or in-progress extraction would train "
            f"on a truncated corpus - re-extract LJSpeech-1.1.tar.bz2 and rerun "
            f"(or pass --expected-wavs 0 if this is deliberately a different "
            f"corpus)."
        )
    print(f"Found {len(wav_names)} wav files")

    os.makedirs(args.output_dir, exist_ok=True)
    audio_path = os.path.join(args.output_dir, "ljspeech_audio.i16")
    index_path = os.path.join(args.output_dir, "ljspeech_index.npz")

    offsets = np.zeros(len(wav_names), dtype=np.int64)
    lengths = np.zeros(len(wav_names), dtype=np.int64)
    utterance_ids = []
    sample_rate = None
    cursor = 0

    # Stream straight to disk: the whole corpus is never in memory at once.
    with open(audio_path, "wb") as out:
        for i, name in enumerate(wav_names):
            samples, rate = read_wav_pcm16(os.path.join(wav_dir, name))
            if sample_rate is None:
                sample_rate = rate
            elif rate != sample_rate:
                raise ValueError(
                    f"{name}: {rate} Hz, but earlier files are {sample_rate} Hz. "
                    f"This pipeline trains at one native rate and writes no "
                    f"resampler - see the README."
                )
            out.write(samples.tobytes())
            offsets[i] = cursor
            lengths[i] = samples.shape[0]
            cursor += samples.shape[0]
            utterance_ids.append(os.path.splitext(name)[0])

            if (i + 1) % 2000 == 0 or i + 1 == len(wav_names):
                print(f"  read {i + 1}/{len(wav_names)} files "
                      f"({cursor / sample_rate / 3600.0:.2f} h so far)")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(wav_names))
    n_val = int(len(wav_names) * args.val_fraction)
    split = np.zeros(len(wav_names), dtype=np.int64)   # 0 = train, 1 = val
    split[perm[:n_val]] = 1

    np.savez(
        index_path,
        offsets=offsets,
        lengths=lengths,
        split=split,
        utterance_ids=np.array(utterance_ids),
        sample_rate=np.int64(sample_rate),
    )

    total_hours = cursor / sample_rate / 3600.0
    print(f"\n{len(wav_names)} utterances, {cursor:,} samples, "
          f"{total_hours:.2f} h at {sample_rate} Hz")
    print(f"Split: {int((split == 0).sum())} train / {int((split == 1).sum())} val "
          f"(held-out utterances of the same single speaker)")
    print(f"Shortest utterance: {lengths.min() / sample_rate:.2f} s  "
          f"Longest: {lengths.max() / sample_rate:.2f} s  "
          f"Mean: {lengths.mean() / sample_rate:.2f} s")
    print(f"Saved {audio_path} ({cursor * 2 / 1e9:.2f} GB int16)")
    print(f"Saved {index_path}")


if __name__ == "__main__":
    main()

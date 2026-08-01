"""Resize PNGs to max dimension and compress to max file size."""
from __future__ import annotations

import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

MAX_BYTES_DEFAULT = 10 * 1024 * 1024
MAX_DIM_DEFAULT = 1080


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=6)
    return buf.getvalue()


def _fit_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    scale = max_dim / longest
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.Resampling.BILINEAR)


def _best_quantized(base: Image.Image, max_colors: int, max_bytes: int) -> bytes | None:
    best = None
    lo, hi = 2, max_colors
    while lo <= hi:
        mid = (lo + hi) // 2
        q = base.quantize(
            colors=mid,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        data = _encode(q)
        if len(data) <= max_bytes:
            best = data
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def process_png(
    path: Path,
    max_bytes: int = MAX_BYTES_DEFAULT,
    max_dim: int = MAX_DIM_DEFAULT,
) -> tuple[bool, int, int, str]:
    original = path.stat().st_size
    with Image.open(path) as img:
        orig_w, orig_h = img.size
        rgb = _fit_max_dim(img.convert("RGB"), max_dim)
        data = _encode(rgb)
        if len(data) > max_bytes:
            q = _best_quantized(rgb, 256, max_bytes)
            if q:
                data = q
            else:
                gray = _fit_max_dim(rgb.convert("L"), max_dim)
                q = _best_quantized(gray, 256, max_bytes)
                if not q:
                    raise RuntimeError(f"Could not compress under {max_bytes} bytes: {path}")
                data = q
        if len(data) >= original and rgb.size == (orig_w, orig_h):
            return False, original, original, path.name
        path.write_bytes(data)
        return True, original, len(data), path.name


def _worker(args: tuple[str, float, int]) -> tuple[bool, str, float, float]:
    path_str, max_mb, max_dim = args
    max_bytes = int(max_mb * 1024 * 1024)
    did, before, after, name = process_png(Path(path_str), max_bytes, max_dim)
    return did, name, before / 1024 / 1024, after / 1024 / 1024


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[])
    parser.add_argument(
        "--list-file",
        type=Path,
        default=None,
        help="Text file with one PNG/directory path per line (avoids OS command-line length limits for large batches)",
    )
    parser.add_argument("--max-mb", type=float, default=10.0)
    parser.add_argument("--max-dim", type=int, default=MAX_DIM_DEFAULT)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    args = parser.parse_args()

    raw_paths = list(args.paths)
    if args.list_file:
        raw_paths.extend(
            line.strip() for line in args.list_file.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    files: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*.png")))
        elif p.is_file():
            files.append(p)

    if not files:
        print("done: 0 file(s)", flush=True)
        return 0

    changed = 0
    work = [(str(p), args.max_mb, args.max_dim) for p in files]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_worker, item) for item in work]
        for fut in as_completed(futures):
            try:
                did, name, before, after = fut.result()
                if did:
                    changed += 1
                    print(f"saved {name}: {before:.2f}MB -> {after:.2f}MB", flush=True)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr, flush=True)
                return 1

    print(f"done: {changed}/{len(files)} file(s) updated", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
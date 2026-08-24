"""
Hand-written 8-bit RGB PNG output (stdlib zlib/struct) shared by
train_mae.py (reconstruction grids) and linear_probe.py (prediction grid).
No imaging library - the same writer evaluate_vit.py uses, factored out so
the two scripts in this project don't duplicate it.
"""

import struct
import zlib

import numpy as np


def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a PNG: one filter byte 0 per
    scanline, IHDR/IDAT/IEND chunks. No imaging library."""
    H, W, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(H))

    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def upscale(img, scale):
    """Nearest-neighbour upscale of an (H, W, 3) image by an integer
    factor."""
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def tensor_to_uint8(x):
    """(3, 32, 32) float [0,1] -> (32, 32, 3) uint8."""
    img = np.clip(x.cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255)
    return img.astype(np.uint8)

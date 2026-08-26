"""
Hand-written 8-bit RGB PNG output (stdlib zlib/struct), shared by
train_dit.py (training-time sample grids) and evaluate_dit.py (sample grids,
CFG sweep, nearest-neighbour check). No imaging library - the same writer
training/mae-cifar100 and training/vit-cifar10 use, factored out so the
scripts in this project don't duplicate it.
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


def make_rgb_grid(images, grid_cols, upscale_factor=4, border=2, pad_color=(32, 32, 32)):
    """Tile a list of (3, 32, 32) float arrays in [0,1] into one uint8 RGB
    image grid, each cell upscaled and framed with a `border`-pixel padding
    of `pad_color`."""
    n = len(images)
    image_size = images[0].shape[1]
    grid_rows = (n + grid_cols - 1) // grid_cols
    cell_img = upscale_factor * image_size
    cell = cell_img + 2 * border

    canvas = np.zeros((grid_rows * cell, grid_cols * cell, 3), dtype=np.uint8)
    frame = np.full((cell, cell, 3), pad_color, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, grid_cols)
        tile = frame.copy()
        tile[border:border + cell_img, border:border + cell_img] = upscale(
            tensor_to_uint8(img), upscale_factor
        )
        y0, x0 = r * cell, c * cell
        canvas[y0:y0 + cell, x0:x0 + cell] = tile
    return canvas

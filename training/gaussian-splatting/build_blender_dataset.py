"""
Turn the raw NeRF-Synthetic (Blender) scenes into a memory-mappable uint8
image corpus plus a camera index, ready for 3D Gaussian Splatting.

NeRF-Synthetic is 8 path-traced Blender scenes (chair, drums, ficus,
hotdog, lego, materials, mic, ship), each 100 train / 100 val / 200 test
renders at 800x800 RGBA on a transparent background, with a shared
horizontal FOV and a per-frame 4x4 camera-to-world matrix. It is the
standard first target for both NeRF and 3DGS, which is what makes the
published PSNR table something this pipeline can be checked against.

Four things are done by hand here that a library would normally hide:

  1. The PNG decoder. Every render is walked chunk by chunk with the stdlib
     struct module, the IDAT payload inflated with stdlib zlib, and the
     scanlines un-filtered by hand - no Pillow, no torchvision, no imageio.
     This is the mirror image of the hand-written PNG *writer* already in
     training/mnist-vae, training/cifar10-vqvae and
     training/flow-matching-mnist, which emit their sample grids the same
     way. The IHDR fields are asserted rather than assumed (8-bit, colour
     type 6 / RGBA, deflate, adaptive filtering, non-interlaced - which is
     what all 3,200 files in this dataset actually are), and every chunk's
     CRC-32 is verified so a truncated download fails here rather than as
     a garbled texture 20 minutes into training.

     The un-filtering is the part worth reading. PNG picks one of five
     filters per scanline, and on this dataset the two with intra-row byte
     dependencies dominate: measured over a sample of these files the row
     histogram is roughly 18% None, 8% Sub, 11% Up, 26% Average, 38%
     Paeth. A byte-at-a-time Python loop over 3,200 images x 800 rows x
     3,200 bytes is not viable, so the loop is inverted: images are
     decoded in *batches*, and because rows depend on the previous row of
     the same image but never on another image, one numpy operation can
     advance every image in the batch by one pixel at once. Sub collapses
     to a cumulative sum along x per channel lane and needs no loop at
     all; Average and Paeth keep an 800-step loop over pixels, but each
     step is a vector op over the whole batch. That turns ~8e9 scalar
     steps into ~1e6 vector ops.

  2. The alpha compositing. The renders are RGBA with a transparent
     background; every published NeRF-Synthetic number - NeRF's, Mip-NeRF's,
     3DGS's - is measured after compositing over **white**. That is done
     here, once, at build time: rgb = rgb*a + (1-a). Changing it (to black,
     or to keeping the alpha channel) silently makes the PSNR this
     pipeline reports incomparable to every baseline in the literature.

  3. The camera convention. transform_matrix is camera-to-world in
     Blender/OpenGL convention: +X right, +Y up, -Z forward. A rasterizer
     wants OpenCV convention: +X right, +Y down, +Z forward. Negating the
     second and third basis *columns* converts between them; the result is
     inverted here to give the world-to-camera matrix the projection
     actually uses. Getting this wrong is the classic NeRF-data footgun -
     it produces an upside-down, mirrored scene that still optimizes to a
     plausible-looking loss curve.

  4. The storage layout. One scene is 400 x 800 x 800 x 3 = 768 MB of
     uint8 (2.3 GB as float32), and all eight are 6.1 GB, so this does not
     write an .npz the way the MNIST/CIFAR/IMDB builders in the sibling
     pipelines do. Every scene gets one raw uint8 file that
     train_gaussians.py opens with np.memmap, next to a small .npz of
     camera matrices, intrinsics, split labels and the scene extent.

Per-split file counts are verified (exactly 100 / 100 / 200) so a partial
or in-progress extraction fails loudly here instead of silently optimizing
against half a scene - the same guardrail as build_ljspeech_dataset.py's
13,100-wav check and build_imdb_dataset.py's 12,500-file check.

Usage:
    uv run --directory training/gaussian-splatting python build_blender_dataset.py \
        --data-dir "C:\\path\\to\\nerf_synthetic" \
        --scene lego \
        --output-dir data

    # or all eight scenes in one pass
    uv run --directory training/gaussian-splatting python build_blender_dataset.py \
        --data-dir "C:\\path\\to\\nerf_synthetic" --scene all --output-dir data
"""

import argparse
import json
import os
import struct
import time
import zlib

import numpy as np

SCENES = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
SPLITS = ["train", "val", "test"]
EXPECTED_COUNTS = {"train": 100, "val": 100, "test": 200}
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# Dataset layout
# --------------------------------------------------------------------------

def resolve_scene_dir(data_dir, scene):
    """The nerf_synthetic.zip drop commonly extracts to a nested
    nerf_synthetic/ subfolder. Accept either the folder that holds the
    scene folders directly, or one wrapping a nerf_synthetic subfolder that
    does - the same tolerance build_ljspeech_dataset.py has for a nested
    LJSpeech-1.1/ and build_imdb_dataset.py for a nested aclImdb/."""
    for candidate in (
        os.path.join(data_dir, scene),
        os.path.join(data_dir, "nerf_synthetic", scene),
    ):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find scene '{scene}' under {data_dir} (looked directly and "
        f"under a nerf_synthetic subfolder). Expected one of: "
        f"{', '.join(SCENES)}"
    )


def list_split_images(scene_dir, split):
    """Return the sorted render filenames for a split.

    The test/ folder holds 600 files, not 200: each render r_N.png ships
    alongside r_N_depth_0000.png and r_N_normal_0000.png auxiliary maps
    that this pipeline does not use. Filtering them out here is why the
    count check below can be an exact equality.
    """
    split_dir = os.path.join(scene_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split folder: {split_dir}")
    return sorted(
        name for name in os.listdir(split_dir)
        if name.startswith("r_") and name.endswith(".png")
        and "_depth_" not in name and "_normal_" not in name
    )


# --------------------------------------------------------------------------
# PNG decoding, by hand
# --------------------------------------------------------------------------

def read_png_scanlines(path):
    """Parse a PNG file and return (width, height, filtered scanline bytes).

    Layout: an 8-byte signature, then a sequence of chunks, each
    uint32 length + 4-byte type + <length bytes> + uint32 CRC-32 of
    (type + body). Only IHDR and IDAT are needed; the pHYs/tEXt chunks
    Blender writes are skipped. The returned bytes are still filtered -
    one leading filter-type byte per scanline - and are un-filtered in
    batch by unfilter_batch().
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError(f"{path}: not a PNG file")

    offset = 8
    header = None
    idat_parts = []
    while offset + 12 <= len(data):
        length, chunk_type = struct.unpack(">I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + length]
        if len(body) < length:
            raise ValueError(f"{path}: truncated {chunk_type.decode()} chunk")
        expected_crc, = struct.unpack(
            ">I", data[offset + 8 + length:offset + 12 + length]
        )
        if zlib.crc32(chunk_type + body) & 0xFFFFFFFF != expected_crc:
            raise ValueError(
                f"{path}: CRC mismatch in {chunk_type.decode()} chunk - the file "
                f"is corrupt or the download was truncated"
            )

        if chunk_type == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif chunk_type == b"IDAT":
            idat_parts.append(body)
        elif chunk_type == b"IEND":
            break
        offset += 12 + length

    if header is None:
        raise ValueError(f"{path}: no IHDR chunk")
    width, height, bit_depth, colour_type, compression, filter_method, interlace = header

    # Assert rather than assume. Every file in NeRF-Synthetic is 8-bit RGBA,
    # deflate-compressed, adaptively filtered and non-interlaced; anything
    # else would need decoder paths this pipeline deliberately does not have.
    if bit_depth != 8:
        raise ValueError(f"{path}: bit depth {bit_depth}, expected 8")
    if colour_type != 6:
        raise ValueError(
            f"{path}: colour type {colour_type}, expected 6 (RGBA). This "
            f"pipeline composites the alpha channel over white and needs it."
        )
    if compression != 0 or filter_method != 0:
        raise ValueError(f"{path}: unsupported compression/filter method")
    if interlace != 0:
        raise ValueError(f"{path}: interlaced PNG, expected non-interlaced")

    raw = zlib.decompress(b"".join(idat_parts))
    expected_len = height * (1 + width * 4)
    if len(raw) != expected_len:
        raise ValueError(
            f"{path}: inflated to {len(raw)} bytes, expected {expected_len}"
        )
    return width, height, raw


def unfilter_batch(raw, height, width, bpp=4):
    """Undo PNG scanline filtering for a whole batch of same-size images.

    raw is (batch, height, 1 + width*bpp) uint8: one filter-type byte then
    the filtered pixel bytes, per scanline. Returns (batch, height, width,
    bpp) uint8.

    Rows within an image are sequentially dependent (Up/Average/Paeth all
    read the previous row) but images are independent, so the batch axis is
    the one that vectorizes. Within a row, Sub and Average and Paeth read
    the pixel bpp bytes to the left, which is why those three cannot simply
    be a single array add:

      0 None     x = raw
      1 Sub      x[p] = raw[p] + x[p-1]                     -> cumulative sum
      2 Up       x[p] = raw[p] + prior[p]                   -> one add
      3 Average  x[p] = raw[p] + ((x[p-1] + prior[p]) >> 1) -> loop over p
      4 Paeth    x[p] = raw[p] + paeth(x[p-1], prior[p], prior[p-1])

    All arithmetic is mod 256. Sub's mod-256 running sum is exactly a
    cumsum masked with 255, since addition mod 256 is associative - so the
    only real loops are Average and Paeth, and each of their steps advances
    every image in the batch simultaneously.
    """
    batch = raw.shape[0]
    filters = raw[:, :, 0]
    if filters.max() > 4:
        raise ValueError(
            f"invalid PNG filter type {int(filters.max())} (expected 0-4)"
        )

    data = raw[:, :, 1:].reshape(batch, height, width, bpp).astype(np.int16)
    out = np.empty((batch, height, width, bpp), dtype=np.uint8)
    prior = np.zeros((batch, width, bpp), dtype=np.int16)
    current = np.empty((batch, width, bpp), dtype=np.int16)

    for y in range(height):
        row = data[:, y]
        row_filters = filters[:, y]

        idx = np.flatnonzero(row_filters == 0)
        if idx.size:
            current[idx] = row[idx]

        idx = np.flatnonzero(row_filters == 1)
        if idx.size:
            # int32 for the accumulation: 800 pixels x 255 overflows int16.
            current[idx] = np.cumsum(row[idx].astype(np.int32), axis=1) & 255

        idx = np.flatnonzero(row_filters == 2)
        if idx.size:
            current[idx] = (row[idx] + prior[idx]) & 255

        idx = np.flatnonzero(row_filters == 3)
        if idx.size:
            filtered, above = row[idx], prior[idx]
            decoded = np.empty_like(filtered)
            left = np.zeros((idx.size, bpp), dtype=np.int16)
            for x in range(width):
                left = (filtered[:, x] + ((left + above[:, x]) >> 1)) & 255
                decoded[:, x] = left
            current[idx] = decoded

        idx = np.flatnonzero(row_filters == 4)
        if idx.size:
            filtered, above = row[idx], prior[idx]
            decoded = np.empty_like(filtered)
            left = np.zeros((idx.size, bpp), dtype=np.int16)        # x[p-1]
            upper_left = np.zeros((idx.size, bpp), dtype=np.int16)  # prior[p-1]
            for x in range(width):
                up = above[:, x]
                # Paeth predictor: of left/up/upper_left, pick whichever is
                # closest to the linear estimate left + up - upper_left.
                pa = np.abs(up - upper_left)
                pb = np.abs(left - upper_left)
                pc = np.abs(left + up - 2 * upper_left)
                predicted = np.where(
                    (pa <= pb) & (pa <= pc),
                    left,
                    np.where(pb <= pc, up, upper_left),
                )
                left = (filtered[:, x] + predicted) & 255
                decoded[:, x] = left
                upper_left = up
            current[idx] = decoded

        out[:, y] = current
        prior[...] = current

    return out


def decode_batch_to_white(paths, width, height):
    """Decode a batch of RGBA PNGs and composite them over white.

    Returns (batch, height, width, 3) uint8. The white background is not a
    stylistic choice - see the module docstring.
    """
    raw = np.empty((len(paths), height, 1 + width * 4), dtype=np.uint8)
    for i, path in enumerate(paths):
        file_width, file_height, scanlines = read_png_scanlines(path)
        if (file_width, file_height) != (width, height):
            raise ValueError(
                f"{path}: {file_width}x{file_height}, but this scene's first "
                f"image is {width}x{height} - all views must share a resolution"
            )
        raw[i] = np.frombuffer(scanlines, dtype=np.uint8).reshape(height, -1)

    rgba = unfilter_batch(raw, height, width).astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    composited = rgba[..., :3] * alpha + (1.0 - alpha)
    return np.clip(composited * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------

def load_camera_metadata(scene_dir, split):
    """Read transforms_<split>.json and return (camera_angle_x, frames)."""
    path = os.path.join(scene_dir, f"transforms_{split}.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "camera_angle_x" not in meta or "frames" not in meta:
        raise ValueError(f"{path}: missing camera_angle_x or frames")
    return float(meta["camera_angle_x"]), meta["frames"]


def blender_to_opencv(c2w):
    """Convert a Blender/OpenGL camera-to-world matrix to OpenCV convention.

    Blender cameras look down -Z with +Y up; a rasterizer projects along +Z
    with +Y down. The rotation part of c2w has the camera's right/up/back
    basis vectors as its columns, so negating columns 1 and 2 flips up->down
    and back->forward. The translation column (the camera centre in world
    space) is untouched.
    """
    converted = c2w.copy()
    converted[:, 1] *= -1.0
    converted[:, 2] *= -1.0
    return converted


def scene_extent_from_cameras(camera_centres):
    """Radius of the sphere enclosing every camera, about their centroid.

    3DGS scales the position learning rate and the clone-vs-split size
    cutoff by this, so a scene twice as large gets proportionally larger
    steps. The reference calls it nerf_normalization["radius"] and pads it
    by 10%.
    """
    centroid = camera_centres.mean(axis=0)
    radius = float(np.linalg.norm(camera_centres - centroid, axis=1).max() * 1.1)
    return centroid.astype(np.float32), radius


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def build_scene(scene, args):
    scene_dir = resolve_scene_dir(args.data_dir, scene)
    print(f"\n=== {scene} ===")
    print(f"Using scene at {scene_dir}")

    # Collect every (path, pose) pair first, so a bad camera file fails
    # before 768 MB of pixels have been written.
    paths, poses, names, split_codes = [], [], [], []
    camera_angle_x = None

    for split in SPLITS:
        image_names = list_split_images(scene_dir, split)
        expected = EXPECTED_COUNTS[split]
        if args.verify_counts and len(image_names) != expected:
            raise RuntimeError(
                f"{scene}/{split}: expected {expected} renders but found "
                f"{len(image_names)}. A partial or in-progress extraction would "
                f"optimize against an incomplete scene - re-extract "
                f"nerf_synthetic.zip and rerun (or pass --verify-counts 0 if "
                f"this is deliberately a different capture)."
            )

        angle, frames = load_camera_metadata(scene_dir, split)
        if camera_angle_x is None:
            camera_angle_x = angle
        elif abs(angle - camera_angle_x) > 1e-9:
            raise ValueError(
                f"{scene}: camera_angle_x differs between splits "
                f"({angle} vs {camera_angle_x}); this pipeline assumes one "
                f"shared intrinsic for the whole scene"
            )
        if len(frames) != len(image_names):
            raise RuntimeError(
                f"{scene}/{split}: transforms_{split}.json has {len(frames)} "
                f"frames but the folder has {len(image_names)} renders"
            )

        available = set(image_names)
        for frame in frames:
            name = os.path.basename(frame["file_path"]) + ".png"
            if name not in available:
                raise RuntimeError(
                    f"{scene}/{split}: transforms_{split}.json references "
                    f"{name}, which is not in the folder"
                )
            paths.append(os.path.join(scene_dir, split, name))
            poses.append(np.array(frame["transform_matrix"], dtype=np.float64))
            names.append(f"{split}/{name}")
            split_codes.append(SPLIT_CODE[split])

        print(f"  {split}: {len(image_names)} renders, {len(frames)} poses")

    c2w = np.stack([blender_to_opencv(p) for p in poses]).astype(np.float32)
    w2c = np.linalg.inv(c2w.astype(np.float64)).astype(np.float32)
    split_codes = np.array(split_codes, dtype=np.int64)

    # Resolution comes from the first file; every other file is checked
    # against it during decode.
    width, height, _ = read_png_scanlines(paths[0])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    centre, radius = scene_extent_from_cameras(c2w[:, :3, 3])

    print(f"  {width}x{height}, camera_angle_x {camera_angle_x:.6f} rad "
          f"({np.degrees(camera_angle_x):.2f} deg), focal {focal:.2f} px")
    print(f"  scene centre {np.round(centre, 4).tolist()}, radius {radius:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    image_path = os.path.join(args.output_dir, f"{scene}_images.u8")
    meta_path = os.path.join(args.output_dir, f"{scene}_meta.npz")

    start = time.time()
    with open(image_path, "wb") as out:
        for begin in range(0, len(paths), args.batch_size):
            chunk = paths[begin:begin + args.batch_size]
            out.write(decode_batch_to_white(chunk, width, height).tobytes())
            done = begin + len(chunk)
            elapsed = time.time() - start
            print(f"  decoded {done}/{len(paths)} renders "
                  f"({elapsed:.1f}s, {done / max(elapsed, 1e-6):.1f} img/s)")

    np.savez(
        meta_path,
        scene=np.array(scene),
        width=np.int64(width),
        height=np.int64(height),
        focal=np.float32(focal),
        # (S-1)/2, not S/2. The reference rasterizer maps NDC to pixel indices
        # with ((v + 1) * S - 1) * 0.5, which puts the principal point half a
        # pixel below the image centre. Half a pixel of systematic reprojection
        # offset is small but free to avoid, and it is exactly the kind of
        # detail that quietly costs a tenth of a dB against the published table.
        cx=np.float32((width - 1) / 2.0),
        cy=np.float32((height - 1) / 2.0),
        camera_angle_x=np.float32(camera_angle_x),
        c2w=c2w,
        w2c=w2c,
        split=split_codes,
        names=np.array(names),
        scene_centre=centre,
        scene_radius=np.float32(radius),
    )

    size_gb = os.path.getsize(image_path) / 1e9
    print(f"  Saved {image_path} ({size_gb:.2f} GB uint8, "
          f"{len(paths)}x{height}x{width}x3)")
    print(f"  Saved {meta_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=r"C:\path\to\nerf_synthetic",
        help="Directory containing the extracted NeRF-Synthetic scene folders "
             "(a nested nerf_synthetic subfolder is also accepted).",
    )
    parser.add_argument(
        "--scene", default="lego",
        help=f"Scene to build, or 'all' for every scene. One of: "
             f"{', '.join(SCENES)}",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Renders decoded together. The un-filtering loop vectorizes over "
             "this axis, so larger is faster but holds batch_size x 2.6 MB of "
             "filtered bytes plus the decoded output in RAM.",
    )
    parser.add_argument(
        "--verify-counts", type=int, default=1,
        help="Refuse to build unless each split has exactly 100/100/200 "
             "renders (guards against a partial extraction). 0 to skip.",
    )
    args = parser.parse_args()

    if args.scene == "all":
        scenes = SCENES
    elif args.scene in SCENES:
        scenes = [args.scene]
    else:
        raise SystemExit(
            f"Unknown scene '{args.scene}'. Expected 'all' or one of: "
            f"{', '.join(SCENES)}"
        )

    started = time.time()
    for scene in scenes:
        build_scene(scene, args)
    print(f"\nBuilt {len(scenes)} scene(s) in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

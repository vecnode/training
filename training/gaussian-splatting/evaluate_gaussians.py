"""
Evaluate a trained Gaussian scene on the held-out NeRF-Synthetic test split,
and export it in a form you can actually fly around.

Three outputs, in increasing order of how much they tell you:

  1. **PSNR / SSIM on the 200 test views.** This is the number the
     literature reports for this dataset, and --compare puts it next to the
     published 3DGS and NeRF figures for the same scene. It is only
     comparable when rendered at the full 800x800 with the white background
     the builder composited - evaluating a --downscale 2 checkpoint prints
     a warning, because a smaller image is an easier target and the number
     would flatter the model.

  2. **render / ground-truth PNG pairs**, plus an optional orbit sequence
     on a camera path that appears in no split. Novel-view synthesis is the
     actual task; a metric averaged over 200 views hides floaters that only
     appear from angles the training cameras never took.

  3. **A .ply point cloud** in the layout the reference implementation
     writes, which the common web viewers (SuperSplat, antimatter15's
     splat viewer, PlayCanvas) load directly. This is the deliverable: the
     scene, in your browser, in real time.

Everything here is hand-written for the same reason as the rest of the
folder - the PNG writer with stdlib zlib/struct (the mirror of the decoder
in build_blender_dataset.py), the binary .ply writer with no plyfile
dependency, and the orbit camera path.

There is deliberately **no LPIPS**, though the 3DGS paper reports it. LPIPS
needs a pretrained VGG or AlexNet, and a from-scratch folder that quietly
downloads an ImageNet backbone to score itself is not from-scratch. This is
the same rule that keeps FID out of training/flow-matching-mnist and
ViSQOL/PESQ out of training/rvq-audio-codec.

Usage:
    uv run --directory training/gaussian-splatting python evaluate_gaussians.py \
        --data-dir data --scene lego \
        --checkpoint-path runs/lego/gaussians_best.pt \
        --output-dir runs/lego

    # every scene you have trained, against the published baselines
    uv run --directory training/gaussian-splatting python evaluate_gaussians.py \
        --compare runs
"""

import argparse
import glob
import json
import math
import os
import struct
import time
import zlib

import numpy as np
import torch

from train_gaussians import (
    gaussian_window, psnr, render, ssim,
)

# --------------------------------------------------------------------------
# Published baselines, for --compare.
#
# 3DGS: Kerbl et al. 2023 (arXiv 2308.04079), 30k iterations, NeRF-Synthetic.
# NeRF: Mildenhall et al. 2020 (arXiv 2003.08934), same split.
# Both are PSNR on the 200 test views at 800x800 over a white background,
# which is exactly what this evaluator measures - that is the whole reason
# this pipeline composites over white at build time.
# --------------------------------------------------------------------------

BASELINE_3DGS = {
    "chair": 35.83, "drums": 26.15, "ficus": 34.87, "hotdog": 37.72,
    "lego": 35.78, "materials": 30.00, "mic": 35.36, "ship": 30.80,
}
BASELINE_NERF = {
    "chair": 33.00, "drums": 25.01, "ficus": 30.13, "hotdog": 36.18,
    "lego": 32.54, "materials": 29.62, "mic": 32.91, "ship": 28.65,
}


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def _png_chunk(tag, payload):
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def write_png(path, image):
    """Write an (H, W, 3) uint8 array as a PNG, by hand.

    Filter type 0 (None) on every scanline: these are photographic renders
    where the adaptive filters buy little, and it keeps the writer to the
    few lines that make the format legible next to the decoder.
    """
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", header))
        f.write(_png_chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(_png_chunk(b"IEND", b""))


def write_ply(path, gaussians):
    """Write the Gaussians as a binary .ply the standard viewers understand.

    The stored values are the **raw** parameters, not the activated ones:
    log scales, logit opacity, an unnormalized quaternion and SH
    coefficients. Every viewer applies exp/sigmoid/normalize itself, so
    writing activated values here produces a scene that loads but looks
    wrong - washed out and the wrong size.

    Property order matters and matches the reference exactly: position,
    an unused normal, f_dc (3), f_rest (45, channel-major), opacity,
    scale (3), rotation (4). The SH rest coefficients are transposed to
    (channels, coefficients) before flattening, which is the ordering the
    viewers expect.
    """
    xyz = gaussians["xyz"].numpy().astype(np.float32)
    normals = np.zeros_like(xyz)
    f_dc = gaussians["features_dc"].transpose(1, 2).flatten(1).numpy().astype(np.float32)
    f_rest = gaussians["features_rest"].transpose(1, 2).flatten(1).numpy().astype(np.float32)
    opacity = gaussians["raw_opacity"].numpy().astype(np.float32)
    scale = gaussians["log_scaling"].numpy().astype(np.float32)
    rotation = gaussians["raw_rotation"].numpy().astype(np.float32)

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{i}" for i in range(scale.shape[1])]
    names += [f"rot_{i}" for i in range(rotation.shape[1])]

    table = np.concatenate(
        [xyz, normals, f_dc, f_rest, opacity, scale, rotation], axis=1
    ).astype(np.float32)

    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {table.shape[0]}"]
    header += [f"property float {name}" for name in names]
    header += ["end_header", ""]
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(table.tobytes())
    return table.shape[0], table.nbytes


# --------------------------------------------------------------------------
# The trained scene, without an optimizer
# --------------------------------------------------------------------------

class LoadedScene:
    """Read-only view of a checkpoint, exposing what render() reads."""

    def __init__(self, gaussians, device):
        self.parameters = {k: v.to(device) for k, v in gaussians.items()
                           if torch.is_tensor(v)}
        self.max_sh_degree = int(gaussians.get("max_sh_degree", 3))
        self.extent = float(gaussians.get("extent", 1.0))

    @property
    def xyz(self):
        return self.parameters["xyz"]

    @property
    def scaling(self):
        return torch.exp(self.parameters["log_scaling"])

    @property
    def rotation(self):
        return self.parameters["raw_rotation"]

    @property
    def opacity(self):
        return torch.sigmoid(self.parameters["raw_opacity"])

    @property
    def features(self):
        return torch.cat(
            [self.parameters["features_dc"], self.parameters["features_rest"]], dim=1
        )

    @property
    def count(self):
        return self.parameters["xyz"].shape[0]


def orbit_cameras(reference_c2w, count, width, height, focal, cx, cy, device):
    """A turntable path at the mean radius and elevation of the real cameras.

    None of these poses appears in any split, which is the point: a scene
    that scores well on the test views but falls apart on a smooth path
    between them has floaters, and no averaged metric will say so.
    """
    centres = reference_c2w[:, :3, 3].cpu().numpy()
    radius = float(np.linalg.norm(centres, axis=1).mean())
    elevation = float(np.arcsin(np.clip(centres[:, 2] / np.linalg.norm(
        centres, axis=1), -1.0, 1.0)).mean())

    cameras = []
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        position = np.array([
            radius * math.cos(elevation) * math.cos(theta),
            radius * math.cos(elevation) * math.sin(theta),
            radius * math.sin(elevation),
        ])
        # Look-at, built directly in OpenCV convention: +Z forward towards
        # the origin, +X right, +Y down.
        forward = -position / np.linalg.norm(position)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, forward
        c2w[:3, 3] = position
        w2c = torch.from_numpy(np.linalg.inv(c2w)).to(device)
        cameras.append({
            "w2c": w2c, "centre": torch.from_numpy(position.astype(np.float32)).to(device),
            "focal": focal, "cx": cx, "cy": cy, "width": width, "height": height,
        })
    return cameras


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------

def print_comparison(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise SystemExit(
            f"No metrics.json found under {root}/*/ - run this script without "
            f"--compare on each trained scene first."
        )

    best = {}
    for row in rows:
        scene = row["scene"]
        if row.get("downscale", 1) != 1:
            continue
        if scene not in best or row["psnr"] > best[scene]["psnr"]:
            best[scene] = row
    if not best:
        raise SystemExit(
            "Every metrics.json found was rendered at --downscale > 1, which "
            "is not comparable to the published table. Re-evaluate at "
            "--downscale 1."
        )

    print(f"\n{'scene':<11}{'ours':>9}{'3DGS':>9}{'NeRF':>9}{'vs 3DGS':>10}"
          f"{'SSIM':>9}{'gaussians':>12}{'iters':>8}")
    print("-" * 77)
    deltas = []
    for scene in sorted(best):
        row = best[scene]
        reference = BASELINE_3DGS.get(scene)
        delta = row["psnr"] - reference if reference else float("nan")
        deltas.append(delta)
        print(f"{scene:<11}{row['psnr']:>9.2f}{reference:>9.2f}"
              f"{BASELINE_NERF.get(scene, float('nan')):>9.2f}{delta:>+10.2f}"
              f"{row['ssim']:>9.4f}{row['gaussians']:>12,}{row['iteration']:>8}")
    print("-" * 77)
    covered = [s for s in best if s in BASELINE_3DGS]
    print(f"{'mean':<11}{np.mean([best[s]['psnr'] for s in covered]):>9.2f}"
          f"{np.mean([BASELINE_3DGS[s] for s in covered]):>9.2f}"
          f"{np.mean([BASELINE_NERF[s] for s in covered]):>9.2f}"
          f"{np.mean(deltas):>+10.2f}")
    if len(covered) < len(BASELINE_3DGS):
        missing = sorted(set(BASELINE_3DGS) - set(covered))
        print(f"\n({len(covered)} of 8 scenes; missing: {', '.join(missing)}. "
              f"The mean is over the scenes present, so it is not the "
              f"published 8-scene average until all eight are trained.)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare", default=None,
        help="Print the cross-scene table from every <dir>/*/metrics.json and "
             "exit. Nothing else is needed.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--checkpoint-path", default="runs/lego/gaussians_best.pt")
    parser.add_argument("--output-dir", default="runs/lego")
    parser.add_argument(
        "--downscale", type=int, default=1,
        help="Render resolution divisor. Leave at 1: the published table is "
             "at 800x800 and anything else is not comparable.",
    )
    parser.add_argument("--num-pairs", type=int, default=8,
                        help="render/ground-truth PNG pairs to write.")
    parser.add_argument("--orbit-frames", type=int, default=0,
                        help="Frames of a turntable path on unseen poses. 0 to skip.")
    parser.add_argument("--export-ply", type=int, default=1)
    parser.add_argument("--near-plane", type=float, default=0.2)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--max-gaussians-per-tile", type=int, default=8192)
    parser.add_argument("--tile-chunk", type=int, default=64)
    parser.add_argument("--tile-slab", type=int, default=2048)
    parser.add_argument("--checkpoint-tiles", type=int, default=0)
    args = parser.parse_args()

    if args.compare:
        print_comparison(args.compare)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu",
                            weights_only=False)
    scene = LoadedScene(checkpoint["gaussians"], device)
    trained_at = int(checkpoint.get("downscale", 1))
    iteration = int(checkpoint.get("iteration", 0))
    print(f"Loaded {args.checkpoint_path}: {scene.count:,} Gaussians, "
          f"iteration {iteration}, trained at downscale {trained_at}")

    if args.downscale != 1:
        print(f"\nWARNING: rendering at downscale {args.downscale}. A smaller "
              f"image is an easier target, so this PSNR is NOT comparable to "
              f"the published NeRF-Synthetic table. --compare will ignore it.")
    elif trained_at != 1:
        print(f"\nNOTE: this checkpoint was trained at downscale {trained_at} "
              f"but is being evaluated at full 800x800. That is a fair "
              f"comparison to the baselines, but the model never saw this "
              f"resolution and will score below a natively-trained one.")

    # --- data ---
    meta = np.load(os.path.join(args.data_dir, f"{args.scene}_meta.npz"),
                   allow_pickle=True)
    height, width = int(meta["height"]), int(meta["width"])
    split = meta["split"]
    memmap = np.memmap(os.path.join(args.data_dir, f"{args.scene}_images.u8"),
                       dtype=np.uint8, mode="r", shape=(split.shape[0], height, width, 3))
    test = np.flatnonzero(split == 2)

    focal = float(meta["focal"]) / args.downscale
    cx, cy = float(meta["cx"]) / args.downscale, float(meta["cy"]) / args.downscale
    out_h, out_w = height // args.downscale, width // args.downscale
    w2c = torch.from_numpy(meta["w2c"]).to(device)
    c2w = torch.from_numpy(meta["c2w"]).to(device)

    background = torch.ones(3, device=device)
    window = gaussian_window(11, 1.5, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- test split ---
    scores = []
    started = time.time()
    with torch.no_grad():
        for rank, index in enumerate(test):
            camera = {"w2c": w2c[index], "centre": c2w[index, :3, 3], "focal": focal,
                      "cx": cx, "cy": cy, "width": out_w, "height": out_h}
            image = render(scene, camera, background, args,
                           scene.max_sh_degree)["image"].clamp(0, 1)

            target = torch.from_numpy(
                np.ascontiguousarray(memmap[index])
            ).to(device).float() / 255.0
            if args.downscale > 1:
                target = (target.reshape(out_h, args.downscale, out_w,
                                         args.downscale, 3).mean(dim=(1, 3)))
            scores.append((psnr(image, target), float(ssim(image, target, window))))

            if rank < args.num_pairs:
                as_bytes = (image * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
                write_png(os.path.join(args.output_dir, f"render_{rank:02d}.png"), as_bytes)
                write_png(
                    os.path.join(args.output_dir, f"gt_{rank:02d}.png"),
                    (target * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy(),
                )
            if (rank + 1) % 50 == 0:
                print(f"  {rank + 1}/{len(test)} test views "
                      f"({(rank + 1) / (time.time() - started):.1f} views/s)")

    mean_psnr = float(np.mean([s[0] for s in scores]))
    mean_ssim = float(np.mean([s[1] for s in scores]))
    render_fps = len(test) / (time.time() - started)

    print(f"\n=== {args.scene} @ {out_w}x{out_h}, {len(test)} test views ===")
    print(f"  PSNR {mean_psnr:.3f}  (min {min(s[0] for s in scores):.2f}, "
          f"max {max(s[0] for s in scores):.2f})")
    print(f"  SSIM {mean_ssim:.4f}")
    print(f"  {scene.count:,} Gaussians, {render_fps:.2f} views/s")
    if args.downscale == 1 and args.scene in BASELINE_3DGS:
        print(f"  published 3DGS {BASELINE_3DGS[args.scene]:.2f} "
              f"({mean_psnr - BASELINE_3DGS[args.scene]:+.2f}), "
              f"NeRF {BASELINE_NERF[args.scene]:.2f} "
              f"({mean_psnr - BASELINE_NERF[args.scene]:+.2f})")
    print(f"  wrote {min(args.num_pairs, len(test))} render/gt PNG pairs")

    with open(os.path.join(args.output_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({"scene": args.scene, "psnr": mean_psnr, "ssim": mean_ssim,
                   "gaussians": scene.count, "iteration": iteration,
                   "downscale": args.downscale, "views": len(test),
                   "render_views_per_second": render_fps,
                   "per_view_psnr": [s[0] for s in scores]}, f, indent=2)

    # --- orbit ---
    if args.orbit_frames:
        orbit_dir = os.path.join(args.output_dir, "orbit")
        os.makedirs(orbit_dir, exist_ok=True)
        cameras = orbit_cameras(c2w, args.orbit_frames, out_w, out_h,
                                focal, cx, cy, device)
        with torch.no_grad():
            for i, camera in enumerate(cameras):
                image = render(scene, camera, background, args,
                               scene.max_sh_degree)["image"].clamp(0, 1)
                write_png(
                    os.path.join(orbit_dir, f"orbit_{i:03d}.png"),
                    (image * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy(),
                )
        print(f"  wrote {args.orbit_frames} orbit frames to {orbit_dir}/")

    # --- ply ---
    if args.export_ply:
        ply_path = os.path.join(args.output_dir, f"{args.scene}.ply")
        count, size = write_ply(ply_path, checkpoint["gaussians"])
        print(f"  wrote {ply_path} ({count:,} Gaussians, {size / 1e6:.1f} MB)")
        print(f"  drop it into https://superspl.at/editor or "
              f"https://antimatter15.com/splat/ to fly around it")


if __name__ == "__main__":
    main()

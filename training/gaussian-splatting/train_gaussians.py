"""
3D Gaussian Splatting, optimized from scratch on one NeRF-Synthetic scene.

Kerbl, Kopanas, Leimkuehler and Drettakis, "3D Gaussian Splatting for
Real-Time Radiance Field Rendering", SIGGRAPH 2023
(https://arxiv.org/abs/2308.04079).

There is no network here. The scene *is* the parameters: a few hundred
thousand anisotropic 3D Gaussians, each carrying a position, a scale, a
rotation quaternion, an opacity and 16 spherical-harmonic colour
coefficients, optimized directly against the training photographs by
gradient descent. Nothing is pretrained and nothing is shared between
scenes - every scene is its own optimization, which is why this sits in
training/ rather than fine-tuning/.

Written by hand here, with no splatting library (no gsplat, no
diff-gaussian-rasterization) and no graphics stack (no COLMAP, no open3d,
no Pillow/torchvision):

  * the EWA projection of a 3D covariance to a 2D screen-space conic
  * tile binning, depth sorting and alpha compositing
  * spherical-harmonics evaluation of view-dependent colour
  * adaptive densification - clone, split, prune, opacity reset - including
    the Adam-state surgery that keeps momentum attached to Gaussians whose
    number changes underneath the optimizer
  * SSIM

**The rasterizer is plain PyTorch, not a CUDA kernel.** That is the point
of this folder, and it is a real trade. The reference implementation ships
a hand-derived CUDA backward pass; here the alpha compositing is written as
an exclusive cumulative product, which autograd differentiates on its own -
so the projection and compositing math stays readable and there is no
backward pass to get wrong. It is several times slower than the reference
kernel. See the README's "Cost" section for measured numbers.

One deviation from the reference rasterizer: depth sorting is global rather
than per-tile. The reference sorts a (tile, depth) key so each tile
composites in its own depth order; a single global depth order induces the
same ordering *within* every tile, so the composite is equivalent.

Each tile's list is walked in slabs of --tile-slab, carrying transmittance
forward, and stops as soon as transmittance saturates - the reference's
early termination. This replaced a fixed per-tile cap, and the cap was not
a harmless approximation: keeping the nearest N splats is only safe when N
layers suffice to saturate transmittance, and where they did not, the
truncation fell on a tile boundary and left visible 16x16 seams across
every dense region of the render. Early termination on actual transmittance
removes the guess, and bounds memory by the slab rather than by the tile's
length. --max-gaussians-per-tile survives only as a safety bound.

NeRF-Synthetic needs no structure-from-motion. The reference initializes
Blender scenes from 100,000 uniformly random points in a cube of side 2.6,
not from an SfM point cloud, so this pipeline has no COLMAP dependency at
all - the geometry is discovered entirely by densification.

Usage:
    # smoke run: is the machinery alive, and how fast is it here
    uv run --directory training/gaussian-splatting python -u train_gaussians.py \
        --data-dir data --scene lego --iterations 500 --downscale 2 \
        --output-dir runs/smoke

    # the real run
    uv run --directory training/gaussian-splatting python -u train_gaussians.py \
        --data-dir data --scene lego --iterations 30000 \
        --output-dir runs/lego
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --------------------------------------------------------------------------
# Spherical harmonics
#
# View-dependent colour is a degree-3 SH expansion: 16 coefficients per
# colour channel, evaluated in the direction from the camera to the
# Gaussian. Degree 0 alone is a constant (view-independent) colour; the
# higher bands are what let a surface look different from different angles,
# which is how specular highlights on `materials` and `drums` survive.
# These constants are the real-valued SH basis normalizations.
# --------------------------------------------------------------------------

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435,
)


def eval_sh(degree, coefficients, directions):
    """Evaluate an SH colour expansion.

    coefficients is (N, 16, 3) and directions is (N, 3) unit vectors.
    Bands above `degree` are ignored, which is what makes the coarse-to-fine
    degree schedule work: the model fits flat colour first and is only
    allowed view-dependence later, once geometry has settled.
    """
    result = SH_C0 * coefficients[:, 0]
    if degree > 0:
        x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
        result = (result
                  - SH_C1 * y * coefficients[:, 1]
                  + SH_C1 * z * coefficients[:, 2]
                  - SH_C1 * x * coefficients[:, 3])
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                      + SH_C2[0] * xy * coefficients[:, 4]
                      + SH_C2[1] * yz * coefficients[:, 5]
                      + SH_C2[2] * (2.0 * zz - xx - yy) * coefficients[:, 6]
                      + SH_C2[3] * xz * coefficients[:, 7]
                      + SH_C2[4] * (xx - yy) * coefficients[:, 8])
            if degree > 2:
                result = (result
                          + SH_C3[0] * y * (3.0 * xx - yy) * coefficients[:, 9]
                          + SH_C3[1] * xy * z * coefficients[:, 10]
                          + SH_C3[2] * y * (4.0 * zz - xx - yy) * coefficients[:, 11]
                          + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[:, 12]
                          + SH_C3[4] * x * (4.0 * zz - xx - yy) * coefficients[:, 13]
                          + SH_C3[5] * z * (xx - yy) * coefficients[:, 14]
                          + SH_C3[6] * x * (xx - 3.0 * yy) * coefficients[:, 15])
    return result + 0.5


def rgb_to_sh_dc(rgb):
    """Inverse of the degree-0 term: the DC coefficient that renders as rgb."""
    return (rgb - 0.5) / SH_C0


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def quaternion_to_rotation(quaternions):
    """(N, 4) quaternions in (w, x, y, z) order -> (N, 3, 3) rotations.

    Normalized here rather than constrained during optimization: the
    parameter is a free 4-vector and only its direction is meaningful.
    """
    q = quaternions / quaternions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


def covariance_3d(scales, quaternions):
    """Sigma = R S S^T R^T for each Gaussian.

    Factorizing the covariance as a rotation times a diagonal scale is what
    keeps it positive semi-definite for free: any values of the parameters
    give a valid covariance, so the optimizer never has to be projected
    back onto a constraint set.
    """
    rotation = quaternion_to_rotation(quaternions)
    scaled = rotation * scales.unsqueeze(1)          # R @ diag(s)
    return scaled @ scaled.transpose(1, 2)


def project_gaussians(xyz, scales, quaternions, camera, near=0.2):
    """Project 3D Gaussians to screen-space conics (the EWA splat).

    Returns projected pixel centres, the 2x2 inverse covariance packed as
    (a, b, c), a 3-sigma pixel radius, view-space depth, and the visibility
    mask. The maths, in order:

      p_view = W p + t                        world -> camera (OpenCV)
      u      = f x/z + cx,  v = f y/z + cy    perspective divide

    A Gaussian does not stay a Gaussian under perspective projection, so
    3DGS uses the EWA approximation (Zwicker et al. 2001): linearize the
    projection at the Gaussian's centre and push the covariance through
    that linear map.

      J      = [[f/z, 0, -f x/z^2], [0, f/z, -f y/z^2]]
      Sigma' = J W Sigma W^T J^T

    The Jacobian blows up towards the frustum edge, so x/z and y/z are
    clamped to 1.3x the half-FOV tangent before J is built - the same guard
    the reference uses.

    Finally Sigma' gains a +0.3 on its diagonal. That is the low-pass
    filter that keeps a Gaussian smaller than a pixel from falling between
    the samples and flickering; it is why a splat can never shrink below
    roughly half a pixel of screen footprint.
    """
    w2c, focal, cx, cy, width, height = (
        camera["w2c"], camera["focal"], camera["cx"], camera["cy"],
        camera["width"], camera["height"],
    )
    rotation, translation = w2c[:3, :3], w2c[:3, 3]

    p_view = xyz @ rotation.T + translation
    depth = p_view[:, 2]
    visible = depth > near

    z = depth.clamp(min=near)
    limit_x = 1.3 * (0.5 * width / focal)
    limit_y = 1.3 * (0.5 * height / focal)
    x = (p_view[:, 0] / z).clamp(-limit_x, limit_x) * z
    y = (p_view[:, 1] / z).clamp(-limit_y, limit_y) * z

    # NDC is the differentiable handle densification reads: the reference's
    # 0.0002 gradient threshold is calibrated in normalized device
    # coordinates, so keeping the intermediate in NDC makes that number mean
    # here what it means there.
    ndc = torch.stack([
        (focal * x / z + cx) * 2.0 / width - 1.0,
        (focal * y / z + cy) * 2.0 / height - 1.0,
    ], dim=1)
    uv = torch.stack([
        (ndc[:, 0] + 1.0) * width * 0.5,
        (ndc[:, 1] + 1.0) * height * 0.5,
    ], dim=1)

    sigma = covariance_3d(scales, quaternions)
    zeros = torch.zeros_like(z)
    jacobian = torch.stack([
        torch.stack([focal / z, zeros, -focal * x / (z * z)], dim=1),
        torch.stack([zeros, focal / z, -focal * y / (z * z)], dim=1),
    ], dim=1)                                                    # (N, 2, 3)

    transform = jacobian @ rotation                              # (N, 2, 3)
    sigma_2d = transform @ sigma @ transform.transpose(1, 2)     # (N, 2, 2)

    a = sigma_2d[:, 0, 0] + 0.3
    b = sigma_2d[:, 0, 1]
    c = sigma_2d[:, 1, 1] + 0.3

    determinant = a * c - b * b
    visible = visible & (determinant > 1e-9)
    inv_det = 1.0 / determinant.clamp(min=1e-9)
    conic = torch.stack([c * inv_det, -b * inv_det, a * inv_det], dim=1)

    # 3 sigma of the larger principal axis, in pixels.
    mid = 0.5 * (a + c)
    spread = (mid * mid - determinant).clamp(min=0.1).sqrt()
    radius = 3.0 * (mid + spread).clamp(min=0.0).sqrt()

    with torch.no_grad():
        visible = visible & (uv[:, 0] + radius > 0) & (uv[:, 0] - radius < width)
        visible = visible & (uv[:, 1] + radius > 0) & (uv[:, 1] - radius < height)

    return ndc, uv, conic, radius, depth, visible


# --------------------------------------------------------------------------
# Rasterization
# --------------------------------------------------------------------------

def _composite_slab(uv, conic, opacity, colours, index, pixel_x, pixel_y,
                    transmittance):
    """Alpha-composite one depth slab of one chunk of tiles.

    index is (tiles, S) of Gaussian ids ordered front-to-back, -1 padding.
    pixel_x/pixel_y are (tiles, P) pixel coordinates. `transmittance` is
    (tiles, P), what is left un-occluded by everything nearer than this
    slab; the updated value is returned so the next slab can continue.

    For each (pixel, Gaussian) pair the splat's contribution is

        alpha = opacity * exp(-1/2 (d^T Sigma'^-1 d))

    with d the pixel's offset from the splat centre, composited front to
    back:

        C = sum_i c_i alpha_i T_i,   T_i = prod_{j<i} (1 - alpha_j)

    T_i is an *exclusive* cumulative product, and writing it as one lets
    autograd produce the gradient that the reference derives by hand in
    CUDA. Padding entries get alpha = 0, so they contribute nothing and
    leave transmittance untouched.
    """
    valid = index >= 0
    gathered = index.clamp(min=0)

    centres = uv[gathered]                                  # (T, S, 2)
    dx = pixel_x.unsqueeze(2) - centres[:, None, :, 0]      # (T, P, S)
    dy = pixel_y.unsqueeze(2) - centres[:, None, :, 1]

    cone = conic[gathered]                                  # (T, S, 3)
    power = -0.5 * (cone[:, None, :, 0] * dx * dx + cone[:, None, :, 2] * dy * dy) \
        - cone[:, None, :, 1] * dx * dy

    alpha = (opacity[gathered][:, None, :] * torch.exp(power.clamp(max=0.0)))
    alpha = alpha.clamp(max=0.99) * valid[:, None, :]

    inclusive = torch.cumprod(1.0 - alpha, dim=2)
    exclusive = torch.cat(
        [torch.ones_like(inclusive[:, :, :1]), inclusive[:, :, :-1]], dim=2
    )
    weight = alpha * exclusive * transmittance.unsqueeze(2)

    colour = torch.einsum("tpk,tkc->tpc", weight, colours[gathered])
    return colour, transmittance * inclusive[:, :, -1]


def rasterize(uv, conic, opacity, colours, depth, radius, width, height,
              background, tile_size=16, max_per_tile=8192, tile_chunk=64,
              tile_slab=2048, use_checkpoint=True):
    """Tile-bin, depth-sort and composite the visible Gaussians.

    Returns (image (H, W, 3), clipped tile count). Binning is done under
    no_grad - which tile a splat touches is a discrete decision and carries
    no gradient; only the compositing is differentiated.
    """
    device = uv.device
    count = uv.shape[0]
    image = background.expand(height * width, 3).clone()
    if count == 0:
        return image.reshape(height, width, 3), 0

    grid_w = (width + tile_size - 1) // tile_size
    grid_h = (height + tile_size - 1) // tile_size
    tile_total = grid_w * grid_h

    with torch.no_grad():
        # --- which tiles does each splat's 3-sigma box touch ---
        x0 = ((uv[:, 0] - radius) / tile_size).floor().clamp(0, grid_w).long()
        x1 = ((uv[:, 0] + radius) / tile_size).ceil().clamp(0, grid_w).long()
        y0 = ((uv[:, 1] - radius) / tile_size).floor().clamp(0, grid_h).long()
        y1 = ((uv[:, 1] + radius) / tile_size).ceil().clamp(0, grid_h).long()
        span_x = (x1 - x0).clamp(min=0)
        span_y = (y1 - y0).clamp(min=0)
        per_gaussian = span_x * span_y
        pairs = int(per_gaussian.sum())
        if pairs == 0:
            return image.reshape(height, width, 3), 0

        # --- expand to one (tile, gaussian) pair per touched tile ---
        gaussian_id = torch.repeat_interleave(
            torch.arange(count, device=device), per_gaussian
        )
        offsets = torch.cumsum(per_gaussian, 0) - per_gaussian
        within = torch.arange(pairs, device=device) - offsets[gaussian_id]
        widths = span_x[gaussian_id]
        tile_id = ((y0[gaussian_id] + within // widths) * grid_w
                   + (x0[gaussian_id] + within % widths))

        # --- sort by (tile, depth) with one integer key ---
        # depth_rank < count, so tile_id * count + rank is collision-free and
        # stays well inside int64.
        depth_rank = torch.empty(count, dtype=torch.long, device=device)
        depth_rank[torch.argsort(depth)] = torch.arange(count, device=device)
        order = torch.argsort(tile_id * count + depth_rank[gaussian_id])
        tile_id, gaussian_id = tile_id[order], gaussian_id[order]

        # --- safety cap only ---
        # Compositing below terminates a tile as soon as its transmittance
        # saturates, so this bound is not the quality/accuracy knob it would
        # be otherwise - it exists to stop a pathological frame allocating
        # without limit. Tiles that hit it are counted and reported.
        tile_counts = torch.bincount(tile_id, minlength=tile_total)
        starts = torch.cumsum(tile_counts, 0) - tile_counts
        rank = torch.arange(pairs, device=device) - starts[tile_id]
        keep = rank < max_per_tile
        clipped = int((tile_counts > max_per_tile).sum())
        tile_id, gaussian_id, rank = tile_id[keep], gaussian_id[keep], rank[keep]
        tile_counts = tile_counts.clamp(max=max_per_tile)
        starts = torch.cumsum(tile_counts, 0) - tile_counts

        # --- group tiles by occupancy, not by position ---
        # A chunk is padded out to its busiest tile, and neighbouring tiles
        # differ wildly - a tile of empty background sits right next to one
        # covering the object. Chunking by tile id therefore pays for padding
        # that chunking by splat count does not, and that padding is what
        # makes large chunks lose. Sorted this way, a chunk is nearly uniform
        # and the chunk size can be raised until the per-call overhead
        # disappears.
        active = torch.nonzero(tile_counts > 0).squeeze(1)
        by_count = active[torch.argsort(tile_counts[active])]
        slot = torch.full((tile_total,), -1, dtype=torch.long, device=device)
        slot[by_count] = torch.arange(by_count.numel(), device=device)

        reorder = torch.argsort(slot[tile_id] * max_per_tile + rank)
        tile_slot = slot[tile_id][reorder]
        gaussian_id, rank = gaussian_id[reorder], rank[reorder]

        counts_sorted = tile_counts[by_count]
        starts_sorted = torch.cumsum(counts_sorted, 0) - counts_sorted
        counts_cpu = counts_sorted.cpu()
        starts_cpu = starts_sorted.cpu()
        tile_active = by_count.numel()

        offset = torch.arange(tile_size, device=device)

    pixel_index, pixel_colour = [], []

    for begin in range(0, tile_active, tile_chunk):
        end = min(begin + tile_chunk, tile_active)
        lo = int(starts_cpu[begin])
        hi = int(starts_cpu[end - 1] + counts_cpu[end - 1])
        if hi <= lo:
            continue

        with torch.no_grad():
            block = end - begin
            depth_k = int(counts_cpu[begin:end].max())
            index = torch.full((block, depth_k), -1, dtype=torch.long, device=device)
            index[tile_slot[lo:hi] - begin, rank[lo:hi]] = gaussian_id[lo:hi]

            ids = by_count[begin:end]
            px = (ids % grid_w).unsqueeze(1) * tile_size + offset
            py = (ids // grid_w).unsqueeze(1) * tile_size + offset
            pixel_x = px[:, None, :].expand(block, tile_size, tile_size).reshape(block, -1)
            pixel_y = py[:, :, None].expand(block, tile_size, tile_size).reshape(block, -1)
            inside = (pixel_x < width) & (pixel_y < height)
            flat = (pixel_y.clamp(max=height - 1) * width
                    + pixel_x.clamp(max=width - 1))

        px, py = pixel_x.to(uv.dtype), pixel_y.to(uv.dtype)
        pixels = pixel_x.shape[1]
        colour = torch.zeros(block, pixels, 3, device=device, dtype=uv.dtype)
        transmittance = torch.ones(block, pixels, device=device, dtype=uv.dtype)

        # Walk the tile's depth-sorted list in slabs, carrying transmittance
        # forward. Two things fall out of this that a single dense pass does
        # not give: the per-slab intermediates are bounded regardless of how
        # many splats a tile holds, and the loop can stop early.
        for begin_slab in range(0, depth_k, tile_slab):
            piece = index[:, begin_slab:begin_slab + tile_slab]
            args = (uv, conic, opacity, colours, piece, px, py, transmittance)
            if use_checkpoint and torch.is_grad_enabled():
                # A slab's (tiles x pixels x slab) intermediates would
                # otherwise all be held for the backward pass - gigabytes at
                # 800x800. Recomputing one slab's forward during backward
                # trades time for a memory ceiling set by a single slab.
                part, transmittance = checkpoint(_composite_slab, *args,
                                                 use_reentrant=False)
            else:
                part, transmittance = _composite_slab(*args)
            colour = colour + part

            # Everything further back contributes at most `transmittance`, so
            # once that is negligible the rest of the list cannot change the
            # pixel. This is the reference rasterizer's early termination, and
            # it is what makes an uncapped per-tile list affordable - without
            # it, a fixed cap has to guess, and guessing low leaves visible
            # 16x16 tile seams wherever the guess was wrong.
            if float(transmittance.detach().max()) < 1e-4:
                break

        colour = colour + transmittance.unsqueeze(2) * background
        pixel_index.append(flat[inside])
        pixel_colour.append(colour[inside])

    if pixel_index:
        image = image.index_put((torch.cat(pixel_index),), torch.cat(pixel_colour))
    return image.reshape(height, width, 3), clipped


def render(model, camera, background, config, sh_degree):
    """Full forward pass for one camera. Returns the image plus the
    bookkeeping densification needs."""
    ndc, uv, conic, radius, depth, visible = project_gaussians(
        model.xyz, model.scaling, model.rotation, camera, near=config.near_plane
    )
    if ndc.requires_grad:
        # The densification signal is dL/d(NDC position); evaluation runs
        # under no_grad, where there is nothing to retain.
        ndc.retain_grad()

    index = torch.nonzero(visible).squeeze(1)
    if index.numel() == 0:
        blank = background.expand(camera["height"] * camera["width"], 3)
        return {"image": blank.reshape(camera["height"], camera["width"], 3),
                "ndc": ndc, "visible": visible, "radius": radius, "clipped": 0}

    directions = model.xyz[index].detach() - camera["centre"]
    directions = directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    colours = eval_sh(sh_degree, model.features[index], directions).clamp(min=0.0)

    image, clipped = rasterize(
        uv[index], conic[index], model.opacity[index].squeeze(1), colours,
        depth[index], radius[index], camera["width"], camera["height"],
        background, tile_size=config.tile_size,
        max_per_tile=config.max_gaussians_per_tile,
        tile_chunk=config.tile_chunk, tile_slab=config.tile_slab,
        use_checkpoint=bool(config.checkpoint_tiles),
    )
    return {"image": image, "ndc": ndc, "visible": visible,
            "radius": radius, "clipped": clipped}


# --------------------------------------------------------------------------
# The scene
# --------------------------------------------------------------------------

def inverse_sigmoid(x):
    return math.log(x / (1.0 - x))


def mean_squared_distance_to_neighbours(points, k=3, chunk=4096):
    """Mean squared distance to the k nearest other points.

    3DGS initializes each Gaussian's scale to its local point spacing, so
    the initial cloud is roughly space-filling rather than a fog of
    identical blobs. The reference calls a `simple-knn` CUDA extension for
    this; a chunked cdist on the GPU is a few lines and runs once.
    """
    out = torch.empty(points.shape[0], device=points.device)
    for begin in range(0, points.shape[0], chunk):
        block = points[begin:begin + chunk]
        distances = torch.cdist(block, points).pow(2)
        nearest, _ = distances.topk(k + 1, dim=1, largest=False)
        out[begin:begin + chunk] = nearest[:, 1:].mean(dim=1)
    return out


class GaussianScene:
    """The parameters, their optimizer, and the adaptive density control.

    Six tensors, all with N as their first axis. Densification changes N,
    which is the part a naive implementation gets wrong: replacing a
    Parameter drops its Adam moments, and a Gaussian that has just been
    split then starts from zero momentum. The optimizer state is therefore
    grown and pruned in step with the parameters below.
    """

    NAMES = ("xyz", "features_dc", "features_rest", "log_scaling",
             "raw_rotation", "raw_opacity")

    def __init__(self, count, extent, device, max_sh_degree=3, seed=0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        # The reference's Blender initialization: uniform points in a cube of
        # side 2.6 centred on the origin. No structure-from-motion.
        xyz = (torch.rand(count, 3, generator=generator) * 2.6 - 1.3).to(device)
        colours = (torch.rand(count, 3, generator=generator) / 255.0).to(device)

        distances = mean_squared_distance_to_neighbours(xyz).clamp(min=1e-7)
        log_scaling = torch.log(distances.sqrt()).unsqueeze(1).repeat(1, 3)

        rotation = torch.zeros(count, 4, device=device)
        rotation[:, 0] = 1.0

        features = torch.zeros(count, (max_sh_degree + 1) ** 2, 3, device=device)
        features[:, 0] = rgb_to_sh_dc(colours)

        self.max_sh_degree = max_sh_degree
        self.extent = extent
        self.device = device
        self.parameters = {
            "xyz": nn.Parameter(xyz.contiguous()),
            "features_dc": nn.Parameter(features[:, :1].contiguous()),
            "features_rest": nn.Parameter(features[:, 1:].contiguous()),
            "log_scaling": nn.Parameter(log_scaling.contiguous()),
            "raw_rotation": nn.Parameter(rotation.contiguous()),
            "raw_opacity": nn.Parameter(
                torch.full((count, 1), inverse_sigmoid(0.1), device=device)
            ),
        }
        self.optimizer = None
        self.gradient_accum = torch.zeros(count, device=device)
        self.gradient_denom = torch.zeros(count, device=device)
        self.max_radii = torch.zeros(count, device=device)

    # -- activations -------------------------------------------------------
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

    # -- optimizer ---------------------------------------------------------
    def configure_optimizer(self, args):
        # Position steps are scaled by the scene extent so the same numbers
        # work on a scene measured in different units.
        groups = [
            {"params": [self.parameters["xyz"]], "name": "xyz",
             "lr": args.position_lr_init * self.extent},
            {"params": [self.parameters["features_dc"]], "name": "features_dc",
             "lr": args.feature_lr},
            {"params": [self.parameters["features_rest"]], "name": "features_rest",
             "lr": args.feature_lr / 20.0},
            {"params": [self.parameters["log_scaling"]], "name": "log_scaling",
             "lr": args.scaling_lr},
            {"params": [self.parameters["raw_rotation"]], "name": "raw_rotation",
             "lr": args.rotation_lr},
            {"params": [self.parameters["raw_opacity"]], "name": "raw_opacity",
             "lr": args.opacity_lr},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_position_lr(self, step, args):
        """Log-linear decay from position_lr_init to position_lr_final."""
        t = min(max(step / max(args.position_lr_max_steps, 1), 0.0), 1.0)
        lr = math.exp(math.log(args.position_lr_init) * (1 - t)
                      + math.log(args.position_lr_final) * t) * self.extent
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr
        return lr

    def _replace(self, name, tensor, state_transform):
        """Swap one parameter, carrying its Adam moments through the change."""
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old = group["params"][0]
            state = self.optimizer.state.get(old, None)
            new = nn.Parameter(tensor.contiguous().requires_grad_(True))
            if state is not None:
                del self.optimizer.state[old]
                state["exp_avg"] = state_transform(state["exp_avg"])
                state["exp_avg_sq"] = state_transform(state["exp_avg_sq"])
                self.optimizer.state[new] = state
            group["params"][0] = new
            self.parameters[name] = new
            return
        raise KeyError(name)

    def prune(self, keep):
        for name in self.NAMES:
            self._replace(name, self.parameters[name][keep], lambda s: s[keep])
        self.gradient_accum = self.gradient_accum[keep]
        self.gradient_denom = self.gradient_denom[keep]
        self.max_radii = self.max_radii[keep]

    def append(self, extra):
        for name in self.NAMES:
            tensor = torch.cat([self.parameters[name], extra[name]], dim=0)
            self._replace(
                name, tensor,
                # New Gaussians start with zero momentum, not a copy of their
                # parent's: they occupy a different position and their history
                # would be misleading.
                lambda s, e=extra[name]: torch.cat([s, torch.zeros_like(e)], dim=0),
            )
        added = extra["xyz"].shape[0]
        zeros = torch.zeros(added, device=self.device)
        self.gradient_accum = torch.cat([self.gradient_accum, zeros])
        self.gradient_denom = torch.cat([self.gradient_denom, zeros])
        self.max_radii = torch.cat([self.max_radii, zeros])

    # -- adaptive density control -----------------------------------------
    def record_gradients(self, ndc_grad, visible, radii):
        if ndc_grad is None:      # nothing was visible this view
            return
        with torch.no_grad():
            magnitude = ndc_grad[visible, :2].norm(dim=1)
            self.gradient_accum[visible] += magnitude
            self.gradient_denom[visible] += 1.0
            self.max_radii[visible] = torch.maximum(
                self.max_radii[visible], radii[visible]
            )

    def densify_and_prune(self, args, iteration, size_threshold):
        """Clone under-reconstructed Gaussians, split over-reconstructed ones,
        then drop the transparent and the oversized.

        The signal for both is the same: a large screen-space position
        gradient means the Gaussian is being pulled in conflicting
        directions by different pixels, i.e. one primitive is trying to
        explain more detail than it can. What differs is the remedy. If the
        Gaussian is *small*, the region is under-covered and it is cloned
        and left to drift apart. If it is *large*, it is over-covering, and
        it is split into two smaller ones sampled from its own
        distribution.
        """
        with torch.no_grad():
            grads = self.gradient_accum / self.gradient_denom.clamp(min=1.0)
            grads[self.gradient_denom == 0] = 0.0
            selected = grads >= args.densify_grad_threshold
            biggest = self.scaling.max(dim=1).values
            size_cutoff = args.percent_dense * self.extent

            clone_mask = selected & (biggest <= size_cutoff)
            split_mask = selected & (biggest > size_cutoff)

            extra = {name: self.parameters[name][clone_mask] for name in self.NAMES}

            n_split = int(split_mask.sum())
            if n_split:
                repeats = args.split_into
                scales = self.scaling[split_mask].repeat(repeats, 1)
                rotations = quaternion_to_rotation(
                    self.parameters["raw_rotation"][split_mask]
                ).repeat(repeats, 1, 1)
                samples = torch.normal(
                    torch.zeros(n_split * repeats, 3, device=self.device), scales
                )
                positions = (torch.bmm(rotations, samples.unsqueeze(2)).squeeze(2)
                             + self.parameters["xyz"][split_mask].repeat(repeats, 1))
                shrunk = torch.log(scales / (0.8 * repeats))
                for name, value in (
                    ("xyz", positions),
                    ("log_scaling", shrunk),
                    ("raw_rotation",
                     self.parameters["raw_rotation"][split_mask].repeat(repeats, 1)),
                    ("raw_opacity",
                     self.parameters["raw_opacity"][split_mask].repeat(repeats, 1)),
                    ("features_dc",
                     self.parameters["features_dc"][split_mask].repeat(repeats, 1, 1)),
                    ("features_rest",
                     self.parameters["features_rest"][split_mask].repeat(repeats, 1, 1)),
                ):
                    extra[name] = torch.cat([extra[name], value], dim=0)

            n_clone = int(clone_mask.sum())
            if extra["xyz"].shape[0]:
                self.append(extra)

            # The parents of a split are replaced by their children.
            drop = torch.cat([
                split_mask,
                torch.zeros(extra["xyz"].shape[0], dtype=torch.bool, device=self.device),
            ])
            transparent = (self.opacity.squeeze(1) < args.prune_opacity)
            drop = drop | transparent
            if size_threshold is not None:
                drop = drop | (self.max_radii > size_threshold)
                drop = drop | (self.scaling.max(dim=1).values
                               > args.prune_scale_fraction * self.extent)

            self.prune(~drop)
            self.gradient_accum.zero_()
            self.gradient_denom.zero_()
            self.max_radii.zero_()

        return n_clone, n_split, int(drop.sum())

    def reset_opacity(self, value=0.01):
        """Force every Gaussian back down to near-transparent.

        Floaters - Gaussians sitting in empty space that happen to be
        consistent with a few views - otherwise survive indefinitely,
        because nothing pushes their opacity down. Resetting everything and
        letting the genuinely-supported Gaussians climb back up is what
        clears them; the ones with no evidence never recover and are pruned.
        """
        with torch.no_grad():
            capped = torch.minimum(
                self.parameters["raw_opacity"],
                torch.full_like(self.parameters["raw_opacity"], inverse_sigmoid(value)),
            )
        self._replace("raw_opacity", capped, lambda s: torch.zeros_like(s))

    def state_dict(self):
        return {name: self.parameters[name].detach().cpu()
                for name in self.NAMES} | {
            "extent": self.extent, "max_sh_degree": self.max_sh_degree}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def psnr(prediction, target):
    mse = ((prediction - target) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp(min=1e-12)))


def gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.outer(kernel)


def ssim(prediction, target, window, size=11):
    """Structural similarity, per channel, averaged.

    Inputs are (H, W, 3) in [0, 1]. Written out rather than imported from
    pytorch-msssim: it is six local statistics and a ratio, and 3DGS's loss
    depends on it, so it should be visible.
    """
    x = prediction.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    kernel = window.expand(3, 1, size, size)
    pad = size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=3)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=3)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_xx = F.conv2d(x * x, kernel, padding=pad, groups=3) - mu_xx
    sigma_yy = F.conv2d(y * y, kernel, padding=pad, groups=3) - mu_yy
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=3) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return (numerator / denominator).mean()


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def box_downscale(images, factor):
    """Integer box filter. The dataset is 800x800, so 2 and 4 divide evenly."""
    n, h, w, c = images.shape
    return (images.reshape(n, h // factor, factor, w // factor, factor, c)
            .float().mean(dim=(2, 4)).round().clamp(0, 255).to(torch.uint8))


def load_scene(args, device):
    meta = np.load(os.path.join(args.data_dir, f"{args.scene}_meta.npz"),
                   allow_pickle=True)
    height, width = int(meta["height"]), int(meta["width"])
    split = meta["split"]
    total = split.shape[0]

    memmap = np.memmap(os.path.join(args.data_dir, f"{args.scene}_images.u8"),
                       dtype=np.uint8, mode="r", shape=(total, height, width, 3))

    wanted = np.flatnonzero(split != 2)          # train + val; test is for eval
    images = torch.from_numpy(np.ascontiguousarray(memmap[wanted])).to(device)
    if args.downscale > 1:
        if height % args.downscale or width % args.downscale:
            raise SystemExit(
                f"--downscale {args.downscale} does not divide {width}x{height}"
            )
        images = box_downscale(images, args.downscale)
        height //= args.downscale
        width //= args.downscale

    focal = float(meta["focal"]) / args.downscale
    cx = float(meta["cx"]) / args.downscale
    cy = float(meta["cy"]) / args.downscale
    w2c = torch.from_numpy(meta["w2c"][wanted]).to(device)
    c2w = torch.from_numpy(meta["c2w"][wanted]).to(device)

    cameras = [{
        "w2c": w2c[i], "centre": c2w[i, :3, 3], "focal": focal,
        "cx": cx, "cy": cy, "width": width, "height": height,
    } for i in range(len(wanted))]

    local_split = split[wanted]
    return {
        "images": images,
        "cameras": cameras,
        "train": np.flatnonzero(local_split == 0),
        "val": np.flatnonzero(local_split == 1),
        "extent": float(meta["scene_radius"]),
        "width": width, "height": height,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def evaluate_split(scene, data, indices, background, args, sh_degree, window):
    scores = []
    with torch.no_grad():
        for i in indices:
            out = render(scene, data["cameras"][int(i)], background, args, sh_degree)
            target = data["images"][int(i)].float() / 255.0
            scores.append((psnr(out["image"].clamp(0, 1), target),
                           float(ssim(out["image"].clamp(0, 1), target, window))))
    return (float(np.mean([s[0] for s in scores])),
            float(np.mean([s[1] for s in scores])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--scene", default="lego")
    parser.add_argument("--output-dir", default="runs/lego")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--downscale", type=int, default=1,
        help="Integer box downscale of the 800x800 renders. 1 is the "
             "resolution every published NeRF-Synthetic number is measured "
             "at; 2 is roughly 4x faster and is the right setting for a "
             "smoke run, but its PSNR is NOT comparable to the baselines.",
    )
    parser.add_argument("--init-points", type=int, default=100000)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--sh-degree-interval", type=int, default=1000,
        help="Promote one SH band every this many iterations. Starting at "
             "degree 0 makes the model fit geometry before view-dependence, "
             "which stops it explaining away shape with colour.",
    )
    parser.add_argument("--near-plane", type=float, default=0.2)

    # rasterizer
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--max-gaussians-per-tile", type=int, default=8192,
        help="Safety cap on a single tile's splat list. Compositing stops "
             "early once transmittance saturates, so this is not a quality "
             "knob - it only bounds a pathological frame. The log reports "
             "any tile that hits it.",
    )
    parser.add_argument(
        "--tile-chunk", type=int, default=64,
        help="Tiles composited per batched call. Small is better: a chunk is "
             "padded out to its busiest tile, and even count-sorted, a wide "
             "chunk spans a wider range of occupancies. Measured optimum on "
             "an RTX 3090 at 800x800; 16 and 256 are both slower.",
    )
    parser.add_argument(
        "--tile-slab", type=int, default=2048,
        help="Depth-sorted splats composited per slab. Large is better: this "
             "is per-call overhead, not bandwidth - peak memory here is ~1 GiB "
             "of 24. Measured optimum; 128 is 1.4x slower than 2048.",
    )
    parser.add_argument(
        "--checkpoint-tiles", type=int, default=1,
        help="Recompute each slab's forward during backward instead of "
             "holding every slab's intermediates. Costs time, and without it "
             "800x800 does not fit in 24 GB.",
    )

    # losses / schedule
    parser.add_argument("--lambda-ssim", type=float, default=0.2)
    parser.add_argument("--position-lr-init", type=float, default=0.00016)
    parser.add_argument("--position-lr-final", type=float, default=0.0000016)
    parser.add_argument("--position-lr-max-steps", type=int, default=30000)
    parser.add_argument("--feature-lr", type=float, default=0.0025)
    parser.add_argument("--opacity-lr", type=float, default=0.05)
    parser.add_argument("--scaling-lr", type=float, default=0.005)
    parser.add_argument("--rotation-lr", type=float, default=0.001)

    # densification
    parser.add_argument("--densify-from", type=int, default=500)
    parser.add_argument("--densify-until", type=int, default=15000)
    parser.add_argument("--densify-interval", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    parser.add_argument("--percent-dense", type=float, default=0.01)
    parser.add_argument("--split-into", type=int, default=2)
    parser.add_argument("--prune-opacity", type=float, default=0.005)
    parser.add_argument("--prune-screen-radius", type=float, default=20.0)
    parser.add_argument("--prune-scale-fraction", type=float, default=0.1)
    parser.add_argument("--opacity-reset-interval", type=int, default=3000)

    # logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--val-views", type=int, default=10)
    parser.add_argument(
        "--single-view", type=int, default=-1,
        help="Optimize against one training view only, with densification "
             "off. A sanity check for the projection and compositing math: "
             "a correct rasterizer overfits one 800x800 image to >35 dB in a "
             "few hundred iterations. Use -1 for normal training.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device found, this will be extremely slow")

    data = load_scene(args, device)
    print(f"Scene '{args.scene}': {len(data['train'])} train / "
          f"{len(data['val'])} val views at {data['width']}x{data['height']}, "
          f"extent {data['extent']:.4f}")

    scene = GaussianScene(args.init_points, data["extent"], device,
                          max_sh_degree=args.sh_degree, seed=args.seed)
    scene.configure_optimizer(args)
    background = torch.ones(3, device=device)      # white, matching the data
    window = gaussian_window(11, 1.5, device)

    single = args.single_view >= 0
    if single:
        train_indices = np.array([data["train"][args.single_view]])
        print(f"Single-view sanity mode: view {train_indices[0]}, "
              f"densification disabled")
    else:
        train_indices = data["train"]

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    best_psnr = -1.0
    order = np.array([], dtype=np.int64)
    cursor = 0
    running_loss, running_psnr, running_clipped = 0.0, 0.0, 0
    started = time.time()

    for iteration in range(1, args.iterations + 1):
        if cursor >= len(order):
            order = np.random.permutation(train_indices)
            cursor = 0
        view = int(order[cursor])
        cursor += 1

        sh_degree = min(args.sh_degree, iteration // args.sh_degree_interval)
        lr = scene.update_position_lr(iteration, args)

        out = render(scene, data["cameras"][view], background, args, sh_degree)
        target = data["images"][view].float() / 255.0

        l1 = (out["image"] - target).abs().mean()
        similarity = ssim(out["image"], target, window)
        loss = (1.0 - args.lambda_ssim) * l1 + args.lambda_ssim * (1.0 - similarity)

        scene.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            running_loss += float(loss)
            running_psnr += psnr(out["image"].clamp(0, 1), target)
            running_clipped += out["clipped"]

            if not single and iteration < args.densify_until:
                scene.record_gradients(out["ndc"].grad, out["visible"], out["radius"])

        scene.optimizer.step()

        if not single and args.densify_from <= iteration < args.densify_until:
            if iteration % args.densify_interval == 0:
                size_threshold = (args.prune_screen_radius
                                  if iteration > args.opacity_reset_interval else None)
                cloned, split, pruned = scene.densify_and_prune(
                    args, iteration, size_threshold
                )
                print(f"  [densify {iteration:6d}] +{cloned} cloned, "
                      f"+{split} split, -{pruned} pruned -> {scene.count} Gaussians")
            # White backgrounds make floaters especially cheap to keep, so the
            # reference resets opacity once at densify_from as well.
            if (iteration % args.opacity_reset_interval == 0
                    or iteration == args.densify_from):
                scene.reset_opacity()
                print(f"  [opacity reset {iteration:6d}]")

        if iteration % args.log_interval == 0:
            elapsed = time.time() - started
            print(f"iter {iteration:6d}/{args.iterations}  "
                  f"loss {running_loss / args.log_interval:.5f}  "
                  f"psnr {running_psnr / args.log_interval:6.2f}  "
                  f"gaussians {scene.count:>7,}  "
                  f"pos_lr {lr:.2e}  sh {sh_degree}  "
                  f"clipped_tiles {running_clipped / args.log_interval:.1f}  "
                  f"{iteration / elapsed:.2f} it/s")
            running_loss, running_psnr, running_clipped = 0.0, 0.0, 0

        if iteration % args.val_interval == 0 or iteration == args.iterations:
            views = data["val"][:args.val_views] if not single else train_indices
            val_psnr, val_ssim = evaluate_split(
                scene, data, views, background, args, args.sh_degree, window
            )
            print(f"  [val {iteration:6d}] psnr {val_psnr:.3f}  "
                  f"ssim {val_ssim:.4f}  ({len(views)} views)")
            history.append({"iteration": iteration, "val_psnr": val_psnr,
                            "val_ssim": val_ssim, "gaussians": scene.count})
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(
                    {"gaussians": scene.state_dict(), "iteration": iteration,
                     "val_psnr": val_psnr, "val_ssim": val_ssim,
                     "scene": args.scene, "downscale": args.downscale,
                     "config": vars(args)},
                    os.path.join(args.output_dir, "gaussians_best.pt"),
                )

    torch.save(
        {"gaussians": scene.state_dict(), "iteration": args.iterations,
         "scene": args.scene, "downscale": args.downscale, "config": vars(args)},
        os.path.join(args.output_dir, "gaussians_last.pt"),
    )
    with open(os.path.join(args.output_dir, "history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total = time.time() - started
    print(f"\nDone in {total / 60:.1f} min ({args.iterations / total:.2f} it/s), "
          f"{scene.count:,} Gaussians, best val PSNR {best_psnr:.3f}")
    print(f"Saved {args.output_dir}/gaussians_best.pt and gaussians_last.pt")


if __name__ == "__main__":
    main()

# gaussian-splatting

**3D Gaussian Splatting** ([Kerbl et al., SIGGRAPH
2023](https://arxiv.org/abs/2308.04079)) optimized **from scratch** on the
NeRF-Synthetic / Blender scenes — with no splatting library (no `gsplat`,
no `diff-gaussian-rasterization`), no graphics stack (no COLMAP, no
`open3d`, no `plyfile`) and no image library (no Pillow, no `torchvision`).
The PNG decoder and writer, the EWA projection of 3D covariances to
screen-space conics, tile binning and alpha compositing, spherical-harmonic
colour, the adaptive densification with its Adam-state surgery, SSIM and
the binary `.ply` export are all written out by hand in
`build_blender_dataset.py` / `train_gaussians.py` / `evaluate_gaussians.py`;
torch supplies tensor ops, autograd, sorting and GPU execution — the same
role it plays in `training/rvq-audio-codec` and
`training/flow-matching-mnist`.

This is a `training/` pipeline (from-scratch, non-LoRA), an independent `uv`
project like every other pipeline folder in this repo — own
`pyproject.toml` (pinned to the CUDA 12.8 torch build), own
`.python-version`, no shared root environment.

**Why this belongs here.** There is no network and no pretrained anything.
The scene *is* the parameters: a few hundred thousand anisotropic 3D
Gaussians, each with a position, a scale, a rotation, an opacity and 16
spherical-harmonic colour coefficients, optimized directly against
photographs by gradient descent. Every scene is its own optimization from a
random point cloud. It is also the first 3D pipeline in the repo, and the
first whose output you can walk around rather than read off a metric.

## Method

Each Gaussian is a blob in space, `G(x) = exp(-1/2 (x-p)^T Sigma^-1 (x-p))`,
with the covariance factorized as `Sigma = R S S^T R^T` so that any values
of the rotation and scale parameters give a valid covariance and the
optimizer never has to be projected back onto a constraint.

Rendering is a rasterization, not a ray march — which is the entire reason
3DGS displaced NeRF:

1. **Project.** A Gaussian does not stay a Gaussian under perspective
   division, so the projection is linearized at the Gaussian's centre and
   the covariance pushed through that linear map (the EWA splat, Zwicker et
   al. 2001): `Sigma' = J W Sigma W^T J^T`. `Sigma'` gets `+0.3` on its
   diagonal — a low-pass filter that stops a sub-pixel splat falling
   between samples and flickering.
2. **Bin and sort.** Each splat's 3-sigma box is assigned to the 16x16
   tiles it touches, and the (tile, Gaussian) pairs are sorted by depth.
3. **Composite.** Front to back, `C = sum_i c_i a_i T_i` with
   `T_i = prod_{j<i} (1 - a_j)`, where `a_i` is the splat's opacity
   attenuated by the conic evaluated at that pixel.

Colour is view-dependent: a degree-3 spherical-harmonic expansion evaluated
along the camera-to-Gaussian direction, so a surface can look different
from different angles. The degree is raised one band every 1,000 iterations
starting from 0, which forces the model to fit geometry before it is
allowed to explain shape away with colour.

### Adaptive density control

Starting from random points, the geometry is discovered entirely by
densification. Every 100 iterations between 500 and 15,000, the
screen-space position gradient — accumulated per Gaussian across the views
it was visible in — decides what happens:

| condition | action | why |
|---|---|---|
| large gradient, **small** Gaussian | **clone** it | the region is under-reconstructed; two Gaussians start together and drift apart |
| large gradient, **large** Gaussian | **split** into 2 | one primitive is over-covering; children are sampled from the parent's own distribution, scales divided by 1.6 |
| opacity < 0.005 | **prune** | it is contributing nothing |
| screen radius > 20 px, or scale > 10% of the scene | **prune** | it has become a sky-filling blob |
| every 3,000 iterations | **reset all opacity to 0.01** | floaters that no view contradicts otherwise survive forever; everything with real support climbs back, everything without is pruned |

A large position gradient means one Gaussian is being pulled in conflicting
directions by different pixels — i.e. it is trying to explain more detail
than one primitive can. Whether the fix is to clone or to split is decided
purely by its size relative to the scene.

**The part that is easy to get wrong** is not the criterion but the
plumbing: densification changes the *number* of Gaussians, and simply
building new `nn.Parameter`s throws away the Adam moments, so every
freshly-split Gaussian restarts from zero momentum and the run quietly
loses quality. `GaussianScene._replace` grows and prunes the optimizer's
`exp_avg` / `exp_avg_sq` in lockstep with the parameters.

### No structure-from-motion

The reference implementation initializes real captures from a COLMAP point
cloud, but initializes **Blender scenes from 100,000 uniformly random
points in a cube of side 2.6**. This pipeline targets Blender scenes, so it
has no SfM dependency at all — the geometry comes entirely from
densification. That is what makes an 800-line from-scratch implementation
feasible.

## The rasterizer is plain PyTorch

The reference ships a CUDA rasterizer with a hand-derived backward pass.
Here the compositing is written as an **exclusive cumulative product**,
which autograd differentiates on its own — so there is no backward pass to
get wrong, and the projection and compositing math stays readable. That is
the trade this folder exists to make, and it costs speed.

Two deliberate deviations, both visible rather than hidden:

- **Depth sorting is global, not per-tile.** A single global depth order
  induces the same ordering *within* every tile, so the composite is
  equivalent.
- **Slabs are gradient-checkpointed** (`--checkpoint-tiles 1`, default).
  The `(tiles x pixels x slab)` intermediates for a whole 800x800 frame do
  not fit in 24 GB otherwise; recomputing one slab's forward during
  backward trades time for a memory ceiling set by a single slab.

### Why the tile list is walked in slabs

Each tile's depth-sorted splat list is composited `--tile-slab` at a time,
carrying transmittance forward, and the loop **stops as soon as
transmittance saturates** — the reference rasterizer's early termination.

This replaced a fixed per-tile cap, and the cap was not a harmless
approximation. Keeping the nearest N splats per tile is only safe when N
layers are enough to saturate transmittance; where they were not, the
truncation landed on a tile boundary and left **visible 16x16 seams** in
the render — a grid of them across every dense region. Early termination on
the actual accumulated transmittance removes the guess: nothing is dropped
that could still change a pixel, and memory stays bounded by the slab
rather than by the tile's length. `--max-gaussians-per-tile` survives only
as a safety bound on a pathological frame, and the log reports any tile
that reaches it.

The lesson generalizes past this folder: a cap chosen for memory reasons
that happens to align with a spatial partition does not degrade gracefully.
It degrades *along the partition*, which is exactly where the eye looks.

## Cost

Measured on an RTX 3090, `lego`, batch of one view per iteration:

| | 400x400 | 800x800 |
|---|---|---|
| initialization (100k diffuse Gaussians, worst case) | 2.68 it/s | 1.00 it/s |
| steady state (64k Gaussians, partly converged) | 3.83 it/s | — |
| peak memory | 1.2 GiB | 1.5 GiB |
| 30,000 iterations | ~2–3 h | ~6–8 h |

The reference CUDA rasterizer does the same scene in 5–10 minutes. **That
40–60x gap is the trade this folder exists to make**, and it is worth being
explicit about: every pixel-splat pair here becomes an element of a
materialized tensor rather than living in a register, so the projection and
compositing stay readable and autograd derives the backward — at 40x the
cost. If you want the speed, the honest move is gsplat's kernels with
everything else still written by hand, not micro-optimizing this.

Two things were measured rather than assumed while tuning it, both of which
went against intuition:

- **Smaller tiles are much slower, not faster.** The reasoning that a
  tensor rasterizer has no shared-memory reuse to amortize — so it should
  prefer small tiles that waste fewer pixel evaluations — predicts 4x4
  should win. It loses by 6x, because the per-call overhead of many more
  chunks swamps the saved arithmetic. 16x16 stays.
- **Tiles are chunked by occupancy, not by position.** A chunk is padded
  out to its busiest tile, and a tile of empty background sits right next
  to one covering the object, so position-chunking pays for padding that
  count-chunking does not. Sorting tiles by splat count before chunking was
  worth **5x** on its own — from 0.74 to 3.83 it/s — and it is the single
  reason the numbers above are usable at all.

`--tile-chunk 64` and `--tile-slab 2048` are the measured optimum; peak
memory is ~1.5 GiB of 24, so this is bound by per-call overhead, not
bandwidth. Raising the chunk past 128 costs memory quadratically for no
speed.

## Data

[NeRF-Synthetic / Blender](https://www.matthewtancik.com/nerf): 8
path-traced scenes, each 100 train / 100 val / 200 test renders at 800x800
RGBA, with a shared FOV and per-frame camera-to-world matrices. Not
included here — download it and point `--data-dir` at the extracted folder
(a nested `nerf_synthetic/` subfolder is also accepted).

`build_blender_dataset.py` verifies **exactly 100 / 100 / 200** renders per
split and refuses to build on a partial extraction — the same guardrail as
`build_ljspeech_dataset.py`'s 13,100-wav check. Note the `test/` folder
holds 600 files: each render ships with `_depth_` and `_normal_` maps this
pipeline does not use.

Two things the builder does that are load-bearing:

- **Alpha is composited over white.** Every published NeRF-Synthetic number
  is measured that way. Changing it silently makes this pipeline's PSNR
  incomparable to every baseline in the literature.
- **Cameras are converted from Blender/OpenGL to OpenCV convention**
  (negate the second and third basis columns, then invert). Getting this
  wrong yields an upside-down mirrored scene that still produces a
  plausible-looking loss curve. The conversion is checked, not assumed:
  `det(R) = +1` (a wrong flip gives −1, a mirror), and every camera's +Z
  axis points at the origin to within a float rounding error.

The PNG decoder is worth a look. PNG picks one of five filters per
scanline, and on this dataset the two with intra-row byte dependencies
dominate (~26% Average, ~38% Paeth), so a byte-at-a-time Python loop over
3,200 images is hopeless. The loop is inverted instead: images are decoded
in batches, and since rows depend on the previous row of the *same* image
but never on another image, one numpy operation advances every image in the
batch by one pixel. Sub collapses to a cumulative sum and needs no loop at
all. Measured: 4.0 s/image at batch 1, **0.17 s/image at batch 100** — 68 s
for a whole scene.

Storage is a memmapped `data/<scene>_images.u8` plus a small
`data/<scene>_meta.npz` of camera matrices and intrinsics, rather than an
`.npz` of pixels: one scene is 768 MB as uint8 and all eight are 6.1 GB.

## Commands

Three, like every other pipeline in this repo.

```bash
uv run --directory training/gaussian-splatting python build_blender_dataset.py --data-dir "C:\path\to\nerf_synthetic" --scene lego --output-dir data
```

```bash
uv run --directory training/gaussian-splatting python -u train_gaussians.py --data-dir data --scene lego --iterations 30000 --output-dir runs/lego
```

```bash
uv run --directory training/gaussian-splatting python evaluate_gaussians.py --data-dir data --scene lego --checkpoint-path runs/lego/gaussians_best.pt --output-dir runs/lego --orbit-frames 60
```

The evaluator writes `metrics.json`, `render_NN.png` / `gt_NN.png` pairs, an
optional turntable sequence on poses that appear in no split, and
`lego.ply`.

**The `.ply` is the point.** It is written in the layout the reference
implementation uses, so it loads directly in
[SuperSplat](https://superspl.at/editor) or
[antimatter15's viewer](https://antimatter15.com/splat/) — drag it in and
fly around the scene in real time. The stored values are the *raw*
parameters (log scales, logit opacity, unnormalized quaternion); viewers
apply the activations themselves, and writing activated values produces a
scene that loads but looks washed out and the wrong size.

To build and train all eight and print the comparison table:

```bash
uv run --directory training/gaussian-splatting python build_blender_dataset.py --data-dir "C:\path\to\nerf_synthetic" --scene all --output-dir data
```

```bash
uv run --directory training/gaussian-splatting python evaluate_gaussians.py --compare runs
```

## How to judge a run

**PSNR on the 200 test views, against the published table.** Unlike most
pipelines in this repo, this one has a well-established baseline for
exactly this data, so `--compare` prints ours next to the 3DGS paper's and
NeRF's per-scene numbers. That is the primary check.

Two caveats it enforces rather than trusts:

- **Only `--downscale 1` counts.** A smaller image is an easier target;
  `--compare` ignores any `metrics.json` rendered at a downscale.
- **Look at the renders and the orbit, not just the mean.** A metric
  averaged over 200 test views hides floaters that only appear from angles
  no camera took. The orbit path exists to find them.

There is deliberately **no LPIPS**, though the 3DGS paper reports it: it
needs a pretrained VGG/AlexNet, and a from-scratch folder that quietly
downloads an ImageNet backbone to score itself is not from-scratch. Same
rule that keeps FID out of `training/flow-matching-mnist` and
ViSQOL/PESQ/NISQA out of `training/rvq-audio-codec`.

## Verified runs

_(filled in from real runs on the repo owner's RTX 3090 — see
`ARCHITECTURE.md` Stage 4)_

## Relation to the rest of the repo

The successor in spirit to `training/flow-matching-mnist`: both are
contemporary methods that displaced a previous standard (rectified flow
over DDPM; 3DGS over NeRF), both are written out rather than imported, and
both refuse a pretrained-network metric. Where every other pipeline here
learns a function that generalizes across inputs, this one fits a single
scene — the parameters have no meaning away from it, which is what "per-
scene optimization" means and why the run is measured in minutes rather
than the hours a dataset-scale model takes.

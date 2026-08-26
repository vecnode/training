"""
A class-conditional Diffusion Transformer (DiT, Peebles & Xie 2022,
arXiv:2212.09748 - the architecture Sora is built on) trained **from
scratch** on CIFAR-100 - no diffusion/flow/transformer library (no
`diffusers`, no `torchcfm`, no `torchdiffeq`, no `transformers`, no `timm`,
no `torchvision`). The patch embedding, the fixed 2D sincos positional
embedding, the adaLN-Zero transformer blocks, the hand-written multi-head
self-attention (QKV projections, scaled dot-product, output projection),
the final unpatchify layer, the conditional-OT probability path, the
velocity-regression loss, the class embedding + classifier-free guidance,
the EMA, and the Euler ODE sampler are all written out below as plain torch
modules / functions; torch is used only for tensor ops, autograd, and GPU
execution, the same role it plays in every other `training/` pipeline (no
`DataLoader`, numpy-permutation batching).

The objective is the same conditional optimal-transport flow matching as
`training/flow-matching-mnist` (Lipman et al., arXiv:2210.02747 with the
independent coupling; at the default `--sigma-min 0.0` exactly the
rectified flow of Liu et al., arXiv:2209.03003), now conditioned on the
CIFAR-100 fine class label:

    t   ~ U(0, 1)                            one t per sample
    x0  ~ N(0, I)
    x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
    u   = x1 - (1 - sigma_min) * x0          the path's velocity
    L   = || v_theta(x_t, t, y) - u ||^2     plain MSE, y = the class

Conditioning: an embedding table maps the 100 fine classes (+ one null
token used for classifier-free guidance) into the transformer's dim
conditioning space, where it is added to the 256-dim sinusoidal time
embedding and fed through every block's adaLN-Zero modulation MLPs
(scale/shift/gate for each norm, zero-initialized so every block starts as
the identity); the final layer modulates with shift/scale only, per the
paper. CFG is trained by dropping the class to the null token with
probability `--cfg-dropout` (0.1, the DiT paper's value); at sample time
the guided velocity is

    v_cfg = v_uncond + cfg * (v_cond - v_uncond)

and sampling is an ODE solve (Euler): start x0 ~ N(0,I) at t=0 and
integrate dx/dt = v_theta(x, t, y) to t=1 (see `sample_ode` here and
`evaluate_dit.py`).

Pixels arrive from build_cifar100_dataset.py in [0,1] and are rescaled
here to [-1,1] so the data sits on the same scale as the N(0,I) end of the
path, exactly like `training/flow-matching-mnist`.

Usage:
    uv run --directory training/dit-cifar100 python train_dit.py \
        --data-path data/cifar100.npz \
        --patch-size 2 --dim 256 --depth 8 --heads 8 \
        --num-epochs 60 --batch-size 256 \
        --output-dir runs/dit_cifar100
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
NUM_CLASSES = 100
NULL_CLASS = 100  # the extra embedding slot: CFG's "no class" token
FREQ_EMB_DIM = 256  # sinusoidal time-embedding width (the paper's TimestepEmbedder)


# ---------------------------------------------------------------------------
# Positional / time embeddings
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t, dim):
    """Transformer-style positional encoding of the continuous time t, so
    the network can tell noise levels apart at several frequencies. t is in
    [0,1] here (not an integer step index as in DDPM), so it is scaled by
    1000 to put the resulting frequencies in the same useful band those
    implementations use - identical to training/flow-matching-mnist."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().view(-1, 1) * 1000.0 * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000.0 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """(grid_size**2, embed_dim) fixed 2D sincos embedding, the MAE recipe
    the DiT paper uses for the positional embedding (learned position
    embeddings were ablated *worse* in the paper, so this one is frozen)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)          # here w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    return np.concatenate([emb_h, emb_w], axis=1)


# ---------------------------------------------------------------------------
# The DiT (Sora-style) blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Conv2d patch embed: 32x32 / patch_size^2 tokens of dim each."""

    def __init__(self, patch_size, in_channels, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (IMAGE_SIZE // patch_size) ** 2

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, T, dim)


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention written out by hand: one QKV
    projection, reshaped to (heads, head_dim), softmax attention, output
    projection. No nn.MultiheadAttention - the math stays visible (same
    module as training/vit-cifar10, copied in by hand)."""

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class DiTBlock(nn.Module):
    """adaLN-Zero block: every norm is modulated by scale/shift and every
    residual branch by a gate, all produced by one MLP from the combined
    time+class conditioning vector. Zero-initializing that MLP's output
    makes each block start as the identity - the "Zero" in adaLN-Zero, and
    the init trick that lets deep DiTs train from scratch."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),   # the paper's approx-gelu
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),   # shift/scale/gate x norm1, norm2
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )
        return x


class FinalLayer(nn.Module):
    """Final adaLN-modulated norm + linear that unpatchifies the tokens
    back to pixels. Per the paper this layer modulates with shift/scale
    only (no gate - the blocks carry the gates) and the linear is
    zero-initialized, so the network starts as "no movement" (velocity ~
    0), which stabilises the first few hundred steps."""

    def __init__(self, dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),   # shift/scale for the final norm
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.linear(
            self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        return x


class DiT(nn.Module):
    """v_theta(x, t, y): maps a point on the path plus its time and class to
    a velocity of the same shape as the image (the output *is* the
    velocity, not a noise prediction). Sora-style: patch embed + fixed 2D
    sincos pos embed + adaLN-Zero blocks + final unpatchify."""

    def __init__(self, patch_size=2, dim=256, depth=8, heads=8, mlp_ratio=4.0,
                 in_channels=3, num_classes=NUM_CLASSES):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (IMAGE_SIZE // patch_size) ** 2
        self.in_channels = in_channels

        self.x_embedder = PatchEmbed(patch_size, in_channels, dim)
        # Fixed (frozen) 2D sincos positional embedding, the paper's choice.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, dim), requires_grad=False
        )
        # Time embedder (the paper's TimestepEmbedder): 256-dim sinusoidal
        # -> MLP(256 -> dim -> dim).
        self.t_embedder = nn.Sequential(
            nn.Linear(FREQ_EMB_DIM, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Class embedder: 100 classes + 1 null token (CFG), into dim so it
        # can be added to the time embedding before the blocks.
        self.y_embedder = nn.Embedding(num_classes + 1, dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(dim, patch_size, in_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # The paper's recipe: xavier the Linears, freeze the sincos pos
        # embed, and zero the adaLN-Zero modulation + final outputs so the
        # network starts as (approximately) the identity velocity field.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)

        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x):
        B, T, _ = x.shape
        p = self.patch_size
        h = w = IMAGE_SIZE // p
        x = x.reshape(B, h, w, p, p, self.in_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_channels, IMAGE_SIZE, IMAGE_SIZE)

    def forward(self, x, t, y):
        c = self.t_embedder(sinusoidal_time_embedding(t, FREQ_EMB_DIM))
        c = c + self.y_embedder(y)                        # (B, dim)
        h = self.x_embedder(x) + self.pos_embed           # (B, T, dim)
        for block in self.blocks:
            h = block(h, c)
        return self.unpatchify(self.final_layer(h, c))


# ---------------------------------------------------------------------------
# EMA, pixel scale, the conditional-OT path, sampling
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of the weights, kept alongside the live
    model and used for sampling. Written out here rather than imported;
    without it the samples of a short flow-matching run are visibly noisier,
    because the last-step weights sit wherever SGD noise left them."""

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


def to_model_scale(x01):
    """[0,1] pixels -> [-1,1], the scale the N(0,I) end of the path lives on."""
    return x01 * 2.0 - 1.0


def to_pixel_scale(x):
    """[-1,1] -> [0,1], clamped, for writing images out."""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def flow_matching_loss(model, x1, y, sigma_min, generator=None):
    """The whole objective, conditioned on the class: regress the
    conditional-OT velocity with plain MSE (same path as
    training/flow-matching-mnist, with y threaded through the model)."""
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    t_broadcast = t.view(-1, 1, 1, 1)
    x_t = (1.0 - (1.0 - sigma_min) * t_broadcast) * x0 + t_broadcast * x1
    u = x1 - (1.0 - sigma_min) * x0
    return F.mse_loss(model(x_t, t, y), u)


def apply_class_dropout(y, null_class, prob, generator=None):
    """Classifier-free guidance's training trick: with probability prob,
    replace the class with the null token, so the model learns both the
    conditional and the unconditional velocity field."""
    if prob <= 0:
        return y
    drop = torch.rand(y.shape[0], device=y.device, generator=generator) < prob
    return torch.where(drop, torch.full_like(y, null_class), y)


@torch.no_grad()
def guided_velocity(model, x, t, y, cfg_scale):
    """v_cfg = v_uncond + cfg * (v_cond - v_uncond). cfg_scale=1.0 skips
    the second (unconditional) forward entirely."""
    v_cond = model(x, t, y)
    if cfg_scale == 1.0:
        return v_cond
    y_null = torch.full_like(y, NULL_CLASS)
    v_uncond = model(x, t, y_null)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_ode(model, classes, num_steps, cfg_scale, sigma_min, device, seed):
    """x0 ~ N(0,I) at t=0, integrated forward to t=1 by Euler steps of the
    guided velocity field. classes: (num_samples,) int64 tensor in 0..99.
    Returns pixels in [0,1]."""
    num = classes.shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(num, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, generator=generator)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((num,), step * dt, device=device, dtype=torch.float32)
        x = x + guided_velocity(model, x, t, classes, cfg_scale) * dt
    return to_pixel_scale(x)


# ---------------------------------------------------------------------------
# Batching, augmentation, the shared train/val loop
# ---------------------------------------------------------------------------

def iterate_batches(X, y, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - one permutation
    per epoch, sliced into batches, so the batching logic stays visible."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def augment_batch(x, pad=4):
    """Random horizontal flip + 4px zero-pad + random crop back to 32x32 -
    all plain torch ops, applied per batch to [0,1] pixels (the same
    augmentation training/mae-cifar100 uses)."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    if flip.any():
        x = torch.where(flip[:, None, None, None], torch.flip(x, dims=[3]), x)
    x = F.pad(x, (pad, pad, pad, pad))                      # (B, 3, 40, 40)
    offs = torch.randint(0, 2 * pad + 1, (B, 2), device=x.device)
    out = torch.empty_like(x[:, :, :IMAGE_SIZE, :IMAGE_SIZE])
    for i in range(B):
        h, w = int(offs[i, 0]), int(offs[i, 1])
        out[i] = x[i, :, h:h + IMAGE_SIZE, w:w + IMAGE_SIZE]
    return out


def run_epoch(model, X, y, batch_size, sigma_min, cfg_dropout, device, rng,
              optimizer=None, ema=None, augment=False, eval_seed=None):
    """One pass over (X, y). If optimizer is given, trains (shuffled, grad
    updates, flip+crop augmentation, class dropout); otherwise evaluates
    (no shuffle, no grad, no dropout) - shared loop so train/val accounting
    can't drift apart. On the eval side the t/noise draws come from a
    generator re-seeded with eval_seed every call, so the reported val loss
    is comparable across epochs instead of moving with the luck of the
    draw. Returns the mean velocity MSE."""
    training = optimizer is not None
    model.train(training)

    generator = None
    if eval_seed is not None:
        generator = torch.Generator(device=device).manual_seed(eval_seed)

    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for xb, yb in iterate_batches(X, y, batch_size, rng, shuffle=training):
            x1 = torch.from_numpy(xb).to(device)            # (B, 3, 32, 32) [0,1]
            if training and augment:
                x1 = augment_batch(x1)
            x1 = to_model_scale(x1)
            yt = torch.from_numpy(yb).to(device)
            if training and cfg_dropout > 0:
                yt = apply_class_dropout(yt, NULL_CLASS, cfg_dropout, generator)
            loss = flow_matching_loss(model, x1, yt, sigma_min, generator=generator)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def make_lr_lambda(warmup_epochs, total_epochs, eta_min_ratio=0.01):
    """Hand-written linear-warmup-then-cosine LR multiplier: ramps 0 -> 1
    over the warmup epochs, then cosine-anneals down to eta_min_ratio - the
    same schedule training/vit-cifar10 and training/mae-cifar100 use;
    warmup is the part transformers need to train stably from scratch."""

    def lr_lambda(epoch):  # 0-based epoch from the scheduler
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return lr_lambda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/cifar100.npz")
    parser.add_argument("--output-dir", default="runs/dit_cifar100")
    parser.add_argument("--patch-size", type=int, default=2,
                        help="Side length of each patch: 32x32 / patch^2 "
                             "tokens. Default 2 -> 16x16 = 256 tokens, the "
                             "same token count as DiT-S/4 at 256px.")
    parser.add_argument("--dim", type=int, default=256,
                        help="Transformer embedding dimension.")
    parser.add_argument("--depth", type=int, default=8,
                        help="Number of adaLN-Zero transformer blocks.")
    parser.add_argument("--heads", type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="MLP hidden size as a multiple of --dim.")
    parser.add_argument("--sigma-min", type=float, default=0.0,
                        help="Conditional-OT path parameter (Lipman et al.). "
                             "0.0 = rectified flow: x_t = (1-t)*x0 + t*x1, "
                             "u = x1 - x0 (same flag as flow-matching-mnist).")
    parser.add_argument("--cfg-dropout", type=float, default=0.1,
                        help="Probability of dropping the class to the null "
                             "token during training (the DiT paper's 0.1); "
                             "this is what makes classifier-free guidance "
                             "possible at sample time.")
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW weight decay (the ViT-family default).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear LR warmup before the cosine anneal.")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay for the sampling weights; 0 disables EMA.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable flip+crop augmentation on training "
                             "batches (A/B of what augmentation is worth).")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training rows held out for "
                             "validation (the CIFAR-100 test split stays "
                             "fully unseen).")
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Save a class-conditional sample grid every N "
                             "epochs (fixed latents + classes, so progress "
                             "is visible across training).")
    parser.add_argument("--sample-cfg", type=float, default=2.0,
                        help="CFG scale used for the training-time sample grids.")
    parser.add_argument("--sample-steps", type=int, default=40,
                        help="Euler steps used for the training-time sample grids.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train_full = data["X_train"]   # (n, 3, 32, 32) float32 in [0,1]
    y_train_full = data["y_train"]
    fine_names = [str(n) for n in data["fine_label_names"]]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"patch={args.patch_size}  dim={args.dim}  depth={args.depth}  "
          f"heads={args.heads}  mlp_ratio={args.mlp_ratio}  "
          f"sigma_min={args.sigma_min}  cfg_dropout={args.cfg_dropout}  "
          f"augment={not args.no_augment}")

    model = DiT(
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=NUM_CLASSES,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_lambda(args.warmup_epochs, args.num_epochs),
    )
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "dit_best.pt")
    final_path = os.path.join(args.output_dir, "dit_final.pt")
    ckpt_meta = {
        "arch": "dit_cifar100",
        "patch_size": args.patch_size,
        "dim": args.dim,
        "depth": args.depth,
        "heads": args.heads,
        "mlp_ratio": args.mlp_ratio,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "sigma_min": args.sigma_min,
        "cfg_dropout": args.cfg_dropout,
    }

    def checkpoint(epoch, val_loss):
        return {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "val_loss": val_loss,
            **ckpt_meta,
        }

    # Training-time sample grid: 6 classes x 4 fixed latents, so both the
    # class conditioning and the sample quality are visible while training.
    sample_classes = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    sample_class_ids = torch.tensor(
        np.repeat(sample_classes, 4), dtype=torch.long, device=device
    )

    def save_sample_grid(epoch):
        model.eval()
        samples = sample_ode(
            model, sample_class_ids, args.sample_steps, args.sample_cfg,
            args.sigma_min, device, seed=args.seed,
        )
        model.train()
        from png_utils import make_rgb_grid, write_png
        grid = make_rgb_grid(list(samples), grid_cols=4)
        path = os.path.join(args.output_dir, f"samples_epoch{epoch:04d}.png")
        write_png(path, grid)
        row_names = "  |  ".join(
            f"row {r}: {fine_names[c]}" for r, c in enumerate(sample_classes)
        )
        print(f"  saved {path}  ({row_names})")

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(
            model, X_train, y_train, args.batch_size, args.sigma_min,
            args.cfg_dropout, device, rng, optimizer=optimizer, ema=ema,
            augment=not args.no_augment,
        )
        scheduler.step()
        val_loss = run_epoch(
            model, X_val, y_val, args.batch_size, args.sigma_min, 0.0, device,
            rng, optimizer=None, eval_seed=args.seed,
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            elapsed = time.time() - start_time
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: mse={train_loss:.4f}  val: mse={val_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"[{elapsed:6.1f}s]"
            )

        if epoch % args.sample_every == 0 or epoch == args.num_epochs:
            save_sample_grid(epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint(epoch, val_loss), best_path)

    torch.save(checkpoint(args.num_epochs, val_loss), final_path)
    print(f"Trained {args.num_epochs} epochs in {time.time() - start_time:.1f}s")
    print(f"Saved best checkpoint (val_mse={best_val_loss:.4f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_dit.py against the held-out test split next.")


if __name__ == "__main__":
    main()

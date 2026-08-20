"""
A flow-matching (rectified-flow) generative model, trained from scratch on
MNIST - no diffusion/flow library (no diffusers, no torchcfm, no
torchdiffeq). The UNet velocity field, the sinusoidal time embedding, the
probability path, the loss, and the EMA are all written out below as plain
torch modules / functions; torch is used only for tensor ops, autograd, and
GPU execution (the same role it plays in training/mnist-vae), not for the
method itself.

The method, in full:

  Flow matching learns a time-dependent velocity field v(x, t) whose ODE
      dx/dt = v(x, t)
  transports the noise distribution p_0 = N(0, I) at t=0 into the data
  distribution p_1 at t=1. Training never solves that ODE. Instead, for a
  single data point x1 and a single noise draw x0, the *conditional* path
  between them is chosen to be a straight line, and the velocity along that
  line is regressed directly:

      t   ~ U(0, 1)                       one t per sample
      x0  ~ N(0, I)
      x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
      u   = x1 - (1 - sigma_min) * x0     the path's velocity, constant in t
      L   = || v_theta(x_t, t) - u ||^2   plain MSE

  That is Lipman et al.'s conditional optimal-transport path
  (arXiv:2210.02747) with the independent coupling; at the default
  sigma_min = 0 it is exactly the rectified flow of Liu et al.
  (arXiv:2209.03003): x_t = (1-t)*x0 + t*x1 and u = x1 - x0.

  Note what is *absent* compared to a DDPM: no noise schedule (no betas, no
  alpha_bar table), no variance parameterization, no ELBO, no
  reweighting term. The marginal-velocity target is recovered in
  expectation because E[u | x_t, t] is the marginal field - so regressing
  the conditional velocity with plain MSE is enough. Sampling is an ODE
  solve, not an ancestral chain (see evaluate_flow.py).

The model - a small UNet (base-channel width x (1, 2, 4)):

    stem Conv(1->C)                                            28x28
    ResBlock(C)                                   -> skip s1   28x28
    Downsample(C->2C)                                          14x14
    ResBlock(2C)                                  -> skip s2   14x14
    Downsample(2C->4C)                                          7x7
    ResBlock(4C) -> SelfAttention(4C) -> ResBlock(4C)            7x7
    Upsample(4C->2C), ResBlock(cat with s2 -> 2C)              14x14
    Upsample(2C->C),  ResBlock(cat with s1 -> C)               28x28
    GroupNorm -> SiLU -> Conv(C->1)                            28x28

  t enters through a sinusoidal embedding + MLP, added as a per-channel
  bias inside every ResBlock. The output has the same shape as the input:
  it *is* the velocity, not a noise prediction.

Pixels arrive from build_mnist_dataset.py in [0,1] and are rescaled here to
[-1,1] so the data sits on the same scale as the N(0,I) end of the path.

Usage:
    uv run --directory training/flow-matching-mnist python train_flow.py \
        --data-path data/mnist.npz \
        --base-channels 32 \
        --num-epochs 40 \
        --batch-size 128 \
        --output-dir runs/mnist_flow
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels):
    """GroupNorm with 8 groups where the width allows it - the usual choice
    in diffusion/flow UNets (BatchNorm interacts badly with the per-sample
    random t, since a batch mixes wildly different noise levels)."""
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


def sinusoidal_time_embedding(t, dim):
    """Transformer-style positional encoding of the continuous time t, so
    the network can tell noise levels apart at several frequencies.

    t is in [0,1] here (not an integer step index as in DDPM), so it is
    scaled by 1000 to put the resulting frequencies in the same useful band
    those implementations use.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().view(-1, 1) * 1000.0 * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    """Pre-activation residual block with the time embedding added as a
    per-channel bias between the two convolutions."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(temb)).view(h.shape[0], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Single-head self-attention over the 7x7 = 49 spatial positions of the
    bottleneck, written out with two bmm calls. Cheap at this resolution,
    and it is what lets distant strokes of a digit agree with each other."""

    def __init__(self, channels):
        super().__init__()
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        scores = torch.bmm(q.transpose(1, 2), k) * (c ** -0.5)   # (B, N, N)
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Upsample(nn.Module):
    """Nearest-neighbour x2 followed by a 3x3 conv - avoids the checkerboard
    artifacts a bare ConvTranspose2d produces on generated (as opposed to
    reconstructed) images."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class VelocityUNet(nn.Module):
    """v_theta(x, t): maps a point on the path plus its time to a velocity
    of the same shape. See the module docstring for the layout."""

    def __init__(self, base_channels=32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        time_dim = base_channels * 4

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.stem = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.down_block1 = ResBlock(c1, c1, time_dim)
        self.downsample1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)  # 28 -> 14
        self.down_block2 = ResBlock(c2, c2, time_dim)
        self.downsample2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)  # 14 -> 7

        self.mid_block1 = ResBlock(c3, c3, time_dim)
        self.mid_attention = SelfAttention2d(c3)
        self.mid_block2 = ResBlock(c3, c3, time_dim)

        self.upsample2 = Upsample(c3, c2)                        # 7 -> 14
        self.up_block2 = ResBlock(c2 + c2, c2, time_dim)
        self.upsample1 = Upsample(c2, c1)                        # 14 -> 28
        self.up_block1 = ResBlock(c1 + c1, c1, time_dim)

        self.out_norm = group_norm(c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=3, padding=1)
        # Start from a zero velocity field: the first steps then predict
        # "no movement" rather than noise, which stabilises early training.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_time_embedding(t, self.time_dim))

        h = self.stem(x)
        skip1 = self.down_block1(h, temb)
        h = self.down_block2(self.downsample1(skip1), temb)
        skip2 = h
        h = self.downsample2(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, temb)

        h = self.up_block2(torch.cat([self.upsample2(h), skip2], dim=1), temb)
        h = self.up_block1(torch.cat([self.upsample1(h), skip1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


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
            else:  # integer buffers (if any) are copied, not averaged
                self.shadow[name].copy_(value)

    def state_dict(self):
        return self.shadow


def to_model_scale(x01):
    """[0,1] pixels -> [-1,1], the scale the N(0,I) end of the path lives on."""
    return x01 * 2.0 - 1.0


def to_pixel_scale(x):
    """[-1,1] -> [0,1], clamped, for writing images out."""
    return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def sample_path(x1, sigma_min, generator=None):
    """Draw one (x_t, t, u) training triple per example.

    x_t is the point on the straight line from a fresh noise draw x0 to the
    real image x1; u is that line's velocity, which is constant along it.
    """
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    t_broadcast = t.view(-1, 1, 1, 1)

    x_t = (1.0 - (1.0 - sigma_min) * t_broadcast) * x0 + t_broadcast * x1
    u = x1 - (1.0 - sigma_min) * x0
    return x_t, t, u


def flow_matching_loss(model, x1, sigma_min, generator=None):
    """The whole objective: regress the conditional velocity with plain MSE."""
    x_t, t, u = sample_path(x1, sigma_min, generator=generator)
    return F.mse_loss(model(x_t, t), u)


def iterate_batches(X, batch_size, rng, shuffle):
    """Plain numpy-index batching (no torch DataLoader) - a single
    permutation per epoch, sliced into batches, so the batching logic stays
    visible rather than delegated to a library abstraction."""
    n = X.shape[0]
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx]


def run_epoch(model, X, batch_size, sigma_min, device, rng, optimizer=None,
              ema=None, eval_seed=None):
    """One pass over X. If optimizer is given, trains (shuffled, grad
    updates); otherwise evaluates (no shuffle, no grad) - shared loop so
    train/val accounting can't drift apart.

    On the eval side the t/noise draws come from a generator re-seeded with
    eval_seed every call, so the reported val loss is comparable across
    epochs instead of moving with the luck of the draw.
    """
    training = optimizer is not None
    model.train(training)

    generator = None
    if eval_seed is not None:
        generator = torch.Generator(device=device).manual_seed(eval_seed)

    total_loss = 0.0
    n_batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterate_batches(X, batch_size, rng, shuffle=training):
            x1 = to_model_scale(
                torch.from_numpy(batch).unsqueeze(1).to(device)      # (B, 1, 28, 28)
            )
            loss = flow_matching_loss(model, x1, sigma_min, generator=generator)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(model)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/mnist.npz")
    parser.add_argument("--output-dir", default="runs/mnist_flow")
    parser.add_argument("--base-channels", type=int, default=32,
                        help="UNet width; channels are this x (1, 2, 4) across the three resolutions.")
    parser.add_argument("--sigma-min", type=float, default=0.0,
                        help="Conditional-OT path parameter (Lipman et al.). 0.0 = rectified "
                             "flow: x_t = (1-t)*x0 + t*x1, u = x1 - x0.")
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay for the sampling weights; 0 disables EMA.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training rows held out for validation "
                             "(the MNIST test split stays fully unseen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = np.load(args.data_path, allow_pickle=True)
    X_train_full = data["X_train"]  # (n, 28, 28) float32 in [0,1]

    n = X_train_full.shape[0]
    perm = rng.permutation(n)
    n_val = int(n * args.val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]

    print(f"Train rows: {X_train.shape[0]}  Val rows: {X_val.shape[0]}  "
          f"base_channels={args.base_channels}  sigma_min={args.sigma_min}")

    model = VelocityUNet(args.base_channels).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = EMA(model, args.ema_decay) if args.ema_decay > 0 else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "flow_best.pt")
    final_path = os.path.join(args.output_dir, "flow_final.pt")

    def checkpoint(epoch, val_loss):
        return {
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "base_channels": args.base_channels,
            "sigma_min": args.sigma_min,
            "epoch": epoch,
            "val_loss": val_loss,
        }

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_epoch(
            model, X_train, args.batch_size, args.sigma_min, device, rng,
            optimizer=optimizer, ema=ema,
        )
        val_loss = run_epoch(
            model, X_val, args.batch_size, args.sigma_min, device, rng,
            optimizer=None, eval_seed=args.seed,
        )

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            print(
                f"epoch {epoch:3d}/{args.num_epochs}  "
                f"train: mse={train_loss:.4f}  val: mse={val_loss:.4f}  "
                f"[{time.time() - start_time:6.1f}s]"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint(epoch, val_loss), best_path)

    torch.save(checkpoint(args.num_epochs, val_loss), final_path)
    print(f"Trained {args.num_epochs} epochs in {time.time() - start_time:.1f}s")
    print(f"Saved best checkpoint (val_mse={best_val_loss:.4f}) to {best_path}")
    print(f"Saved final checkpoint to {final_path}")
    print("Run evaluate_flow.py against the held-out test split next.")


if __name__ == "__main__":
    main()

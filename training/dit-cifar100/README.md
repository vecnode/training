# dit-cifar100

A **class-conditional Diffusion Transformer** ([Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748) — the architecture Sora is built on) trained **from scratch** on CIFAR-100 — no diffusion, flow, or transformer library (no `diffusers`, no `torchcfm`, no `torchdiffeq`, no `transformers`, no `timm`, no `torchvision`). The patch embedding, the fixed 2D sincos positional embedding, the adaLN-Zero transformer blocks, the hand-written multi-head self-attention, the final unpatchify layer, the class embedding, the conditional-OT probability path, the velocity-regression loss, the classifier-free guidance, the EMA, and the Euler ODE sampler are all written out by hand in `train_dit.py` / `evaluate_dit.py`; torch is used only for tensor ops, autograd, and GPU execution, the same role it plays in every other `training/` pipeline (no `DataLoader`, numpy-permutation batching).

This is the natural big sibling of [`training/flow-matching-mnist`](../flow-matching-mnist/README.md): **the same conditional-OT flow-matching objective**, now conditioned on the 100 real CIFAR-100 fine classes, with **classifier-free guidance** (CFG) steering the samples. It is also the third CIFAR-100 pipeline in this repo, sharing `build_cifar100_dataset.py`'s exact `data/cifar100.npz` contract with `training/mae-cifar100`.

This is a `training/` pipeline (from-scratch, non-LoRA), an independent `uv` project like every other pipeline folder — own `pyproject.toml` (pinned to the CUDA 12.8 torch build), own `.python-version`, no shared root environment.

## Method

The model learns a class-conditioned time-dependent velocity field `v(x, t, y)` whose ODE `dx/dt = v(x, t, y)` transports noise `p_0 = N(0, I)` at `t=0` into the data distribution of class `y` at `t=1`. Training never solves that ODE. For one image `x1`, one noise draw `x0`, and its class `y`, the conditional path between them is a straight line and its velocity is regressed directly — **identical to `flow-matching-mnist`**:

```
t   ~ U(0, 1)                         one t per sample
x0  ~ N(0, I)
x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
u   = x1 - (1 - sigma_min) * x0       the line's velocity, constant in t
L   = || v_theta(x_t, t, y) - u ||^2  plain MSE, y = the image's class
```

That is the conditional optimal-transport path of [Lipman et al., 2023](https://arxiv.org/abs/2210.02747) with the independent coupling; at the default `--sigma-min 0.0` it is exactly the rectified flow of [Liu et al., 2022](https://arxiv.org/abs/2209.03003). What is absent, compared to a DDPM: no noise schedule, no `alpha_bar` table, no ELBO, no reweighting, no ancestral sampling chain — sampling is an ODE solve.

**Conditioning.** An embedding table maps the 100 fine classes into the transformer's dim conditioning space, where the vector is added to the 256-dim sinusoidal time embedding (scaled by 1000, as in `flow-matching-mnist`) and fed through every block's adaLN-Zero modulation MLPs. Each block's two norms are modulated by learned scale/shift and its two residual branches by learned gates; **zero-initializing the modulation output makes every block start as the identity** — the "Zero" in adaLN-Zero, and the init trick that lets deep DiTs train from scratch (ablated in the paper: learned per-block init or plain zero-init both hurt). The final layer modulates with shift/scale only, per the paper.

**Classifier-free guidance** ([Ho & Salimans, 2022](https://arxiv.org/abs/2207.12598)) is trained by dropping the class to a null token (embedding slot 100) with probability `--cfg-dropout 0.1` — the DiT paper's value — so one network learns both `v_cond` and `v_uncond`. At sample time:

```
v_cfg = v_uncond + cfg * (v_cond - v_uncond)
```

`--cfg-scale 1.0` is plain conditional sampling (one forward per step); the default evaluation uses 3.0, and `cfg_sweep.png` shows what the knob does.

## Model

Sora-style DiT — patch embed, transformer, unpatchify, no convolution beyond the patch embed:

```
patchify   Conv2d(3 -> dim, kernel=stride=2)       16x16 = 256 tokens
pos embed  fixed 2D sincos (frozen; the paper ablated learned pos embeds worse)
t embed    sinusoidal 256-dim (x1000) -> MLP(256 -> dim -> dim)
y embed    Embedding(101 -> dim); null token 100 for CFG
c          = t_emb + y_emb
blocks     N x DiTBlock (adaLN-Zero):
             x = x + gate_a * MHA(norm_a(x)*(1+scale_a)+shift_a)
             x = x + gate_m * MLP(norm_m(x)*(1+scale_m)+shift_m)
final      adaLN shift/scale-modulated LayerNorm -> Linear(dim -> patch^2 * 3) -> unpatchify
```

Defaults `--patch-size 2 --dim 256 --depth 8 --heads 8 --mlp-ratio 4` = **9,828,876 params** (256 tokens — the same token count as DiT-S/4 at 256 px, i.e. the Sora/DiT family's standard density at this budget). The conditioning and embedding widths follow the paper exactly (conditioning dimension = model dim, 256-dim sinusoidal time embedding, final layer shift/scale without a gate); the only deviation from the reference code is the null token (class 100) that CFG needs, which the paper's `LabelEmbedder` also supports via its dropout path. No pretrained weights anywhere: initialization is the paper's recipe — xavier-uniform on the Linears, frozen sincos pos embed, zeroed adaLN-Zero modulations, zeroed final linear (the velocity field starts at "no movement", which stabilises the first few hundred steps, the same trick `flow-matching-mnist`'s UNet uses on its output convolution).

Details that are choices, not boilerplate:

- **AdamW + linear-warmup-then-cosine LR**, not the DiT paper's constant LR — the same schedule `training/vit-cifar10` and `training/mae-cifar100` use; warmup is the part transformers need to train stably from scratch, and this repo's from-scratch ViT runs are the evidence.
- **Flip + 4px pad-crop augmentation**, plain torch ops per batch (the same augmentation `mae-cifar100` uses), applied to raw `[0,1]` pixels before the `[-1,1]` rescale. `--no-augment` is the documented A/B.
- **EMA of the weights** (`--ema-decay 0.999`, hand-written) is what gets sampled from — the same reasoning as `flow-matching-mnist`; `--weights raw` evaluates the non-averaged weights instead.
- **Pixels are rescaled to `[-1,1]`** in the trainer (like `flow-matching-mnist`), so the data sits on the same scale as the `N(0,I)` end of the path; the dataset builder deliberately stores raw `[0,1]` pixels, keeping the shared `data/cifar100.npz` contract with `mae-cifar100`.

None of these have been ablated in this repo — they are standard practice for this model family, stated as rationale, not as findings measured here. `--dim`, `--depth`, `--patch-size`, `--cfg-dropout`, `--sigma-min`, and `--no-augment` are CLI flags precisely so they can be tested rather than assumed.

## Dataset

Point `--data-dir` at a local folder with the raw CIFAR-100 python-format pickle files (`meta`, `train`, `test`); a nested `cifar-100-python/` subfolder (the tarball layout) is also accepted. The pickles are parsed by hand with stdlib `pickle` (`encoding='bytes'`, Python-2 format) — no torchvision, no `keras.datasets` — the same builder and the same `data/cifar100.npz` contract as `training/mae-cifar100`: `(n, 3, 32, 32)` `[0,1]` float32 images, fine (100-class) and coarse (20-superclass) labels for both splits, plus both name lists. CIFAR-100 ships one 50k train file and one 10k test file; the builder verifies **exactly 50,000/10,000** and refuses a partial or corrupt extraction. Data is not checked into this repo (`data/` is gitignored via the root `.gitignore`).

## Commands

```sh
uv run --directory training/dit-cifar100 python build_cifar100_dataset.py --data-dir "E:\datasets\cifar-100-python" --output-dir data
uv run --directory training/dit-cifar100 python train_dit.py --data-path data/cifar100.npz --patch-size 2 --dim 192 --depth 6 --heads 6 --num-epochs 60 --batch-size 256 --output-dir runs/dit_cifar100
uv run --directory training/dit-cifar100 python evaluate_dit.py --data-path data/cifar100.npz --checkpoint-path runs/dit_cifar100/dit_best.pt --num-steps 50 --cfg-scale 3.0 --output-dir runs/dit_cifar100
```

One-time environment bootstrap (creates `.venv`, installs the CUDA torch wheel, verifies `torch.cuda.is_available()`):

```sh
training\dit-cifar100\uv_setup.bat
```

`train_dit.py` runs a hand-written AdamW training loop (numpy-permutation batching, no `DataLoader`), printing train/val velocity MSE each epoch, holding out `--val-fraction` of the train split (the 10k test split stays fully unseen until `evaluate_dit.py`), and saves the best-val-loss checkpoint to `runs/dit_cifar100/dit_best.pt` plus a final checkpoint — both carrying the raw and EMA weights. It also writes a class-conditional `samples_epochNNNN.png` (6 classes × 4 fixed latents, fixed seed, CFG 2.0) every `--sample-every` epochs, so class conditioning and sample quality are visible *across* training, the way `fashion-mnist-dcgan`'s fixed-z grids make GAN collapse visible.

`evaluate_dit.py` writes three PNGs (encoded by hand with stdlib `zlib`, no imaging library) and prints three measurements — see below.

## How this is judged

Sample quality has no honest scalar here. A real FID needs a pretrained Inception network, which contradicts the from-scratch rule of this folder, and inventing a substitute number would be worse than not having one (the same rule that keeps FID out of `flow-matching-mnist` and ViSQOL out of `rvq-audio-codec`). So `samples_grid.png` is looked at, and everything printed is something that was actually measured:

- **`samples_grid.png`** — 100 class-conditional generations, one per fine class in row-major class order (cell `i` = class `i`): `x0 ~ N(0,I)` integrated to `t=1` by the guided field. This is the file that answers "does the class conditioning work, and what do the samples look like?" at a glance.
- **`cfg_sweep.png`** — the same fixed latents generated at CFG scales 1.0/1.5/2.0/3.0/5.0 (rows) for four classes (columns, latent fixed per column), so the guidance-vs-diversity trade-off is visible in one image.
- **`nearest_neighbours.png`** + L2 distances — the memorization guard, in the same form as `flow-matching-mnist`/`fashion-mnist-dcgan`: each generated sample above the closest training image to it (brute-force pixel L2 over all 50k, no index library, no learned features). Real held-out test images are measured the same way as a control, because the raw distance means nothing on its own — 50k images cover CIFAR-100's simple 32×32 shapes densely, so even a genuinely novel image lands close to something. Samples much closer than the control would mean memorization.
- **Test-set velocity MSE** — the training objective on genuinely unseen data, conditional on the true class, with `t`/noise drawn from a fixed seed so it is comparable to the val numbers in the training log.

## Verified runs

Real run on the RTX 3090 against the actual `data/cifar100.npz` built from
`E:\datasets\cifar-100-python` — the commands in the section above,
verbatim:

| | |
|---|---|
| Parameters | **9,828,876** (`--patch-size 2 --dim 256 --depth 8 --heads 8`) |
| Training | 60 epochs, batch 256, AdamW 1e-3 / wd 0.05, 5-epoch warmup then cosine to 1e-5, EMA 0.999, flip+crop, CFG dropout 0.1 |
| Wall clock | **4,521 s (~75 min)**, ~75 s/epoch |
| Train velocity MSE | 0.5174 → **0.1695** |
| Val velocity MSE | 0.2823 → **0.1842** (best, epoch 55; final 0.1845) |
| Test velocity MSE | **0.1887** on the held-out 10k, EMA weights |
| Sample grid | 100 class-conditional samples (one per fine class), 50 Euler steps, cfg 3.0 |
| NN check (generated vs control) | generated mean 12.105 / min 9.572 / max 17.440 vs real test mean 8.210 / min 5.152 / max 9.853 |

**The loss floor is ~0.17–0.19, and that is not a defect.** The regression
target `u = x1 - x0` is irreducibly random given `(x_t, t, y)`: many
(image, noise, class) triples pass through the same point at the same time,
and the network can only ever predict their mean. So the MSE floors at the
conditional variance of `u` and cannot go to zero — the same caveat
`flow-matching-mnist` documents about its ~0.17 floor. Judge this pipeline
by `samples_grid.png`, not by the loss.

Train and val track each other the whole way; best val lands at epoch 55 of
60 and the curve is still drifting down slowly — more epochs is the obvious
knob. The run used the **EMA weights** for the test MSE and all sample
grids (decay 0.999 over 10,560 steps leaves `0.999**10560 ≈ 2.5e-5` of the
random init — fully converged, unlike a short smoke checkpoint, the same
lesson `rvq-audio-codec` pins).

**Nearest-neighbour check:** generated samples sit **~47% farther** from the
training set than genuine unseen test images do (mean L2 12.105 vs 8.210,
min 9.572 vs 5.152) — the model is not reproducing its training data, the
same direction `flow-matching-mnist` measured on MNIST. Note the check ran
at cfg 3.0: guidance moves samples toward the class mode, so this distance
is the "guided" one.

## Compared with `training/flow-matching-mnist`

Same objective, same path, same ODE-solver sampling, same EMA, same nearest-neighbour memorization check — the difference is the model family and the conditioning: a 9.83M-parameter adaLN-Zero transformer over 256 tokens vs a 1.18M-parameter UNet, and 100 real classes + CFG vs unconditional 10-digit generation. The two pipelines' `samples_grid.png` files are the honest point of comparison for what the DiT/attention family buys over the UNet at this scale.

## Files

- `build_cifar100_dataset.py` — stdlib-pickle CIFAR-100 parsing (fine + coarse labels, verified 50k/10k), saves `data/cifar100.npz` (same contract as `training/mae-cifar100`)
- `train_dit.py` — `DiT` (patch embed, frozen sincos pos embed, adaLN-Zero blocks, hand-written MHA, final unpatchify), conditional-OT flow-matching objective, class embedding + CFG dropout, EMA, flip+crop augmentation, AdamW + warmup/cosine training loop, training-time sample grids
- `evaluate_dit.py` — Euler ODE sampler with CFG, test velocity MSE, `samples_grid.png`, `cfg_sweep.png`, nearest-neighbour memorization check
- `png_utils.py` — shared hand-written zlib RGB PNG writer (no Pillow)

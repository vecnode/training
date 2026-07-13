# fine-tuning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Skeleton for fine-tuning small language models with [Axolotl](https://axolotl.ai/) and `uv`, scaling from a single RTX 3090 (24GB) up to multi-GPU.

Example task wired up end-to-end: OCR text → summary text LoRA/QLoRA, sized for
a single RTX 3090 (24GB) and scalable from there without changing scripts —
only the config file changes.

## Setup

```bash
uv_bootstrap.bat   # or ./uv_bootstrap.ps1
```

This creates `.venv/`, syncs dependencies (torch pinned to the CUDA 12.8 build),
installs `axolotl`, and pulls Axolotl's example/deepspeed configs. Every download —
the venv, pip/uv cache, and Hugging Face model/dataset cache — is redirected under
`.cache/` in this repo (`UV_CACHE_DIR`, `HF_HOME`). Nothing touches your global
caches, and `rd /s /q .venv .cache` (or `Remove-Item -Recurse -Force .venv, .cache`)
resets the workspace completely.

## Folder structure

```
fine-tuning/
├── configs/            # Axolotl YAML configs (committed)
├── DATASET/            # your raw data — gitignored, see DATASET/README.md
├── DATASET_JSONL/      # generated train/val splits — gitignored
├── scripts/            # prepare_dataset.py + train/merge/inference/evaluate wrappers
├── eval/               # evaluate.py: ROUGE/BLEU scoring + predictions.csv
├── output/             # adapters, merged models, eval results — gitignored
└── .cache/             # uv + Hugging Face caches — gitignored
```

Only `configs/`, `scripts/`, `eval/`, and the README stubs in `DATASET/`,
`DATASET_JSONL/`, `output/` are version-controlled. Data and model weights are not.

## Workflow

1. Drop your raw CSV/JSONL into `DATASET/` (see [`DATASET/README.md`](DATASET/README.md)).
2. Build train/val splits:
   ```bash
   uv run python scripts/prepare_dataset.py \
     --input DATASET/your_file.csv \
     --text-col text --summary-col summary --val-split 0.1
   ```
3. Train:
   ```bash
   scripts\train.bat qlora-3090-24gb.yml
   ```
4. Merge the adapter into a standalone model (optional, needed for serving outside Axolotl/peft):
   ```bash
   scripts\merge_lora.bat qlora-3090-24gb.yml
   ```
5. Spot-check generations:
   ```bash
   scripts\inference.bat qlora-3090-24gb.yml
   ```
6. Score against the held-out split:
   ```bash
   scripts\evaluate.bat qlora-3090-24gb.yml
   ```

## Config sizing

| Config | Model class | Quantization | VRAM target | Notes |
|---|---|---|---|---|
| `configs/qlora-3090-24gb.yml` | 7-8B | 4-bit QLoRA | single RTX 3090 (24GB) | default baseline |
| `configs/lora-3090-24gb.yml` | 1-3B | none (full LoRA) | single RTX 3090 (24GB) | higher-fidelity adapter, smaller model |
| `configs/multi-gpu-scale.yml` | 32B+ | 4-bit QLoRA + deepspeed zero2/3 | multi-GPU | same dataset/adapter format, just bump model size, batch size, deepspeed stage |

All three configs read from `DATASET_JSONL/train.jsonl` / `val.jsonl` and use the
same Alpaca-style record schema, so scaling up is a config swap, not a rewrite.

## License

Licensed under the [MIT License](./LICENSE).

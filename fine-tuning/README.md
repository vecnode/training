# Fine-tuning Workspace

Adapter/LoRA fine-tuning pipelines. Each subfolder is an independent `uv`
project (own `pyproject.toml`/`.venv`) — see its README for setup and data.

| Folder | Framework | Base model |
|---|---|---|
| [`vicuna-7b-lora/`](vicuna-7b-lora/README.md) | `transformers` + `peft` | `lmsys/vicuna-7b-v1.5`, loaded directly (no LLaVA checkpoint, no vision encoder — see its README) |
| [`qwen25-3b-lora/`](qwen25-3b-lora/README.md) | `transformers` + `peft` | `Qwen/Qwen2.5-3B-Instruct` — same pattern as `vicuna-7b-lora/`, smaller/faster, ChatML instead of Vicuna's `USER:/ASSISTANT:` (see its README) |



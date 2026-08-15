# Serving Workspace

Production inference for trained pipelines. Each subfolder is independently
deployable with its own `uv` project — one `serving/<pipeline>/` per pipeline
in [`../fine-tuning/`](../fine-tuning/README.md) that has a serving story.

| Folder | Serves |
|---|---|
| [`vicuna-7b-lora/`](vicuna-7b-lora/README.md) | [`../fine-tuning/vicuna-7b-lora/`](../fine-tuning/vicuna-7b-lora/README.md)'s trained adapter |

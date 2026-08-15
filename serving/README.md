# Serving Workspace

Production inference for trained pipelines. Each subfolder is independently
deployable with its own `uv` project — one `serving/<pipeline>/` per pipeline
in [`../fine-tuning/`](../fine-tuning/README.md) that has a serving story.

| Folder | Serves |
|---|---|
| [`llava15-lora/`](llava15-lora/README.md) | [`../fine-tuning/llava15-lora/`](../fine-tuning/llava15-lora/README.md)'s trained adapter |

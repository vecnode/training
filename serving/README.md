# Serving Workspace

Production inference for trained pipelines. Each subfolder is independently
deployable with its own `uv` project — one `serving/<pipeline>/` per pipeline
in [`../fine-tuning/`](../fine-tuning/README.md) that has a serving story.

No serving pipelines currently. `vicuna-7b-lora/` — a FastAPI service that
loaded the base Vicuna-7B model plus
[`../fine-tuning/vicuna-7b-lora/`](../fine-tuning/vicuna-7b-lora/README.md)'s
trained adapter — was removed by the repo owner. If a serving folder returns,
the boundary it kept is worth keeping: `serving/<pipeline>/` reads the
fine-tuning pipeline's trained output directory (adapter or merged model) and
never imports its training code, which is what makes it independently
deployable.

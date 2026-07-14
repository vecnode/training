# output/

Training and evaluation artifacts land here: LoRA/QLoRA adapters, merged models, checkpoints,
logs, and eval prediction CSVs. Only this README is committed — everything else is gitignored
(see `.gitignore`) since it's large and fully reproducible from `configs/` + `DATASET_JSONL/`.

## Layout produced by the default configs

```
output/
├── qlora-3090-24gb/     # adapter + checkpoints from configs/qlora-3090-24gb.yml
├── lora-3090-24gb/      # adapter + checkpoints from configs/lora-3090-24gb.yml
├── merged/              # full merged model after scripts/merge_lora
└── eval/                # predictions.csv + metrics.json from scripts/evaluate
```

Safe to delete at any time — rerun training/eval to regenerate.

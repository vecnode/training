"""Print and inspect the deployed model's weights from the command line.

Loads the same artifact the server would (fused model if present, else the LoRA
adapter), prints a model summary, and dumps per-tensor stats for a name filter.

Run (from this folder):

    .venv\Scripts\python.exe inspect_weights.py                 # summary + q_proj sample
    .venv\Scripts\python.exe inspect_weights.py --filter v_proj --limit 20
    .venv\Scripts\python.exe inspect_weights.py --filter lora_  # only present in adapter mode
"""

from __future__ import annotations

import argparse
from pathlib import Path

from infer import DEFAULT_ADAPTER, Summarizer, resolve_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect deployed model weights")
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--merged-model", type=Path, default=None)
    p.add_argument("--filter", default="q_proj", help="Only show tensors whose name contains this")
    p.add_argument("--limit", type=int, default=20, help="Max tensors to print")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.adapter_dir, args.merged_model)
    summarizer = Summarizer(**source)
    summarizer.print_summary()

    print(f"\nWeights matching '{args.filter}' (limit {args.limit}):")
    rows = summarizer.inspect_weights(limit=args.limit, name_filter=args.filter)
    if not rows:
        print("  (no tensors matched)")
    for r in rows:
        print(f"  {r['name']}")
        print(f"    shape={r['shape']} dtype={r['dtype']} numel={r['numel']} "
              f"mean={r['mean']} std={r['std']} l2={r['l2_norm']} grad={r['requires_grad']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

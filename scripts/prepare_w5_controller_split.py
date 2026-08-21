#!/usr/bin/env python3
"""Freeze question-disjoint W5 controller train/dev pools from formal train-only data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/rl/formal_5k/train.parquet")
    p.add_argument("--output-dir", default="data/w5_controller")
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    rows = []
    for _, src in df.iterrows():
        extra = plain(src["extra_info"])
        reward = plain(src["reward_model"])
        ground = reward["ground_truth"]
        rows.append({
            "sample_id": str(extra["sample_id"]),
            "question": str(extra["question"]),
            "gold_answers": [str(x) for x in ground["target"]],
            "supporting_facts": plain(extra.get("supporting_facts") or ground.get("supporting_facts") or []),
            "split": "formal_train_only",
        })
    if len(rows) != len({x["sample_id"] for x in rows}):
        raise RuntimeError("duplicate sample_id in formal train")

    def order(row: dict[str, Any]) -> str:
        return hashlib.sha256(f'{args.seed}:{row["sample_id"]}'.encode()).hexdigest()

    rows.sort(key=order)
    dev, train = rows[: args.dev_size], rows[args.dev_size :]
    for row in dev:
        row["controller_split"] = "dev"
    for row in train:
        row["controller_split"] = "train"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out / "controller_train4500.jsonl",
        "dev": out / "controller_dev500.jsonl",
        "all": out / "controller_all5000.jsonl",
    }
    for name, values in (("train", train), ("dev", dev), ("all", dev + train)):
        with paths[name].open("w") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    train_ids = {x["sample_id"] for x in train}
    dev_ids = {x["sample_id"] for x in dev}
    summary = {
        "gate": "W5_QUESTION_SPLIT_PASS",
        "source": args.input,
        "source_rows": len(rows),
        "train_questions": len(train),
        "dev_questions": len(dev),
        "question_overlap": len(train_ids & dev_ids),
        "seed": args.seed,
        "files": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "gold_fields_runtime_visible": False,
    }
    (out / "split_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

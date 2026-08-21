#!/usr/bin/env python3
"""Stage the single W3 repair dataset with token-mass-balanced CONTINUE rows."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/60_provider_decoupled_replay/final_mix/decision_sft_balanced.jsonl"
DEV = ROOT / "results/60_provider_decoupled_replay/behavior_dev40/decision_sft.jsonl"
OUT = ROOT / "results/64_webmt_lora_v2/dataset"
CONTINUE_REPEAT = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    staged = []
    counts = {}
    chars = {}
    for row in rows:
        kind = row["metadata"]["decision_type"]
        repeat = CONTINUE_REPEAT if kind == "post_obs_continue" else 1
        counts[kind] = counts.get(kind, 0) + repeat
        chars[kind] = chars.get(kind, 0) + repeat * len(row["conversations"][-1]["value"])
        staged.extend([row] * repeat)
    # Stable ordering makes the staged file hash reproducible; the seeded training
    # dataloader performs the actual shuffle.
    staged.sort(key=lambda x: (x["metadata"].get("sample_id", ""), x["metadata"]["decision_type"]))
    OUT.mkdir(parents=True, exist_ok=True)
    train, dev = OUT / "train.jsonl", OUT / "dev.jsonl"
    with train.open("w", encoding="utf-8") as f:
        for row in staged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dev.write_bytes(DEV.read_bytes())
    info = {}
    for name, file_name in (("w3_token_balanced_train", "train.jsonl"), ("w3_behavior_dev40", "dev.jsonl")):
        info[name] = {
            "file_name": file_name, "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation"},
        }
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    manifest = {
        "gate": "W3_TOKEN_BALANCED_DATA_PASS", "repair": "continue_repeat_4",
        "train_rows": len(staged), "dev_rows": sum(1 for _ in dev.open()),
        "row_counts": counts, "target_char_mass": chars,
        "train_sha256": sha(train), "dev_sha256": sha(dev),
        "init_model": "GRPO@400", "protocol_changed": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

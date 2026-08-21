#!/usr/bin/env python3
"""Stage audited W3 decision train/dev files for LlamaFactory."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/61_webmt_lora/dataset"
TRAIN = ROOT / "results/60_provider_decoupled_replay/final_mix/decision_sft_balanced.jsonl"
DEV = ROOT / "results/60_provider_decoupled_replay/behavior_dev40/decision_sft.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train_out, dev_out = OUT / "train.jsonl", OUT / "dev.jsonl"
    shutil.copyfile(TRAIN, train_out)
    shutil.copyfile(DEV, dev_out)
    info = {}
    for name, file_name in (("w3_final_train", "train.jsonl"), ("w3_behavior_dev40", "dev.jsonl")):
        info[name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from", "content_tag": "value", "user_tag": "human",
                "assistant_tag": "gpt", "observation_tag": "observation",
            },
        }
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    manifest = {
        "gate": "W3_FINAL_SFT_DATA_PASS",
        "train_rows": sum(1 for _ in train_out.open()),
        "dev_rows": sum(1 for _ in dev_out.open()),
        "train_sha256": sha(train_out), "dev_sha256": sha(dev_out),
        "train_source": str(TRAIN), "dev_source": str(DEV),
        "train_on_prompt": False, "mask_history": True, "observation_masked": True,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

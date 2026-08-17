#!/usr/bin/env python3
"""Register 1.5D sharegpt_filled as LlamaFactory coldstart_v2. No training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SOURCE = REPO / "results/18_teacher_reasoning_v2/full/sharegpt_filled.jsonl"
AUDIT = REPO / "results/18_teacher_reasoning_v2/full/replacement_audit.json"
LF_INFO = Path("/data1/hcc/LlamaFactory/data/dataset_info.json")
PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
PREFIX = "eca_qwen3_8b_coldstart_v2"
OLD_V1 = ("eca_qwen3_8b_coldstart_train", "eca_qwen3_8b_coldstart_dev")
DROP = "hotpotqa_distractor_train_5a8bbd2e5542995d1e6f1435"
ADD = "hotpotqa_distractor_train_5ab547e35542997d4ad1f0ea"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--replacement-audit", type=Path, default=AUDIT)
    p.add_argument("--dev-ratio", type=float, default=0.05)
    p.add_argument("--dataset-info", type=Path, default=REPO / "config/dataset_info_sft_v2.json")
    p.add_argument("--register-llamafactory", type=Path, default=LF_INFO)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def lf_entry(file_path: Path) -> dict:
    return {
        "file_name": str(file_path.resolve()),
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
        },
    }


def pending_n(rows: list[dict]) -> int:
    return sum(1 for r in rows if PLACEHOLDER in json.dumps(r, ensure_ascii=False))


def sample_preview(rows: list[dict], category: str) -> dict:
    hit = next(r for r in rows if r.get("category") == category)
    roles = [c.get("from") for c in hit.get("conversations") or []]
    gpt_has_obs = any(
        c.get("from") == "gpt" and "<observation" in (c.get("value") or "").lower()
        for c in hit.get("conversations") or []
    )
    preview = {
        "sample_id": hit.get("sample_id"),
        "category": category,
        "roles": roles,
        "pending": PLACEHOLDER in json.dumps(hit, ensure_ascii=False),
        "obs_in_gpt": gpt_has_obs,
    }
    if category == "evidence_reasoning":
        gpt = next((c.get("value") or "" for c in hit["conversations"] if c.get("from") == "gpt"), "")
        preview["think_head"] = gpt[:400]
        preview["has_think"] = "<think>" in gpt
        preview["has_answer"] = "<answer>" in gpt
    return preview


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.exist_ok:
        raise SystemExit(f"output dir not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.source)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]
    audit = json.loads(args.replacement_audit.read_text(encoding="utf-8"))
    dropped = str(audit.get("dropped_id") or DROP)
    added = str(audit.get("replacement_id") or ADD)

    rng = random.Random(args.seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_dev = max(1, int(round(len(shuffled) * args.dev_ratio)))
    n_dev = min(n_dev, max(1, len(shuffled) // 10))
    dev = shuffled[:n_dev]
    train = shuffled[n_dev:]

    train_path = out / f"{PREFIX}_train.jsonl"
    dev_path = out / f"{PREFIX}_dev.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)

    train_ids = {r["sample_id"] for r in train}
    dev_ids = {r["sample_id"] for r in dev}
    all_ids = {r["sample_id"] for r in rows}
    entries = {
        f"{PREFIX}_train": lf_entry(train_path),
        f"{PREFIX}_dev": lf_entry(dev_path),
    }
    local_info = {
        f"{PREFIX}_train": lf_entry(train_path) | {"file_name": train_path.name},
        f"{PREFIX}_dev": lf_entry(dev_path) | {"file_name": dev_path.name},
    }
    (out / "dataset_info.json").write_text(json.dumps(local_info, indent=2) + "\n", encoding="utf-8")
    args.dataset_info.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_info.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    lf_patched = False
    v1_untouched = True
    if args.register_llamafactory.is_file():
        info = json.loads(args.register_llamafactory.read_text(encoding="utf-8"))
        v1_before = {k: info.get(k) for k in OLD_V1}
        info.update(entries)
        v1_after = {k: info.get(k) for k in OLD_V1}
        v1_untouched = v1_before == v1_after
        args.register_llamafactory.write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lf_patched = True

    yaml_text = (REPO / "config/sft_lora.yaml").read_text(encoding="utf-8")
    ev_preview = sample_preview(rows, "evidence_reasoning")
    sf_preview = sample_preview(rows, "search_format")
    hard = {
        "total_4550": len(rows) == 4550,
        "split_sum": len(train) + len(dev) == len(rows),
        "id_overlap_zero": len(train_ids & dev_ids) == 0,
        "pending_zero": pending_n(train) == 0 and pending_n(dev) == 0,
        "dropped_absent": dropped not in all_ids,
        "replacement_present": added in all_ids,
        "v2_names": f"{PREFIX}_train" in entries and f"{PREFIX}_dev" in entries,
        "yaml_dataset_v2": f"dataset: {PREFIX}_train" in yaml_text,
        "yaml_eval_v2": f"eval_dataset: {PREFIX}_dev" in yaml_text,
        "yaml_not_v1": "dataset: eca_qwen3_8b_coldstart_train" not in yaml_text,
        "yaml_output_gate2": "outputs/21_sft_qwen3_8b_lora" in yaml_text,
        "v1_untouched": v1_untouched,
        "evidence_filled": ev_preview["has_think"] and ev_preview["has_answer"] and not ev_preview["pending"],
        "search_obs_role": "observation" in sf_preview["roles"] and not sf_preview["obs_in_gpt"],
        "no_train_process": True,
    }
    gate = "GATE_REGISTER_COLDSTART_V2_PASS" if all(hard.values()) else "GATE_REGISTER_COLDSTART_V2_FAIL"
    report = {
        "gate": gate,
        "hard_gates": hard,
        "FINAL_SOURCE": str(args.source),
        "TOTAL": len(rows),
        "n_train": len(train),
        "n_dev": len(dev),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "train_dev_overlap": len(train_ids & dev_ids),
        "train_pending": pending_n(train),
        "dev_pending": pending_n(dev),
        "dropped_present": int(dropped in all_ids),
        "replacement_present": int(added in all_ids),
        "dataset_info": str(args.dataset_info),
        "llamafactory_patched": lf_patched,
        "evidence_reasoning_preview": ev_preview,
        "search_format_preview": sf_preview,
        "NO_TRAIN_PROCESS": True,
    }
    (out / "register_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(gate)
    if args.debug:
        print(f"TRAIN={train_path} DEV={dev_path}")
    if gate != "GATE_REGISTER_COLDSTART_V2_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

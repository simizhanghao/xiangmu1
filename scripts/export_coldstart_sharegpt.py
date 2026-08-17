"""Export coldstart JSONL → LlamaFactory ShareGPT JSONL.

Hard gates:
  1) <reasoning> → <think> (system + target)
  2) search_format: observation is a separate ShareGPT role (not learned)

Usage (repo root):
  python scripts/export_coldstart_sharegpt.py \
    --input data/sft/coldstart_v1.jsonl \
    --prefix eca_coldstart_v1 \
    --output-dir data/sft/llamafactory \
    --register-llamafactory /data1/hcc/LlamaFactory/data/dataset_info.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

AGENT_SYSTEM_PROMPT_THINK = (
    "You are an evidence-cost-aware research agent. "
    "Use only these tags when responding: "
    "<internal>, <search>, <evidence>, <think>, <answer>. "
    "Choose either internal knowledge or search, not both. "
    "When documents are provided, select supporting sentences as evidence "
    "before answering. Keep thinking short and grounded in evidence. "
    "Put the final answer inside <answer>...</answer>."
)

_OBS_RE = re.compile(
    r"<observation>(.*?)</observation>", re.DOTALL | re.IGNORECASE
)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export coldstart to ShareGPT for LF.")
    p.add_argument("--input", type=str, default="data/sft/coldstart_v1.jsonl")
    p.add_argument("--output-dir", type=str, default="data/sft/llamafactory")
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Output/dataset name prefix, e.g. eca_coldstart_v1. "
        "Default: infer from input stem (coldstart_v1 -> eca_coldstart_v1).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-ratio", type=float, default=0.05)
    p.add_argument("--smoke-size", type=int, default=80)
    p.add_argument(
        "--register-llamafactory",
        type=str,
        default="",
        help="Optional path to LlamaFactory/data/dataset_info.json to patch.",
    )
    return p.parse_args()


def infer_prefix(input_path: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    stem = input_path.stem  # coldstart_v1
    if stem.startswith("eca_"):
        return stem
    return f"eca_{stem}"


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def rewrite_reasoning_to_think(text: str) -> str:
    text = text.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")
    text = text.replace("<Reasoning>", "<think>").replace("</Reasoning>", "</think>")
    # system prompt wording
    text = text.replace("<reasoning>", "<think>")
    text = text.replace("Keep reasoning short", "Keep thinking short")
    text = text.replace(", <reasoning>,", ", <think>,")
    return text


def split_search_target(target: str) -> Tuple[str, str, str]:
    """Return (search_block, observation_body, post_obs_assistant)."""
    m_search = _SEARCH_RE.search(target)
    m_obs = _OBS_RE.search(target)
    if not m_search or not m_obs:
        raise ValueError("search_format target missing <search> or <observation>")
    search_block = f"<search>\n{m_search.group(1).strip()}\n</search>"
    obs_body = m_obs.group(1).strip()
    # everything after </observation>
    after = target[m_obs.end() :].strip()
    after = rewrite_reasoning_to_think(after)
    if not after:
        raise ValueError("empty post-observation assistant content")
    return search_block, obs_body, after


def to_sharegpt(row: Dict[str, Any]) -> Dict[str, Any]:
    category = row["category"]
    user_msg = None
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            user_msg = m["content"]
            break
    if not user_msg:
        raise ValueError(f"{row.get('sft_id')}: missing user message")

    system = AGENT_SYSTEM_PROMPT_THINK
    target = rewrite_reasoning_to_think(row["target"])

    if category == "search_format":
        search_block, obs_body, after = split_search_target(row["target"])
        conversations = [
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": search_block},
            {"from": "observation", "value": obs_body},
            {"from": "gpt", "value": after},
        ]
        obs_in_gpt = "<observation" in after.lower() or "<observation" in search_block.lower()
        if obs_in_gpt:
            raise ValueError(f"{row['sft_id']}: observation leaked into gpt turn")
    else:
        # Ensure no observation tag in learnable target
        if _OBS_RE.search(target):
            raise ValueError(
                f"{row['sft_id']}: non-search sample still contains <observation>"
            )
        conversations = [
            {"from": "human", "value": user_msg},
            {"from": "gpt", "value": target},
        ]

    return {
        "conversations": conversations,
        "system": system,
        "sft_id": row["sft_id"],
        "sample_id": row["sample_id"],
        "category": category,
        "metadata": {
            "phase": "2D0",
            "source_split": (row.get("metadata") or {}).get("source_split", "train"),
            "context_view": (row.get("metadata") or {}).get("context_view"),
            "reasoning_source": (row.get("provenance") or {}).get("reasoning_source"),
            "tag_protocol": "think_v1",
            "observation_role": "sharegpt_observation"
            if category == "search_format"
            else None,
        },
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_smoke(
    rows: List[Dict[str, Any]], size: int, rng: random.Random
) -> List[Dict[str, Any]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat in by_cat:
        rng.shuffle(by_cat[cat])
    # roughly proportional
    cats = list(by_cat.keys())
    picked: List[Dict[str, Any]] = []
    # at least a few per category
    per = max(1, size // max(len(cats), 1))
    for cat in cats:
        picked.extend(by_cat[cat][:per])
    rng.shuffle(picked)
    if len(picked) > size:
        picked = picked[:size]
    # fill
    pool = [r for r in rows if r not in picked]
    rng.shuffle(pool)
    while len(picked) < size and pool:
        picked.append(pool.pop())
    return picked


def patch_dataset_info(path: Path, prefix: str) -> None:
    info = json.loads(path.read_text(encoding="utf-8"))
    # Prefer files copied/symlinked under LlamaFactory/data/
    entries = {}
    for split in ("train", "dev", "smoke"):
        name = f"{prefix}_{split}"
        entries[name] = {
            "file_name": f"{name}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    info.update(entries)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched dataset_info entries {list(entries)} into {path}")


def validate_export(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_search = 0
    n_obs_role = 0
    n_reasoning_left = 0
    n_think = 0
    for r in rows:
        blob = json.dumps(r, ensure_ascii=False)
        if "<reasoning>" in blob.lower():
            n_reasoning_left += 1
        if "<think>" in blob:
            n_think += 1
        if r.get("category") == "search_format" or any(
            c.get("from") == "observation" for c in r["conversations"]
        ):
            n_search += 1
            roles = [c["from"] for c in r["conversations"]]
            if "observation" in roles:
                n_obs_role += 1
            # gpt turns must not contain observation tags
            for c in r["conversations"]:
                if c["from"] == "gpt" and "<observation" in c["value"].lower():
                    raise SystemExit(
                        f"FAIL: observation inside gpt for {r.get('sft_id')}"
                    )
    return {
        "n": len(rows),
        "n_with_think_tag": n_think,
        "n_reasoning_tag_remaining": n_reasoning_left,
        "n_search_style": n_search,
        "n_observation_role": n_obs_role,
        "by_category": dict(Counter(r["category"] for r in rows)),
    }


def main() -> None:
    args = parse_args()
    in_path = resolve(args.input)
    out_dir = resolve(args.output_dir)
    if not in_path.is_file():
        raise SystemExit(f"missing input: {in_path}")
    prefix = infer_prefix(in_path, args.prefix)

    raw = [
        json.loads(l)
        for l in in_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    converted: List[Dict[str, Any]] = []
    errors: List[str] = []
    for row in raw:
        try:
            converted.append(to_sharegpt(row))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('sft_id')}: {exc}")

    if errors:
        print(f"ERROR: {len(errors)} conversion failures", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    n_dev = max(1, int(round(len(converted) * args.dev_ratio)))
    # keep at least ~95% train
    n_dev = min(n_dev, max(1, len(converted) // 10))
    dev = converted[:n_dev]
    train = converted[n_dev:]
    smoke = stratified_smoke(train, min(args.smoke_size, len(train)), rng)

    train_path = out_dir / f"{prefix}_train.jsonl"
    dev_path = out_dir / f"{prefix}_dev.jsonl"
    smoke_path = out_dir / f"{prefix}_smoke.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    write_jsonl(smoke_path, smoke)
    # keep a versioned report next to outputs (do not clobber other prefixes)
    report_name = (
        "export_report.json" if prefix.endswith("v0") else f"export_report_{prefix}.json"
    )

    report = {
        "input": str(in_path),
        "prefix": prefix,
        "n_input": len(raw),
        "n_train": len(train),
        "n_dev": len(dev),
        "n_smoke": len(smoke),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "outputs": {
            "train": str(train_path),
            "dev": str(dev_path),
            "smoke": str(smoke_path),
        },
        "protocol": {
            "think_tag": True,
            "reasoning_tag_forbidden": True,
            "observation_as_sharegpt_role": True,
        },
        "validate_all": validate_export(converted),
        "validate_smoke": validate_export(smoke),
    }
    (out_dir / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if report["validate_all"]["n_reasoning_tag_remaining"] != 0:
        raise SystemExit("FAIL: <reasoning> still present after export")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir} prefix={prefix}")

    if args.register_llamafactory:
        lf_info = Path(args.register_llamafactory)
        if not lf_info.is_absolute():
            lf_info = REPO_ROOT / lf_info
        if lf_info.is_file():
            patch_dataset_info(lf_info, prefix)
        else:
            print(f"WARNING: dataset_info not found: {lf_info}", file=sys.stderr)


if __name__ == "__main__":
    main()

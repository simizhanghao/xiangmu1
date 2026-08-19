#!/usr/bin/env python3
"""Gate 5.5: freeze ~5K formal GRPO train from the 8k HotpotQA pool.

Policy prompt: system + Question only.
Reward: gold answers + supporting_facts (title, sentence_id).
Excludes frozen-dev@200 and sealed test. Records SFT overlap; does not drop it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
PARENT = REPO.parent
sys.path.insert(0, str(REPO))

from src.rl.candidate_index import write_contexts_jsonl  # noqa: E402
from src.sft.prototype_builder import AGENT_SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 5.5 freeze formal GRPO 5K.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/rl/formal_5k",
    )
    p.add_argument("--max-samples", type=int, default=5000, help="Train size.")
    p.add_argument("--n-formal-dev", type=int, default=1000)
    p.add_argument("--n-verl-val", type=int, default=16)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--train-pool",
        type=Path,
        default=PARENT / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl",
    )
    p.add_argument(
        "--frozen-dev-ids",
        type=Path,
        default=PARENT / "data/eval/hotpotqa_200_ids.txt",
    )
    p.add_argument(
        "--sealed-ids",
        type=Path,
        default=REPO / "data/sealed/hotpotqa_test500_ids.txt",
    )
    p.add_argument(
        "--sft-sharegpt",
        type=Path,
        default=REPO / "results/18_teacher_reasoning_v2/full/sharegpt_filled.jsonl",
    )
    p.add_argument("--preview-n", type=int, default=32)
    p.add_argument(
        "--export-formal-dev-jsonl",
        action="store_true",
        help="Write frozen formal-dev IDs to an Agent eval JSONL. Does not rebuild parquet.",
    )
    p.add_argument(
        "--eval-jsonl-out",
        type=Path,
        default=REPO / "data/eval/hotpotqa_formal_dev_1000.jsonl",
    )
    return p.parse_args()


def load_ids(path: Path) -> Set[str]:
    if not path.is_file():
        raise SystemExit(f"missing ids: {path}")
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_ids_from_sharegpt(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    ids: Set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("sample_id") or row.get("id")
            if sid:
                ids.add(str(sid))
    return ids


def sf_minimal(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sf in sample.get("supporting_facts") or []:
        title = sf.get("title")
        sid = sf.get("sentence_id", sf.get("sent_id"))
        if title is None or sid is None:
            continue
        try:
            out.append({"title": str(title), "sentence_id": int(sid)})
        except (TypeError, ValueError):
            continue
    return out


def eligible(sample: Dict[str, Any]) -> bool:
    q = (sample.get("question") or "").strip()
    golds = [str(x).strip() for x in (sample.get("gold_answers") or []) if str(x).strip()]
    ctx = sample.get("contexts") or []
    return bool(q and golds and sf_minimal(sample) and ctx)


def to_verl_row(sample: Dict[str, Any], *, split: str, idx: int) -> Dict[str, Any]:
    sid = str(sample["sample_id"])
    question = sample["question"]
    golds = list(sample.get("gold_answers") or [])
    sf = sf_minimal(sample)
    return {
        "data_source": "hotpotqa_distractor_candidate",
        "agent_name": "eca_search_agent",
        "prompt": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ],
        "ability": "qa",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"target": golds, "supporting_facts": sf},
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "sample_id": sid,
            "question": question,
            "supporting_facts": sf,
            "need_tools_kwargs": True,
            "tools_kwargs": {"search": {"create_kwargs": {"sample_id": sid}}},
        },
    }


def assert_no_prompt_leak(rows: List[Dict[str, Any]], name: str) -> None:
    for row in rows:
        blob = json.dumps(row["prompt"], ensure_ascii=False)
        if "supporting_facts" in blob or '"contexts"' in blob:
            raise SystemExit(f"LEAK in {name} prompt: {row['extra_info']['sample_id']}")
        user = row["prompt"][1]["content"]
        if not user.startswith("Question: "):
            raise SystemExit(f"bad user prompt: {row['extra_info']['sample_id']}")


def write_ids(path: Path, ids: List[str]) -> None:
    path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def export_formal_dev_jsonl(args: argparse.Namespace) -> None:
    """Materialize Gate 5.5 formal-dev IDs as a frozen eval JSONL. IDs stay frozen."""
    ids_path = args.output_dir / "formal_dev_ids.txt"
    train_ids_path = args.output_dir / "train_ids.txt"
    if not ids_path.is_file():
        raise SystemExit(f"missing formal-dev ids: {ids_path}")
    ordered = [ln.strip() for ln in ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(ordered) != args.n_formal_dev:
        raise SystemExit(f"expected {args.n_formal_dev} formal-dev ids, got {len(ordered)}")
    idset = set(ordered)
    if len(idset) != len(ordered):
        raise SystemExit("duplicate formal-dev ids")

    train_ids = load_ids(train_ids_path) if train_ids_path.is_file() else set()
    frozen = load_ids(args.frozen_dev_ids)
    sealed = load_ids(args.sealed_ids) if args.sealed_ids.is_file() else set()
    overlap = {
        "train": sorted(idset & train_ids),
        "frozen_dev": sorted(idset & frozen),
        "sealed": sorted(idset & sealed),
    }
    if any(overlap.values()):
        raise SystemExit(f"FORMAL_DEV_OVERLAP { {k: len(v) for k, v in overlap.items()} }")

    wanted = {sid: None for sid in ordered}
    with args.train_pool.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            if sid in wanted and wanted[sid] is None:
                if not eligible(row):
                    raise SystemExit(f"ineligible formal-dev row: {sid}")
                wanted[sid] = row
    missing = [sid for sid, row in wanted.items() if row is None]
    if missing:
        raise SystemExit(f"missing {len(missing)} pool rows, first={missing[0]}")

    out = args.eval_jsonl_out
    if out.exists():
        raise SystemExit(f"refuse overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sid in ordered:
            f.write(json.dumps(wanted[sid], ensure_ascii=False) + "\n")

    manifest = {
        "gate": "FORMAL_DEV_1000_EXPORT_PASS",
        "n": len(ordered),
        "eval_file": str(out),
        "ids_file": str(ids_path),
        "pool": str(args.train_pool),
        "overlap_train": 0,
        "overlap_frozen_dev": 0,
        "overlap_sealed": 0,
        "first_id": ordered[0],
        "last_id": ordered[-1],
        "note": "Confirm-only split. Does not replace frozen-dev@200. Do not use for selection until all four models are scored.",
    }
    man_path = out.with_name(out.stem + "_manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("FORMAL_DEV_1000_EXPORT_PASS")


def main() -> None:
    args = parse_args()
    if args.export_formal_dev_jsonl:
        export_formal_dev_jsonl(args)
        return
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refuse overwrite nonempty {out}")
    out.mkdir(parents=True)

    frozen_dev = load_ids(args.frozen_dev_ids)
    sealed = load_ids(args.sealed_ids)
    forbid = frozen_dev | sealed
    if frozen_dev & sealed:
        raise SystemExit("frozen-dev overlaps sealed test")

    pool = load_jsonl(args.train_pool)
    pool_ids = [str(r["sample_id"]) for r in pool]
    if len(pool_ids) != len(set(pool_ids)):
        raise SystemExit("duplicate sample_id in 8k pool")

    cand = [
        r
        for r in pool
        if str(r["sample_id"]) not in forbid and eligible(r)
    ]
    need = args.max_samples + args.n_formal_dev + args.n_verl_val
    if len(cand) < need:
        raise SystemExit(f"eligible {len(cand)} < need {need}")

    rng = random.Random(args.seed)
    rng.shuffle(cand)
    if args.debug:
        args.max_samples = min(8, args.max_samples)
        args.n_formal_dev = min(4, args.n_formal_dev)
        args.n_verl_val = min(2, args.n_verl_val)

    i0 = 0
    train_raw = cand[i0 : i0 + args.max_samples]
    i0 += args.max_samples
    formal_dev_raw = cand[i0 : i0 + args.n_formal_dev]
    i0 += args.n_formal_dev
    val_raw = cand[i0 : i0 + args.n_verl_val]

    train_ids = [str(r["sample_id"]) for r in train_raw]
    formal_dev_ids = [str(r["sample_id"]) for r in formal_dev_raw]
    val_ids = [str(r["sample_id"]) for r in val_raw]
    named = {
        "train": set(train_ids),
        "formal_dev": set(formal_dev_ids),
        "verl_val": set(val_ids),
        "frozen_dev": frozen_dev,
        "sealed_test": sealed,
    }
    for a, b in (
        ("train", "formal_dev"),
        ("train", "verl_val"),
        ("train", "frozen_dev"),
        ("train", "sealed_test"),
        ("formal_dev", "verl_val"),
        ("formal_dev", "frozen_dev"),
        ("formal_dev", "sealed_test"),
        ("verl_val", "frozen_dev"),
        ("verl_val", "sealed_test"),
    ):
        ov = named[a] & named[b]
        if ov:
            raise SystemExit(f"overlap {a}∩{b} n={len(ov)}")

    train_rows = [to_verl_row(r, split="train", idx=i) for i, r in enumerate(train_raw)]
    val_rows = [to_verl_row(r, split="val", idx=i) for i, r in enumerate(val_raw)]
    assert_no_prompt_leak(train_rows + val_rows, "verl")

    import datasets

    train_path = out / "train.parquet"
    val_path = out / "val.parquet"
    datasets.Dataset.from_list(train_rows).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_rows).to_parquet(str(val_path))
    n_idx = write_contexts_jsonl(train_raw + val_raw, out / "contexts_index.jsonl")

    sft_ids = sample_ids_from_sharegpt(args.sft_sharegpt)
    preview = []
    for r in train_raw[: args.preview_n]:
        preview.append(
            {
                "sample_id": r["sample_id"],
                "question": r["question"],
                "gold_answers": r.get("gold_answers"),
                "n_supporting_facts": len(sf_minimal(r)),
                "n_contexts": len(r.get("contexts") or []),
            }
        )
    preview_path = out / "human_preview_32.jsonl"
    with preview_path.open("w", encoding="utf-8") as f:
        for row in preview:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_ids(out / "train_ids.txt", train_ids)
    write_ids(out / "formal_dev_ids.txt", formal_dev_ids)
    write_ids(out / "verl_val_ids.txt", val_ids)

    manifest = {
        "gate": "GATE55_FORMAL_5K_PASS",
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_formal_dev": len(formal_dev_ids),
        "n_verl_val": len(val_ids),
        "n_index": n_idx,
        "unique_train": len(set(train_ids)) == len(train_ids),
        "overlap_train_frozen_dev": 0,
        "overlap_train_sealed": 0,
        "overlap_train_sft": len(set(train_ids) & sft_ids),
        "sft_ids_available": len(sft_ids),
        "policy_sees": ["system", "question"],
        "policy_must_not_see": ["gold_answers", "contexts", "supporting_facts"],
        "reward_sees": ["gold_answers", "supporting_facts"],
        "pool": str(args.train_pool),
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "contexts_index": str(out / "contexts_index.jsonl"),
        "note": (
            "SFT overlap is allowed curriculum reuse, recorded only. "
            "Do not open sealed test. Formal GRPO restarts from 22_ SFT merged."
        ),
    }
    (out / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not (
        manifest["unique_train"]
        and manifest["n_train"] == args.max_samples
        and n_idx == len(train_raw) + len(val_raw)
    ):
        raise SystemExit("GATE55_FAIL")
    print("GATE55_FORMAL_5K_PASS")


if __name__ == "__main__":
    main()

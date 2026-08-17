#!/usr/bin/env python3
"""Builder v2: deterministic 4550 skeleton from frozen 1.5C manifest. No Kimi."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

PARENT = Path("/data1/hcc/deepresearch")
sys.path.insert(0, str(PARENT))

from src.sft.coldstart_builder import assert_train_only, load_frozen_ids
from src.sft.prototype_builder import (
    AGENT_SYSTEM_PROMPT,
    base_row,
    build_internal,
    build_search_format,
    format_documents_for_user,
    format_evidence_block,
    gold_answer_of,
    gold_titles_covered,
    index_by_sample_id,
    load_jsonl,
    make_sft_id,
    oracle_documents,
    resolve_evidence_refs,
    validate_sft_row,
)

REPO = Path(__file__).resolve().parents[1]
POOL = Path("/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
MANIFEST = REPO / "results/16_select_8b_coldstart_v2/selection_manifest.jsonl"
RETRIEVAL = (
    PARENT / "results/retrieval_candidate_bm25_n8000_20260807_162150/retrieval_results.jsonl"
)
FROZEN_VAL = PARENT / "data/eval/hotpotqa_200_ids.txt"
PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
QUOTAS = {
    "internal": 950,
    "search_format": 1250,
    "evidence": 1150,
    "evidence_reasoning": 1200,
}
BUILDER = "qwen3_8b_coldstart_builder_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--pool", type=Path, default=POOL)
    p.add_argument("--retrieval-cache", type=Path, default=RETRIEVAL)
    p.add_argument("--frozen-val-ids", type=Path, default=FROZEN_VAL)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def load_export():
    path = PARENT / "scripts/export_coldstart_sharegpt.py"
    spec = importlib.util.spec_from_file_location("export_coldstart_sharegpt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def contexts_as_docs(sample: dict) -> list[dict]:
    docs = []
    for i, ctx in enumerate(sample.get("contexts") or [], 1):
        docs.append(
            {
                "document_id": ctx["document_id"],
                "title": ctx["title"],
                "text": ctx.get("text") or " ".join(ctx.get("sentences") or []),
                "rank": i,
                "score": None,
            }
        )
    return docs


def retrieval_for(sample: dict, retrieval_map: dict[str, dict]) -> dict:
    row = retrieval_map.get(sample["sample_id"])
    if row and row.get("documents") and gold_titles_covered(sample, row):
        return row
    gold_docs = oracle_documents(sample)
    if row and row.get("documents"):
        have = {d["title"] for d in row["documents"]}
        docs = list(row["documents"])
        for doc in gold_docs:
            if doc["title"] not in have:
                docs.append(doc)
        out = dict(row)
        out["documents"] = docs
        retr = dict(out.get("retriever") or {})
        retr["gold_merged"] = True
        retr.setdefault("name", "bm25s")
        retr.setdefault("scope", "candidate")
        out["retriever"] = retr
        return out
    pool_docs = contexts_as_docs(sample)
    return {
        "sample_id": sample["sample_id"],
        "documents": pool_docs or gold_docs,
        "retriever": {
            "name": "hotpotqa_distractor_contexts",
            "scope": "pool_contexts",
        },
    }


def stamp(row: dict, *, mix_tag: str, seed: int, extra_meta: dict | None = None) -> dict:
    row["provenance"]["builder"] = BUILDER
    row["metadata"]["phase"] = "1.5C"
    row["metadata"]["mix_tag"] = mix_tag
    row["metadata"]["source_split"] = "train"
    row["metadata"]["seed"] = seed
    if extra_meta:
        row["metadata"].update(extra_meta)
    return row


def build_evidence_row(sample: dict, seed: int, retrieval_row: dict) -> dict:
    refs = resolve_evidence_refs(sample)
    docs = list(retrieval_row.get("documents") or []) or contexts_as_docs(sample)
    gold = gold_answer_of(sample)
    target = (
        f"<evidence>\n{format_evidence_block(refs)}\n</evidence>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    retr = dict(retrieval_row.get("retriever") or {})
    retr.setdefault("name", "bm25s")
    retr.setdefault("scope", "candidate")
    row = base_row(
        sample,
        category="evidence",
        taxonomy_label="train",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER,
            "teacher_id": None,
            "retriever": retr,
            "reasoning_source": None,
            "context_view": "bm25_or_gold",
        },
        metadata={
            "observation_in_target": False,
            "context_view": "bm25_or_gold",
            "n_input_docs": len(docs),
        },
    )
    row["sft_id"] = make_sft_id(sample["sample_id"], "evidence", "v2")
    return stamp(row, mix_tag="evidence_v2", seed=seed)


def build_reasoning_pending(sample: dict, seed: int, retrieval_row: dict, band: str | None) -> dict:
    refs = resolve_evidence_refs(sample)
    docs = list(retrieval_row.get("documents") or []) or contexts_as_docs(sample)
    gold = gold_answer_of(sample)
    target = (
        f"<evidence>\n{format_evidence_block(refs)}\n</evidence>\n"
        f"<think>\n{PLACEHOLDER}\n</think>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    retr = dict(retrieval_row.get("retriever") or {})
    row = base_row(
        sample,
        category="evidence_reasoning",
        taxonomy_label="pending_teacher",
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER,
            "teacher_id": None,
            "retriever": retr,
            "reasoning_source": "pending",
            "context_view": "bm25_or_gold",
        },
        metadata={
            "observation_in_target": False,
            "context_view": "bm25_or_gold",
            "reasoning_band": band,
            "teacher_pending": True,
        },
    )
    row["sft_id"] = make_sft_id(sample["sample_id"], "evidence_reasoning", "pending_v2")
    return stamp(row, mix_tag="evidence_reasoning_pending_v2", seed=seed)


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.exist_ok:
        raise SystemExit(f"output dir not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_jsonl(str(args.manifest))
    if args.max_samples and args.max_samples > 0:
        manifest = manifest[: args.max_samples]
    by_cat = {}
    for row in manifest:
        by_cat.setdefault(row["category"], []).append(row)
    pool = index_by_sample_id(load_jsonl(str(args.pool)))
    cache_found = args.retrieval_cache.is_file()
    retrieval_map = (
        index_by_sample_id(load_jsonl(str(args.retrieval_cache))) if cache_found else {}
    )
    frozen = load_frozen_ids(str(args.frozen_val_ids)) if args.frozen_val_ids.is_file() else set()

    built: list[dict] = []
    rejected: list[dict] = []
    n_gold_merged = 0
    for item in manifest:
        sid = item["sample_id"]
        sample = pool[sid]
        cat = item["category"]
        retr = retrieval_for(sample, retrieval_map)
        if (retr.get("retriever") or {}).get("gold_merged"):
            n_gold_merged += 1
        if cat == "internal":
            row = build_internal(sample, "direct_correct", args.seed)
            row["sft_id"] = make_sft_id(sid, "internal", "v2")
            row = stamp(row, mix_tag="internal_direct_ok_v2", seed=args.seed)
        elif cat == "search_format":
            row = build_search_format(sample, "search_required", args.seed, retr)
            row["sft_id"] = make_sft_id(sid, "search_format", "v2")
            row = stamp(row, mix_tag="search_format_v2", seed=args.seed)
        elif cat == "evidence":
            row = build_evidence_row(sample, args.seed, retr)
        elif cat == "evidence_reasoning":
            row = build_reasoning_pending(sample, args.seed, retr, item.get("reasoning_band"))
        else:
            raise SystemExit(f"unknown category {cat}")
        errs = validate_sft_row(row)
        if errs:
            rejected.append({"sft_id": row.get("sft_id"), "errors": errs})
            continue
        built.append(row)

    assert_train_only(built, frozen)
    export_mod = load_export()
    sharegpt = [export_mod.to_sharegpt(row) for row in built]
    export_stats = export_mod.validate_export(sharegpt)

    counts = dict(Counter(r["category"] for r in built))
    built_ids = [r["sample_id"] for r in built]
    man_reason = {r["sample_id"] for r in by_cat.get("evidence_reasoning", [])}
    built_reason = {r["sample_id"] for r in built if r["category"] == "evidence_reasoning"}
    obs_in_gpt = 0
    pending_ok = 0
    for row in sharegpt:
        if row["category"] == "evidence_reasoning" and PLACEHOLDER in json.dumps(row):
            pending_ok += 1
        for turn in row["conversations"]:
            if turn["from"] == "gpt" and "<observation" in turn["value"].lower():
                obs_in_gpt += 1
    overlap_val = sorted(set(built_ids) & frozen)
    hard = {
        "n_4550": len(built) == 4550,
        "unique_sample_id_4550": len(set(built_ids)) == 4550,
        "quotas": counts == QUOTAS,
        "reasoning_ids_match_manifest": man_reason == built_reason,
        "kimi_calls_zero": True,
        "sft_lora_launched_zero": True,
        "rejected_zero": len(rejected) == 0,
        "obs_not_in_gpt": obs_in_gpt == 0,
        "overlap_val200_zero": len(overlap_val) == 0,
        "pending_slots": pending_ok == QUOTAS["evidence_reasoning"],
    }
    gate = "GATE_BUILDER_V2_PASS" if all(hard.values()) else "GATE_BUILDER_V2_FAIL"
    audit = {
        "gate": gate,
        "hard_gates": hard,
        "counts": counts,
        "n": len(built),
        "unique": len(set(built_ids)),
        "rejected": rejected[:20],
        "n_rejected": len(rejected),
        "n_gold_merged_retrieval": n_gold_merged,
        "retrieval_cache_found": cache_found,
        "retrieval_cache": str(args.retrieval_cache),
        "document_source": "bm25_cache" if cache_found else "pool_contexts",
        "overlap_val200": len(overlap_val),
        "kimi_calls": 0,
        "sft_lora_launched": 0,
        "sealed_test_content_unread": True,
        "export": export_stats,
        "pending_placeholder": PLACEHOLDER,
        "manifest": str(args.manifest),
    }

    canon_path = out / "canonical.jsonl"
    share_path = out / "sharegpt.jsonl"
    with canon_path.open("w", encoding="utf-8") as handle:
        for row in built:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with share_path.open("w", encoding="utf-8") as handle:
        for row in sharegpt:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "build_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "ids_evidence_reasoning.json").write_text(
        json.dumps(sorted(built_reason), indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))
    print(gate)
    if args.debug:
        print(f"CANON={canon_path} SHARE={share_path} N={len(built)}")
    if gate != "GATE_BUILDER_V2_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

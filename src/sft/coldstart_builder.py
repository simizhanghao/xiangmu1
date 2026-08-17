"""Phase 2C: build ~3k train-only cold-start SFT rows.

- Source: HotpotQA distractor/train only (never frozen validation-200)
- Evidence views: clean (oracle docs) + noisy (oracle + distractors)
- Internal: Direct-correct labels from Base Qwen
- Reasoning: template_v0 with provenance
- Search-format: candidate BM25 protocol only
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.sft.prototype_builder import (
    AGENT_SYSTEM_PROMPT,
    base_row,
    build_internal,
    build_reasoning,
    build_search_format,
    format_documents_for_user,
    format_evidence_block,
    gold_answer_of,
    gold_titles_covered,
    make_sft_id,
    oracle_documents,
    resolve_evidence_refs,
    validate_sft_row,
)

BUILDER_NAME = "phase2c_coldstart_builder_v0"

DEFAULT_TARGETS = {
    "evidence_reasoning": 1500,
    "evidence": 600,
    "internal": 450,
    "search_format": 450,
}


def load_frozen_ids(path: str) -> Set[str]:
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def assert_train_only(
    samples: Sequence[Dict[str, Any]], frozen_ids: Set[str]
) -> None:
    bad_split = [s["sample_id"] for s in samples if "_train_" not in s["sample_id"]]
    if bad_split:
        raise ValueError(
            f"non-train sample_ids present: n={len(bad_split)} e.g. {bad_split[0]}"
        )
    overlap = [s["sample_id"] for s in samples if s["sample_id"] in frozen_ids]
    if overlap:
        raise ValueError(
            f"LEAKAGE: {len(overlap)} sample_ids overlap frozen validation-200 "
            f"e.g. {overlap[0]}"
        )


def distractor_documents(
    sample: Dict[str, Any],
    *,
    max_distractors: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    gold_titles = {sf["title"] for sf in sample.get("supporting_facts") or []}
    cands = [
        c for c in (sample.get("contexts") or []) if c["title"] not in gold_titles
    ]
    rng.shuffle(cands)
    docs: List[Dict[str, Any]] = []
    for i, ctx in enumerate(cands[:max_distractors], start=1):
        docs.append(
            {
                "document_id": ctx["document_id"],
                "title": ctx["title"],
                "text": ctx["text"],
                "rank": i,
                "score": None,
                "metadata": {
                    "sentences": list(ctx.get("sentences") or []),
                    "role": "distractor",
                },
            }
        )
    return docs


def pack_input_documents(
    sample: Dict[str, Any],
    *,
    noisy: bool,
    max_distractors: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], str]:
    gold_docs = oracle_documents(sample)
    for d in gold_docs:
        d.setdefault("metadata", {})["role"] = "gold_supporting"
    if not noisy:
        return gold_docs, "clean"
    distractors = distractor_documents(
        sample, max_distractors=max_distractors, rng=rng
    )
    packed = list(gold_docs) + list(distractors)
    rng.shuffle(packed)
    for i, d in enumerate(packed, start=1):
        d["rank"] = i
    return packed, "noisy"


def build_evidence_view(
    sample: Dict[str, Any],
    *,
    with_reasoning: bool,
    noisy: bool,
    max_distractors: int,
    seed: int,
    rng: random.Random,
) -> Dict[str, Any]:
    refs = resolve_evidence_refs(sample)
    docs, view = pack_input_documents(
        sample, noisy=noisy, max_distractors=max_distractors, rng=rng
    )
    gold = gold_answer_of(sample)
    evidence_block = format_evidence_block(refs)
    if with_reasoning:
        category = "evidence_reasoning"
        reasoning = build_reasoning(refs, gold)
        target = (
            f"<evidence>\n{evidence_block}\n</evidence>\n"
            f"<think>\n{reasoning}\n</think>\n"
            f"<answer>\n{gold}\n</answer>"
        )
        reasoning_source = "template_v0"
    else:
        category = "evidence"
        target = (
            f"<evidence>\n{evidence_block}\n</evidence>\n"
            f"<answer>\n{gold}\n</answer>"
        )
        reasoning_source = None

    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    view_tag = f"{view}_v0"
    n_dist = sum(
        1 for d in docs if (d.get("metadata") or {}).get("role") == "distractor"
    )
    n_gold = sum(
        1 for d in docs if (d.get("metadata") or {}).get("role") == "gold_supporting"
    )
    row = base_row(
        sample,
        category=category,
        taxonomy_label="train",
        messages=messages,
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": {
                "name": "oracle_plus_distractors" if noisy else "oracle",
                "scope": (
                    "oracle_supporting_docs_plus_distractors"
                    if noisy
                    else "oracle_supporting_docs"
                ),
                "n_distractors": n_dist,
            },
            "reasoning_source": reasoning_source,
            "context_view": view,
        },
        metadata={
            "phase": "2C",
            "mix_tag": f"{category}_{view}",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
            "source_split": "train",
            "context_view": view,
            "n_input_docs": len(docs),
            "n_gold_docs": n_gold,
            "n_distractor_docs": n_dist,
        },
    )
    row["sft_id"] = make_sft_id(sample["sample_id"], category, view_tag)
    return row


def build_internal_from_direct(
    sample: Dict[str, Any], seed: int, label_row: Dict[str, Any]
) -> Dict[str, Any]:
    row = build_internal(sample, taxonomy_label="direct_correct", seed=seed)
    row["provenance"]["builder"] = BUILDER_NAME
    row["provenance"]["direct_label"] = {
        "exact_match": label_row.get("exact_match"),
        "token_f1": label_row.get("token_f1"),
        "prediction": label_row.get("prediction"),
    }
    row["metadata"]["phase"] = "2C"
    row["metadata"]["mix_tag"] = "internal_direct_correct"
    row["metadata"]["source_split"] = "train"
    row["sft_id"] = make_sft_id(sample["sample_id"], "internal", "direct_v0")
    return row


def build_search_format_train(
    sample: Dict[str, Any],
    seed: int,
    retrieval_row: Dict[str, Any],
) -> Dict[str, Any]:
    row = build_search_format(sample, "train", seed, retrieval_row)
    row["provenance"]["builder"] = BUILDER_NAME
    row["metadata"]["phase"] = "2C"
    row["metadata"]["source_split"] = "train"
    row["sft_id"] = make_sft_id(sample["sample_id"], "search_format", "cand_v0")
    return row


def assign_coldstart(
    train_samples: Sequence[Dict[str, Any]],
    *,
    frozen_ids: Set[str],
    direct_labels: Dict[str, Dict[str, Any]],
    retrieval: Dict[str, Dict[str, Any]],
    seed: int = 42,
    targets: Optional[Dict[str, int]] = None,
    max_distractors: int = 3,
    noisy_fraction: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build mixture. Returns (accepted, rejected, build_stats)."""
    targets = dict(targets or DEFAULT_TARGETS)
    assert_train_only(train_samples, frozen_ids)

    rng = random.Random(seed)
    by_id = {s["sample_id"]: s for s in train_samples}
    all_ids = sorted(by_id.keys())
    rng.shuffle(all_ids)

    direct_correct_ids = [
        sid
        for sid in all_ids
        if direct_labels.get(sid, {}).get("direct_correct")
        or float(direct_labels.get(sid, {}).get("exact_match") or 0) >= 1.0
    ]
    rng.shuffle(direct_correct_ids)

    search_eligible = [
        sid
        for sid in all_ids
        if sid in retrieval and gold_titles_covered(by_id[sid], retrieval[sid])
    ]
    rng.shuffle(search_eligible)

    used: Set[str] = set()
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    def try_add(row: Dict[str, Any]) -> bool:
        errs = validate_sft_row(row)
        if errs:
            rejected.append({"sft_id": row.get("sft_id"), "errors": errs})
            return False
        accepted.append(row)
        return True

    def count_cat(cat: str) -> int:
        return sum(1 for r in accepted if r["category"] == cat)

    for sid in direct_correct_ids:
        if count_cat("internal") >= targets["internal"]:
            break
        if sid in used:
            continue
        if try_add(build_internal_from_direct(by_id[sid], seed, direct_labels[sid])):
            used.add(sid)

    for sid in search_eligible:
        if count_cat("search_format") >= targets["search_format"]:
            break
        if sid in used:
            continue
        if try_add(build_search_format_train(by_id[sid], seed, retrieval[sid])):
            used.add(sid)

    def fill_evidence(with_reasoning: bool, need: int) -> int:
        if need <= 0:
            return 0
        got = 0
        remain = [sid for sid in all_ids if sid not in used]
        rng.shuffle(remain)
        for sid in remain:
            if got >= need:
                break
            gold_titles = {
                sf["title"] for sf in by_id[sid].get("supporting_facts") or []
            }
            n_dist_avail = sum(
                1
                for c in by_id[sid].get("contexts") or []
                if c["title"] not in gold_titles
            )
            noisy = (rng.random() < noisy_fraction) and n_dist_avail > 0
            try:
                row = build_evidence_view(
                    by_id[sid],
                    with_reasoning=with_reasoning,
                    noisy=noisy,
                    max_distractors=max_distractors,
                    seed=seed,
                    rng=rng,
                )
            except Exception as exc:  # noqa: BLE001
                rejected.append({"sft_id": f"{sid}__evidence__", "errors": [str(exc)]})
                continue
            if try_add(row):
                used.add(sid)
                got += 1
        return got

    fill_evidence(True, targets["evidence_reasoning"])
    fill_evidence(False, targets["evidence"])
    # top-up reasoning if short
    fill_evidence(True, max(0, targets["evidence_reasoning"] - count_cat("evidence_reasoning")))

    cat_order = {
        "internal": 0,
        "evidence": 1,
        "evidence_reasoning": 2,
        "search_format": 3,
    }
    accepted.sort(key=lambda r: (cat_order.get(r["category"], 9), r["sft_id"]))

    built = Counter(r["category"] for r in accepted)
    stats = {
        "targets": targets,
        "built": dict(built),
        "shortfall": {k: max(0, targets[k] - built.get(k, 0)) for k in targets},
        "n_direct_correct_available": len(direct_correct_ids),
        "n_search_eligible": len(search_eligible),
        "n_unique_sample_ids": len({r["sample_id"] for r in accepted}),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "context_views": dict(
            Counter(
                (r.get("metadata") or {}).get("context_view", "n/a") for r in accepted
            )
        ),
        "builder": BUILDER_NAME,
    }
    return accepted, rejected, stats

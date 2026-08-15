"""Phase 2B: build / validate cold-start SFT prototype rows.

Deterministic gold evidence from HotpotQA supporting_facts.
Template reasoning (no LLM teacher). Candidate-BM25 for search_format only.
Evidence text equality: whitespace-normalized (strip + collapse).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.eval.metrics import normalize_answer

CATEGORIES = frozenset(
    {"internal", "evidence", "evidence_reasoning", "search_format"}
)

AGENT_SYSTEM_PROMPT = (
    "You are an evidence-cost-aware research agent. "
    "Use only these tags when responding: "
    "<internal>, <search>, <evidence>, <think>, <answer>. "
    "Choose either internal knowledge or search, not both. "
    "When documents are provided, select supporting sentences as evidence "
    "before answering. Keep thinking short and grounded in evidence. "
    "Put the final answer inside <answer>...</answer>."
)

BUILDER_NAME = "phase2_sft_builder_v0"
VIEW_TAG = "v0"

_TAG_RE = {
    "internal": re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE),
    "search": re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE),
    "observation": re.compile(
        r"<observation>(.*?)</observation>", re.DOTALL | re.IGNORECASE
    ),
    "evidence": re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE),
    "reasoning": re.compile(
        r"<(?:reasoning|think)>(.*?)</(?:reasoning|think)>",
        re.DOTALL | re.IGNORECASE,
    ),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
}

_EVIDENCE_LINE_RE = re.compile(
    r"\[document_id=(?P<document_id>[^\|\]]+?)\s*\|\s*"
    r"title=(?P<title>.*?)\s*\|\s*"
    r"sentence_id=(?P<sentence_id>\d+)\]\s*\n"
    r"(?P<text>.*?)(?=\n\[document_id=|\Z)",
    re.DOTALL,
)


def whitespace_norm(text: str) -> str:
    """Frozen 2B equality: strip + collapse whitespace."""
    return " ".join((text or "").split())


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    import json
    from pathlib import Path

    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_sample_id(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["sample_id"]: r for r in rows}


def resolve_evidence_refs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build gold evidence_refs from supporting_facts + contexts."""
    title_to_ctx = {c["title"]: c for c in sample.get("contexts") or []}
    refs: List[Dict[str, Any]] = []
    for sf in sample.get("supporting_facts") or []:
        title = sf["title"]
        sentence_id = int(sf["sentence_id"])
        ctx = title_to_ctx.get(title)
        if ctx is None:
            raise ValueError(f"{sample['sample_id']}: missing context title {title!r}")
        sentences = ctx.get("sentences") or []
        if sentence_id < 0 or sentence_id >= len(sentences):
            raise ValueError(
                f"{sample['sample_id']}: sentence_id {sentence_id} out of range "
                f"for title {title!r}"
            )
        text = sentences[sentence_id]
        refs.append(
            {
                "document_id": ctx["document_id"],
                "title": title,
                "sentence_id": sentence_id,
                "text": text,
            }
        )
    return refs


def oracle_documents(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts = sample.get("contexts") or []
    title_to_ctx = {c["title"]: c for c in contexts}
    docs: List[Dict[str, Any]] = []
    seen = set()
    rank = 1
    for sf in sample.get("supporting_facts") or []:
        title = sf["title"]
        if title in seen:
            continue
        seen.add(title)
        ctx = title_to_ctx.get(title)
        if ctx is None:
            raise ValueError(
                f"{sample['sample_id']}: oracle title missing: {title!r}"
            )
        docs.append(
            {
                "document_id": ctx["document_id"],
                "title": ctx["title"],
                "text": ctx["text"],
                "rank": rank,
                "score": None,
                "metadata": {"sentences": list(ctx.get("sentences") or [])},
            }
        )
        rank += 1
    return docs


def format_documents_for_user(documents: Sequence[Dict[str, Any]]) -> str:
    blocks = []
    for doc in documents:
        blocks.append(
            f"[DOC] document_id={doc['document_id']} title={doc['title']}\n"
            f"{doc['text']}"
        )
    return "\n\n".join(blocks)


def format_observation(documents: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[{doc['document_id']}] {doc['title']}: {doc['text']}" for doc in documents
    )


def format_evidence_block(refs: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for ref in refs:
        text = whitespace_norm(ref["text"])
        parts.append(
            f"[document_id={ref['document_id']} | title={ref['title']} | "
            f"sentence_id={ref['sentence_id']}]\n{text}"
        )
    return "\n\n".join(parts)


def build_reasoning(refs: Sequence[Dict[str, Any]], gold_answer: str) -> str:
    """Deterministic short bridge; teacher_id remains null (template)."""
    lines = []
    for i, ref in enumerate(refs, 1):
        lines.append(
            f"Evidence {i} ({ref['title']}): {whitespace_norm(ref['text'])}"
        )
    lines.append(
        f"Combining these supporting facts yields the answer: {gold_answer}."
    )
    return "\n".join(lines)


def gold_answer_of(sample: Dict[str, Any]) -> str:
    answers = sample.get("gold_answers") or []
    if not answers:
        raise ValueError(f"{sample['sample_id']}: empty gold_answers")
    return answers[0]


def make_sft_id(sample_id: str, category: str, view: str = VIEW_TAG) -> str:
    return f"{sample_id}__{category}__{view}"


def base_row(
    sample: Dict[str, Any],
    *,
    category: str,
    taxonomy_label: str,
    messages: List[Dict[str, str]],
    target: str,
    evidence_refs: List[Dict[str, Any]],
    documents: List[Dict[str, Any]],
    provenance: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    gold = gold_answer_of(sample)
    return {
        "sample_id": sample["sample_id"],
        "sft_id": make_sft_id(sample["sample_id"], category),
        "source_dataset": "hotpotqa",
        "category": category,
        "taxonomy_label": taxonomy_label,
        "messages": messages,
        "target": target,
        "gold_answer": gold,
        "gold_answers": list(sample.get("gold_answers") or []),
        "evidence_refs": evidence_refs,
        "documents": documents,
        "provenance": provenance,
        "metadata": metadata,
    }


def build_internal(
    sample: Dict[str, Any], taxonomy_label: str, seed: int
) -> Dict[str, Any]:
    gold = gold_answer_of(sample)
    target = (
        "<internal>\nUse internal knowledge.\n</internal>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {sample['question']}"},
    ]
    return base_row(
        sample,
        category="internal",
        taxonomy_label=taxonomy_label,
        messages=messages,
        target=target,
        evidence_refs=[],
        documents=[],
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": None,
            "reasoning_source": None,
        },
        metadata={
            "phase": "2B",
            "mix_tag": f"internal_{taxonomy_label}",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
        },
    )


def build_evidence(
    sample: Dict[str, Any],
    taxonomy_label: str,
    seed: int,
    *,
    with_reasoning: bool,
) -> Dict[str, Any]:
    refs = resolve_evidence_refs(sample)
    docs = oracle_documents(sample)
    gold = gold_answer_of(sample)
    evidence_block = format_evidence_block(refs)
    if with_reasoning:
        category = "evidence_reasoning"
        reasoning = build_reasoning(refs, gold)
        target = (
            f"<evidence>\n{evidence_block}\n</evidence>\n"
            f"<reasoning>\n{reasoning}\n</reasoning>\n"
            f"<answer>\n{gold}\n</answer>"
        )
        mix_tag = f"evidence_reasoning_{taxonomy_label}"
        reasoning_source = "template_v0"
    else:
        category = "evidence"
        target = (
            f"<evidence>\n{evidence_block}\n</evidence>\n"
            f"<answer>\n{gold}\n</answer>"
        )
        mix_tag = f"evidence_oracle_{taxonomy_label}"
        reasoning_source = None

    user = (
        f"Question: {sample['question']}\n\n"
        f"Documents:\n{format_documents_for_user(docs)}"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return base_row(
        sample,
        category=category,
        taxonomy_label=taxonomy_label,
        messages=messages,
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": {"name": "oracle", "scope": "oracle_supporting_docs"},
            "reasoning_source": reasoning_source,
        },
        metadata={
            "phase": "2B",
            "mix_tag": mix_tag,
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": False,
        },
    )


def gold_titles_covered(
    sample: Dict[str, Any], retrieval_row: Dict[str, Any]
) -> bool:
    gold = {sf["title"] for sf in sample.get("supporting_facts") or []}
    got = {d["title"] for d in retrieval_row.get("documents") or []}
    return bool(gold) and gold <= got


def build_search_format(
    sample: Dict[str, Any],
    taxonomy_label: str,
    seed: int,
    retrieval_row: Dict[str, Any],
) -> Dict[str, Any]:
    if not gold_titles_covered(sample, retrieval_row):
        raise ValueError(
            f"{sample['sample_id']}: gold titles not fully covered by candidate Top-K"
        )
    refs = resolve_evidence_refs(sample)
    docs = list(retrieval_row.get("documents") or [])
    gold = gold_answer_of(sample)
    query = sample["question"]  # heuristic query; teacher_id null
    observation = format_observation(docs)
    evidence_block = format_evidence_block(refs)
    target = (
        f"<search>\n{query}\n</search>\n"
        f"<observation>\n{observation}\n</observation>\n"
        f"<evidence>\n{evidence_block}\n</evidence>\n"
        f"<answer>\n{gold}\n</answer>"
    )
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {sample['question']}"},
    ]
    retriever = dict(retrieval_row.get("retriever") or {})
    return base_row(
        sample,
        category="search_format",
        taxonomy_label=taxonomy_label,
        messages=messages,
        target=target,
        evidence_refs=refs,
        documents=docs,
        provenance={
            "supporting_facts": list(sample.get("supporting_facts") or []),
            "builder": BUILDER_NAME,
            "teacher_id": None,
            "retriever": retriever,
            "reasoning_source": None,
            "query_source": "question_copy",
        },
        metadata={
            "phase": "2B",
            "mix_tag": f"search_format_candidate_{taxonomy_label}",
            "seed": seed,
            "level": (sample.get("metadata") or {}).get("level"),
            "type": (sample.get("metadata") or {}).get("type"),
            "observation_in_target": True,
            "observation_text": observation,
            "loss_mask_note": (
                "Exclude <observation>...</observation> body from SFT loss; "
                "search/evidence/answer enter loss."
            ),
            "note": "candidate scope ≠ full corpus",
        },
    )


def parse_tagged_target(target: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for name, pattern in _TAG_RE.items():
        found[name] = [m.group(1).strip() for m in pattern.finditer(target or "")]
    return found


def validate_sft_row(row: Dict[str, Any]) -> List[str]:
    """Return human-readable errors; empty list means accept."""
    errors: List[str] = []
    sid = row.get("sft_id", "<unknown>")

    for key in (
        "sample_id",
        "sft_id",
        "source_dataset",
        "category",
        "taxonomy_label",
        "messages",
        "target",
        "gold_answer",
        "gold_answers",
        "evidence_refs",
        "documents",
        "provenance",
        "metadata",
    ):
        if key not in row:
            errors.append(f"{sid}: missing field {key}")

    category = row.get("category")
    if category not in CATEGORIES:
        errors.append(f"{sid}: invalid category {category!r}")

    messages = row.get("messages") or []
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"{sid}: messages must be [system, user, ...]")
    else:
        roles = [m.get("role") for m in messages]
        if roles[0] != "system" or "user" not in roles:
            errors.append(f"{sid}: messages must start with system and include user")
        if any(m.get("role") == "assistant" for m in messages):
            errors.append(f"{sid}: assistant must not be in messages (use target)")

    target = row.get("target") or ""
    tags = parse_tagged_target(target)
    if len(tags.get("answer") or []) != 1:
        errors.append(f"{sid}: target must contain exactly one <answer>")
    else:
        pred = tags["answer"][0]
        gold = row.get("gold_answer") or ""
        if normalize_answer(pred) != normalize_answer(gold):
            errors.append(
                f"{sid}: answer != gold after normalize "
                f"({pred!r} vs {gold!r})"
            )
        # answer must be last tag content-wise: answer closes at end
        if not re.search(
            r"<answer>.*?</answer>\s*\Z", target, flags=re.DOTALL | re.IGNORECASE
        ):
            errors.append(f"{sid}: <answer> must be the final tag in target")

    has_internal = bool(tags.get("internal"))
    has_search = bool(tags.get("search"))
    if has_internal and has_search:
        errors.append(f"{sid}: internal and search are mutually exclusive")

    if category == "internal":
        if not has_internal or has_search or tags.get("evidence") or tags.get(
            "reasoning"
        ):
            errors.append(f"{sid}: internal template invalid")
        if row.get("evidence_refs"):
            errors.append(f"{sid}: internal must have empty evidence_refs")
    elif category == "evidence":
        if has_internal or has_search or not tags.get("evidence") or tags.get(
            "reasoning"
        ):
            errors.append(f"{sid}: evidence template invalid")
    elif category == "evidence_reasoning":
        if (
            has_internal
            or has_search
            or not tags.get("evidence")
            or not tags.get("reasoning")
        ):
            errors.append(f"{sid}: evidence_reasoning template invalid")
        else:
            reasoning = tags["reasoning"][0]
            gold_n = normalize_answer(row.get("gold_answer") or "")
            reasoning_n = normalize_answer(reasoning)
            if gold_n and gold_n not in reasoning_n:
                errors.append(f"{sid}: reasoning not answer-consistent")
            # template_v0 embeds evidence spans; Kimi teacher is bridge-only
            # and must NOT paste evidence verbatim.
            reasoning_source = (row.get("provenance") or {}).get("reasoning_source")
            require_evidence_spans = reasoning_source not in {
                "kimi2.6",
                "teacher",
                "kimi",
            }
            if require_evidence_spans:
                for ref in row.get("evidence_refs") or []:
                    frag = whitespace_norm(ref.get("text") or "")
                    if frag and frag not in whitespace_norm(reasoning):
                        errors.append(
                            f"{sid}: reasoning missing evidence text from "
                            f"{ref.get('title')}"
                        )
    elif category == "search_format":
        if (
            has_internal
            or not has_search
            or not tags.get("observation")
            or not tags.get("evidence")
        ):
            errors.append(f"{sid}: search_format template invalid")
        retr = (row.get("provenance") or {}).get("retriever") or {}
        if retr.get("scope") == "full_corpus":
            errors.append(f"{sid}: prototype forbids full_corpus scope")

    # evidence_refs ↔ contexts consistency for non-internal
    if category != "internal":
        refs = row.get("evidence_refs") or []
        if not refs:
            errors.append(f"{sid}: evidence_refs empty for {category}")
        for ref in refs:
            for k in ("document_id", "title", "sentence_id", "text"):
                if k not in ref:
                    errors.append(f"{sid}: evidence_ref missing {k}")
            if whitespace_norm(ref.get("text", "")) == "":
                errors.append(f"{sid}: empty evidence text")

        # target evidence block must match refs (whitespace-norm)
        if tags.get("evidence"):
            body = tags["evidence"][0]
            parsed = list(_EVIDENCE_LINE_RE.finditer(body))
            if len(parsed) != len(refs):
                errors.append(
                    f"{sid}: evidence block count {len(parsed)} != refs {len(refs)}"
                )
            else:
                for m, ref in zip(parsed, refs):
                    if m.group("document_id").strip() != ref["document_id"]:
                        errors.append(f"{sid}: evidence document_id mismatch")
                    if m.group("title").strip() != ref["title"]:
                        errors.append(f"{sid}: evidence title mismatch")
                    if int(m.group("sentence_id")) != int(ref["sentence_id"]):
                        errors.append(f"{sid}: evidence sentence_id mismatch")
                    if whitespace_norm(m.group("text")) != whitespace_norm(ref["text"]):
                        errors.append(f"{sid}: evidence text mismatch (ws-norm)")

    return errors


def assign_and_build(
    eval_rows: Sequence[Dict[str, Any]],
    taxonomy: Dict[str, str],
    retrieval: Dict[str, Dict[str, Any]],
    *,
    seed: int = 42,
    max_search_format: int = 60,
    include_c_evidence_dual: bool = True,
    max_c_evidence_dual: int = 40,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build prototype rows. Returns (accepted, rejected_info)."""
    import random

    rng = random.Random(seed)
    samples_by_id = {s["sample_id"]: s for s in eval_rows}
    by_label: Dict[str, List[str]] = {k: [] for k in "ABCDEO"}
    for sid, sample in samples_by_id.items():
        label = taxonomy.get(sid, "unknown")
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(sid)

    for lab in by_label:
        by_label[lab].sort()
        rng.shuffle(by_label[lab])

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    def try_add(row: Dict[str, Any]) -> None:
        errs = validate_sft_row(row)
        if errs:
            rejected.append({"sft_id": row.get("sft_id"), "errors": errs})
        else:
            accepted.append(row)

    # Type I: all D + E
    for lab in ("E", "D"):
        for sid in by_label.get(lab, []):
            try:
                try_add(build_internal(samples_by_id[sid], lab, seed))
            except Exception as exc:  # noqa: BLE001
                rejected.append(
                    {"sft_id": f"{sid}__internal__{VIEW_TAG}", "errors": [str(exc)]}
                )

    # Type III: all C → evidence_reasoning
    for sid in by_label.get("C", []):
        try:
            try_add(
                build_evidence(samples_by_id[sid], "C", seed, with_reasoning=True)
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "sft_id": f"{sid}__evidence_reasoning__{VIEW_TAG}",
                    "errors": [str(exc)],
                }
            )

    # Type II: all A + B → evidence
    for lab in ("A", "B"):
        for sid in by_label.get(lab, []):
            try:
                try_add(
                    build_evidence(
                        samples_by_id[sid], lab, seed, with_reasoning=False
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rejected.append(
                    {"sft_id": f"{sid}__evidence__{VIEW_TAG}", "errors": [str(exc)]}
                )

    # Dual view: some C also as plain evidence (optional)
    if include_c_evidence_dual:
        c_ids = list(by_label.get("C", []))
        for sid in c_ids[:max_c_evidence_dual]:
            try:
                row = build_evidence(
                    samples_by_id[sid], "C", seed, with_reasoning=False
                )
                row["sft_id"] = make_sft_id(sid, "evidence", "dual_v0")
                row["metadata"]["mix_tag"] = "evidence_oracle_C_dual"
                try_add(row)
            except Exception as exc:  # noqa: BLE001
                rejected.append(
                    {"sft_id": f"{sid}__evidence__dual_v0", "errors": [str(exc)]}
                )

    # Type IV: search_format from A/B with full gold title coverage
    eligible: List[Tuple[str, str]] = []
    for lab in ("A", "B"):
        for sid in by_label.get(lab, []):
            cache = retrieval.get(sid)
            if cache and gold_titles_covered(samples_by_id[sid], cache):
                eligible.append((sid, lab))
    rng.shuffle(eligible)
    for sid, lab in eligible[:max_search_format]:
        try:
            try_add(
                build_search_format(
                    samples_by_id[sid], lab, seed, retrieval[sid]
                )
            )
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {"sft_id": f"{sid}__search_format__{VIEW_TAG}", "errors": [str(exc)]}
            )

    # stable output order: by category then sft_id
    cat_order = {
        "internal": 0,
        "evidence": 1,
        "evidence_reasoning": 2,
        "search_format": 3,
    }
    accepted.sort(key=lambda r: (cat_order.get(r["category"], 9), r["sft_id"]))
    return accepted, rejected


def summarize(
    accepted: Sequence[Dict[str, Any]], rejected: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    from collections import Counter

    by_cat = Counter(r["category"] for r in accepted)
    by_tax = Counter(r["taxonomy_label"] for r in accepted)
    n = len(accepted)
    return {
        "n_accepted": n,
        "n_rejected": len(rejected),
        "by_category": dict(by_cat),
        "by_taxonomy_label": dict(by_tax),
        "category_rates": {k: round(v / n, 4) if n else 0.0 for k, v in by_cat.items()},
        "evidence_text_equality": "whitespace_normalized",
        "reasoning_source": "template_v0",
        "search_query_source": "question_copy",
        "builder": BUILDER_NAME,
    }

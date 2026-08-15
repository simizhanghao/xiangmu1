"""Phase 2D3-C: parse agent protocol tags and score evidence citations.

Deterministic only (no LLM judge). Evidence keys are (title, sentence_id).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.eval.metrics import exact_match, normalize_answer, token_f1
from src.sft.prototype_builder import (
    AGENT_SYSTEM_PROMPT,
    _EVIDENCE_LINE_RE,
    parse_tagged_target,
    resolve_evidence_refs,
    whitespace_norm,
)

EvidenceKey = Tuple[str, int]

TAG_NAMES = ("internal", "search", "observation", "evidence", "reasoning", "answer")


def extract_tags(text: str) -> Dict[str, List[str]]:
    """Parse protocol tags; `reasoning` matches both <think> and <reasoning>."""
    return parse_tagged_target(text or "")


def parse_evidence_keys(evidence_body: str) -> List[Dict[str, Any]]:
    """Parse evidence lines → list of {document_id,title,sentence_id,text}."""
    refs: List[Dict[str, Any]] = []
    for m in _EVIDENCE_LINE_RE.finditer(evidence_body or ""):
        refs.append(
            {
                "document_id": m.group("document_id").strip(),
                "title": m.group("title").strip(),
                "sentence_id": int(m.group("sentence_id")),
                "text": whitespace_norm(m.group("text")),
            }
        )
    return refs


def gold_evidence_keys(sample: Dict[str, Any]) -> Set[EvidenceKey]:
    refs = resolve_evidence_refs(sample)
    return {(r["title"], int(r["sentence_id"])) for r in refs}


def pred_evidence_keys(refs: Sequence[Dict[str, Any]]) -> Set[EvidenceKey]:
    return {(r["title"], int(r["sentence_id"])) for r in refs}


def set_prf(pred: Set[EvidenceKey], gold: Set[EvidenceKey]) -> Dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def first_tag_body(tags: Dict[str, List[str]], name: str) -> Optional[str]:
    vals = tags.get(name) or []
    return vals[0] if vals else None


def protocol_flags_evidence_use(text: str) -> Dict[str, Any]:
    """Validity for docs-conditioned generation (evidence / think / answer)."""
    tags = extract_tags(text)
    has_ev = bool(tags.get("evidence"))
    has_think = bool(tags.get("reasoning"))
    has_ans = bool(tags.get("answer"))
    has_internal = bool(tags.get("internal"))
    has_search = bool(tags.get("search"))
    # Valid if answer present and not mixing routing tags; evidence preferred.
    protocol_valid = bool(
        has_ans and (has_ev or has_think) and not (has_internal and has_search)
    )
    return {
        "answer_tag": has_ans,
        "evidence_tag": has_ev,
        "think_tag": has_think,
        "internal_tag": has_internal,
        "search_tag": has_search,
        "protocol_valid": protocol_valid,
        "tags": {k: len(tags.get(k) or []) for k in TAG_NAMES},
    }


def protocol_flags_routing(text: str) -> Dict[str, Any]:
    """Validity for question-only routing (internal XOR search)."""
    tags = extract_tags(text)
    has_internal = bool(tags.get("internal"))
    has_search = bool(tags.get("search"))
    has_ans = bool(tags.get("answer"))
    exclusive = has_internal != has_search  # XOR
    if has_internal:
        protocol_valid = exclusive and has_ans and not has_search
        route = "internal"
    elif has_search:
        protocol_valid = exclusive and not has_internal
        route = "search"
    else:
        protocol_valid = False
        route = "none"
    return {
        "route": route,
        "answer_tag": has_ans,
        "internal_tag": has_internal,
        "search_tag": has_search,
        "evidence_tag": bool(tags.get("evidence")),
        "think_tag": bool(tags.get("reasoning")),
        "protocol_valid": protocol_valid,
        "tags": {k: len(tags.get(k) or []) for k in TAG_NAMES},
    }


def score_evidence_use(
    generation: str,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    tags = extract_tags(generation)
    flags = protocol_flags_evidence_use(generation)
    answer = first_tag_body(tags, "answer") or ""
    # Fallback: if no answer tag, use full generation for EM (still flagged invalid)
    pred_for_qa = answer if answer else generation.strip()
    golds = sample.get("gold_answers") or []
    ev_body = first_tag_body(tags, "evidence") or ""
    pred_refs = parse_evidence_keys(ev_body)
    gold_keys = gold_evidence_keys(sample)
    pred_keys = pred_evidence_keys(pred_refs)
    prf = set_prf(pred_keys, gold_keys)
    return {
        **flags,
        "prediction": pred_for_qa,
        "evidence_precision": prf["precision"],
        "evidence_recall": prf["recall"],
        "evidence_f1": prf["f1"],
        "n_pred_evidence": len(pred_keys),
        "n_gold_evidence": len(gold_keys),
        "exact_match": exact_match(pred_for_qa, golds),
        "token_f1": token_f1(pred_for_qa, golds),
        "think_text": first_tag_body(tags, "reasoning"),
        "pred_evidence_refs": pred_refs,
    }


def score_routing(generation: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    tags = extract_tags(generation)
    flags = protocol_flags_routing(generation)
    answer = first_tag_body(tags, "answer") or ""
    search_q = first_tag_body(tags, "search")
    golds = sample.get("gold_answers") or []
    pred_for_qa = answer if answer else ""
    return {
        **flags,
        "prediction": pred_for_qa,
        "search_query": (search_q or "").strip() if search_q is not None else None,
        "exact_match": exact_match(pred_for_qa, golds) if pred_for_qa else 0.0,
        "token_f1": token_f1(pred_for_qa, golds) if pred_for_qa else 0.0,
        "internal_text": first_tag_body(tags, "internal"),
    }


def agent_system_prompt() -> str:
    return AGENT_SYSTEM_PROMPT

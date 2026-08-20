"""Evidence GRPO reward: R = Answer + λ_e EvidenceF1 + 0.1 Format − λ_s N_search.

Deterministic supporting-fact F1 on (title, sentence_id).
Default λ_e=0.5; λ_s=0. Set ECA_SEARCH_COST_WEIGHT or reward_weights for cost.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from src.eval.metrics import exact_match
from src.eval.protocol import extract_tags, first_tag_body, parse_evidence_keys, pred_evidence_keys, set_prf
from src.rl.reward_breakdown import RewardWeights, combine_rewards, weights_from_mapping

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_INTERNAL_RE = re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE)

EvidenceKey = Tuple[str, int]


def extract_answer(solution_str: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(solution_str or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def _as_gold_list(ground_truth: Any) -> List[str]:
    if ground_truth is None:
        return []
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", ground_truth.get("gold_answers"))
        return _as_gold_list(target)
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, Sequence) and not isinstance(ground_truth, (str, bytes)):
        return [str(x) for x in ground_truth]
    return [str(ground_truth)]


def _gold_sf_keys(ground_truth: Any, extra_info: Dict[str, Any] | None) -> Set[EvidenceKey]:
    """Gold evidence keys from reward-visible channels only (never from prompt)."""
    sf = None
    if isinstance(ground_truth, dict):
        sf = ground_truth.get("supporting_facts")
    if not sf and isinstance(extra_info, dict):
        sf = extra_info.get("supporting_facts")
    keys: Set[EvidenceKey] = set()
    for item in sf or []:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        sid = item.get("sentence_id", item.get("sent_id"))
        if title is None or sid is None:
            continue
        try:
            keys.add((str(title), int(sid)))
        except (TypeError, ValueError):
            continue
    return keys


def format_valid(solution_str: str) -> float:
    text = solution_str or ""
    ans = extract_answer(text)
    if not ans:
        return 0.0
    cut = text[: _ANSWER_RE.search(text).end()]  # type: ignore[union-attr]
    if bool(_SEARCH_RE.search(cut)) and bool(_INTERNAL_RE.search(cut)):
        return 0.0
    return 1.0


def evidence_f1_score(solution_str: str, gold_keys: Set[EvidenceKey]) -> Dict[str, float]:
    """Empty/malformed <evidence> → F1=0. Otherwise set-F1 vs gold SF."""
    tags = extract_tags(solution_str or "")
    body = first_tag_body(tags, "evidence") or ""
    nonempty = 1.0 if body.strip() else 0.0
    refs = parse_evidence_keys(body)
    pred_keys = pred_evidence_keys(refs)
    valid = 1.0 if (nonempty and len(refs) > 0) else 0.0
    if not gold_keys:
        prf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        prf = set_prf(pred_keys, gold_keys)
        if not nonempty or not refs:
            prf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "evidence_precision": float(prf["precision"]),
        "evidence_recall": float(prf["recall"]),
        "evidence_f1": float(prf["f1"]),
        "evidence_nonempty": nonempty,
        "evidence_valid": valid,
        "n_pred_evidence": float(len(pred_keys)),
        "n_gold_evidence": float(len(gold_keys)),
    }


def _weights(extra_info: Dict[str, Any] | None) -> RewardWeights:
    mapping: Dict[str, Any] = {}
    if isinstance(extra_info, dict) and isinstance(extra_info.get("reward_weights"), dict):
        mapping.update(extra_info["reward_weights"])
    env_e = os.environ.get("ECA_EVIDENCE_WEIGHT")
    if env_e is not None:
        mapping["evidence_weight"] = float(env_e)
    env_s = os.environ.get("ECA_SEARCH_COST_WEIGHT")
    if env_s is not None:
        mapping["search_cost_weight"] = float(env_s)
    return weights_from_mapping(mapping)


def _search_count(solution_str: str, extra_info: Dict[str, Any] | None) -> float:
    if isinstance(extra_info, dict):
        if extra_info.get("search_count") is not None:
            return float(extra_info["search_count"])
        cost = extra_info.get("cost_info")
        if isinstance(cost, dict) and cost.get("search_count") is not None:
            return float(cost["search_count"])
    return float(len(_SEARCH_RE.findall(solution_str or "")))


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    del data_source, kwargs
    golds = _as_gold_list(ground_truth)
    pred = extract_answer(solution_str) or ""
    em = float(exact_match(pred, golds)) if pred else 0.0
    fmt = float(format_valid(solution_str))
    gold_keys = _gold_sf_keys(ground_truth, extra_info)
    ev = evidence_f1_score(solution_str, gold_keys)
    n_search = _search_count(solution_str, extra_info)
    w = _weights(extra_info)
    br = combine_rewards(
        answer=em,
        evidence=ev["evidence_f1"],
        format_r=fmt,
        cost=n_search,
        weights=w,
    )
    return {
        "score": br.total,
        "total_reward": br.total,
        "em": em,
        "answer_em": em,
        "answer_reward": br.answer_reward,
        "format": fmt,
        "format_reward": br.format_reward,
        "evidence_reward": br.evidence_reward,
        "search_count": n_search,
        "cost_reward": br.cost_reward,
        "search_cost_weight": w.search_cost_weight,
        "evidence_precision": ev["evidence_precision"],
        "evidence_recall": ev["evidence_recall"],
        "evidence_f1": ev["evidence_f1"],
        "evidence_nonempty": ev["evidence_nonempty"],
        "evidence_valid": ev["evidence_valid"],
        "n_pred_evidence": ev["n_pred_evidence"],
        "n_gold_evidence": ev["n_gold_evidence"],
        "answer_weight": w.answer_weight,
        "evidence_weight": w.evidence_weight,
        "format_weight": w.format_weight,
        "pred": pred,
        "gold": golds[0] if golds else "",
        "sample_id": (extra_info or {}).get("sample_id"),
    }

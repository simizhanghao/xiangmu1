"""Basic deterministic evaluation metrics over TraceRecord.

Answers "given one unified-format trace, how do we score it with fixed
rules": answer quality (EM / token F1), search cost facts (search count,
lexical duplicate queries), and structural validity.

Metrics report facts only. Reward weighting, clipping, and penalties belong
to the reward layer (src/rewards/), never here. English short-answer QA
normalization (SQuAD style) for the current baseline; whitespace
tokenization, no multilingual support. Standard library only.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Sequence, Union

from src.eval.trace_schema import TraceRecord, validate_trace_record

_ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize an English short answer (SQuAD style).

    Lowercase, strip punctuation, drop standalone articles (a/an/the),
    and collapse consecutive whitespace.
    """
    lowered = text.lower()
    without_punc = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    without_articles = _ARTICLES_PATTERN.sub(" ", without_punc)
    return " ".join(without_articles.split())


def _as_gold_list(gold_answers: Union[str, Sequence[str]]) -> List[str]:
    """Coerce a single gold string or a sequence of golds into a list."""
    if isinstance(gold_answers, str):
        return [gold_answers]
    return list(gold_answers)


def exact_match(
    prediction: str,
    gold_answers: Union[str, Sequence[str]],
) -> float:
    """Return 1.0 if the normalized prediction equals any normalized gold.

    Takes the max over golds; an empty gold sequence yields 0.0.
    """
    golds = _as_gold_list(gold_answers)
    if not golds:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    for gold in golds:
        if normalize_answer(gold) == normalized_prediction:
            return 1.0
    return 0.0


def _f1_single(prediction_tokens: List[str], gold_tokens: List[str]) -> float:
    """Token-multiset F1 between one prediction and one gold."""
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_prf(
    prediction: str,
    gold_answers: Union[str, Sequence[str]],
) -> Dict[str, float]:
    """Token P/R/F1 after normalization; keep the gold with max F1 (Hotpot-style)."""
    golds = _as_gold_list(gold_answers)
    if not golds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prediction_tokens = normalize_answer(prediction).split()
    best = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not prediction_tokens and not gold_tokens:
            cand = {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        elif not prediction_tokens or not gold_tokens:
            cand = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            common = Counter(prediction_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                cand = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            else:
                precision = num_same / len(prediction_tokens)
                recall = num_same / len(gold_tokens)
                cand = {
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall),
                }
        if cand["f1"] > best["f1"]:
            best = cand
    if best["f1"] < 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return best


def token_f1(
    prediction: str,
    gold_answers: Union[str, Sequence[str]],
) -> float:
    """Whitespace-token F1 after normalization, max over golds.

    Both sides empty -> 1.0 for that gold; exactly one side empty -> 0.0;
    an empty gold sequence yields 0.0 overall.
    """
    return float(token_prf(prediction, gold_answers)["f1"])


def hotpot_joint(
    answer_em: float,
    answer_precision: float,
    answer_recall: float,
    evidence_em: float,
    evidence_precision: float,
    evidence_recall: float,
) -> Dict[str, float]:
    """Official HotpotQA joint: product of answer and supporting-fact P/R.

    Report-only. Do not use for checkpoint selection.
    """
    joint_precision = float(answer_precision) * float(evidence_precision)
    joint_recall = float(answer_recall) * float(evidence_recall)
    if joint_precision + joint_recall > 0:
        joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    else:
        joint_f1 = 0.0
    return {
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "joint_f1": joint_f1,
        "joint_em": float(answer_em) * float(evidence_em),
    }


def count_search_steps(record: TraceRecord) -> int:
    """Number of steps with step_type == "search"."""
    return sum(1 for step in record.steps if step.step_type == "search")


def count_duplicate_queries(record: TraceRecord) -> int:
    """Count lexically duplicated search queries after normalization.

    The first occurrence of a query is not a duplicate; every later
    occurrence counts as one (3 identical queries -> 2 duplicates).
    Lexical (normalized-string) comparison only; no semantic matching.
    """
    seen: Counter = Counter()
    duplicates = 0
    for step in record.steps:
        if step.step_type != "search":
            continue
        query = normalize_answer(step.content)
        if seen[query] > 0:
            duplicates += 1
        seen[query] += 1
    return duplicates


def format_valid(record: TraceRecord) -> bool:
    """True iff the record passes structural validation."""
    return len(validate_trace_record(record)) == 0


def basic_metrics(record: TraceRecord) -> Dict[str, Union[float, int, bool]]:
    """Compute the fixed basic metric dict for one TraceRecord.

    Search metrics are recomputed from steps; pre-filled CostInfo values
    are neither read nor trusted. The record is not modified.
    """
    return {
        "exact_match": exact_match(record.final_answer, record.gold_answers),
        "token_f1": token_f1(record.final_answer, record.gold_answers),
        "search_count": count_search_steps(record),
        "duplicate_query_count": count_duplicate_queries(record),
        "format_valid": format_valid(record),
    }

#!/usr/bin/env python3
"""Behavior matrix for frozen hidden depth labels versus Web Agent traces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import token_f1

_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
_NORM = re.compile(r"[^a-z0-9]+")
_STOP = {"the", "and", "for", "was", "what", "which", "who", "where", "with", "from", "that", "this"}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def words(value: str) -> set[str]:
    return {x for x in _WORD.findall(value.lower()) if x not in _STOP}


def qnorm(value: str) -> str:
    return _NORM.sub(" ", value.lower()).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("annotations", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--baseline-answer-f1", type=float)
    a = p.parse_args()
    labels = {str(x["sample_id"]): int(x["minimal_depth"]) for x in load(a.annotations)}
    traces = [x for x in load(a.trace) if str(x.get("sample_id")) in labels and labels[str(x.get("sample_id"))] in (1, 2)]
    d1 = d2 = d1_stop = d2_continue = finish = 0
    extra = duplicate = conditioned = new_evidence = 0
    f1s: list[float] = []
    for row in traces:
        depth = labels[str(row["sample_id"])]
        searches = [x for x in row.get("steps") or [] if x.get("step_type") == "search"]
        observations = [x for x in row.get("steps") or [] if x.get("step_type") == "observation"]
        answers = [x for x in row.get("steps") or [] if x.get("step_type") == "answer"]
        finish += int(bool(answers and str(answers[-1].get("content") or "").strip()))
        if answers:
            f1s.append(max(token_f1(str(answers[-1].get("content") or ""), [str(x)]) for x in row.get("gold_answers") or [""]))
        if depth == 1:
            d1 += 1
            d1_stop += int(len(searches) == 1)
        else:
            d2 += 1
            d2_continue += int(len(searches) >= 2)
        for i in range(1, len(searches)):
            extra += 1
            query = str(searches[i].get("content") or "")
            duplicate += int(qnorm(query) in {qnorm(str(x.get("content") or "")) for x in searches[:i]})
            prior_obs = str(observations[i - 1].get("content") or "") if i - 1 < len(observations) else ""
            conditioned += int(bool((words(query) - words(str(row.get("question") or ""))) & words(prior_obs)))
            prior_docs = {d for x in observations[:i] for d in x.get("document_ids") or []}
            current_docs = set(observations[i].get("document_ids") or []) if i < len(observations) else set()
            new_evidence += int(bool(current_docs - prior_docs))
    mean_f1 = sum(f1s) / max(1, len(f1s))
    metrics = {
        "gate": "WEB_ADAPTIVE_DEPTH_FAIL",
        "evaluated": len(traces),
        "depth1_count": d1,
        "depth2_count": d2,
        "stop_at_d1": d1_stop / max(1, d1),
        "continue_at_d2": d2_continue / max(1, d2),
        "finish_rate": finish / max(1, len(traces)),
        "duplicate_extra_query_rate": duplicate / max(1, extra),
        "obs_conditioned_extra_query_rate": conditioned / max(1, extra),
        "new_evidence_extra_query_rate": new_evidence / max(1, extra),
        "answer_f1": mean_f1,
    }
    answer_ok = a.baseline_answer_f1 is None or mean_f1 >= a.baseline_answer_f1 - 0.02
    passed = (
        d1 > 0 and d2 > 0 and metrics["finish_rate"] >= 0.95
        and metrics["duplicate_extra_query_rate"] <= 0.25
        and metrics["obs_conditioned_extra_query_rate"] >= 0.40
        and metrics["new_evidence_extra_query_rate"] >= 0.30
        and answer_ok
    )
    metrics["gate"] = "WEB_MULTITURN_SFT_PASS" if passed else "WEB_ADAPTIVE_DEPTH_FAIL"
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

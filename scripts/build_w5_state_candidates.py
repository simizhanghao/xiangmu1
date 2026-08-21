#!/usr/bin/env python3
"""Convert natural Search1 traces into leakage-safe W5 state/checker candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.research_memory import ResearchMemory
from src.eval.metrics import normalize_answer
from src.tools.candidate_bm25 import format_observation_text


def title_norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def answer_in(text: str, answers: list[str]) -> bool:
    haystack = normalize_answer(text)
    return any(len(normalize_answer(x)) >= 2 and normalize_answer(x) in haystack for x in answers)


def render_state(question: str, query: str, docs: list[dict[str, Any]], budget: int) -> tuple[str, str, str]:
    memory = ResearchMemory(question=question, max_searches=budget, evidence_limit=8, char_budget=5000)
    memory.add_search(query, docs)
    observation = format_observation_text(docs)
    rendered = memory.render()
    controller_input = f"Question: {question}\n\nCurrent Observation:\n{observation}\n\n{rendered}"
    return observation, rendered, controller_input


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--questions", default="data/w5_controller/controller_train4500.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--deployment-budget", type=int, default=4)
    args = p.parse_args()

    questions = {x["sample_id"]: x for x in map(json.loads, Path(args.questions).open())}
    natural, masked = [], []
    reasons: dict[str, int] = {}
    for trace in map(json.loads, Path(args.trace).open()):
        sample = questions[trace["sample_id"]]
        docs = list(trace.get("documents") or [])
        query = str((trace.get("metadata") or {}).get("search_queries", [sample["question"]])[0])
        if not docs:
            reasons["no_documents"] = reasons.get("no_documents", 0) + 1
            continue
        answers = [str(x) for x in sample.get("gold_answers") or []]
        yes_no = all(normalize_answer(x) in {"yes", "no"} for x in answers)
        visible = [d["document_id"] for d in docs if answer_in(str(d.get("text") or ""), answers)]
        gold_titles = {title_norm(x["title"]) for x in sample.get("supporting_facts") or []}
        found_titles = {title_norm(d.get("title") or "") for d in docs}
        title_hits = sum(any(g and (g in f or f in g) for f in found_titles) for g in gold_titles)
        title_recall = title_hits / max(1, len(gold_titles))
        # These are checker candidates, not final labels. Ambiguous states are sent
        # to the grounded checker rather than silently treated as gold.
        if yes_no:
            candidate = "needs_grounded_checker"
        elif visible:
            candidate = "sufficient_candidate"
        else:
            candidate = "insufficient_candidate"
        observation, memory, controller_input = render_state(
            sample["question"], query, docs, args.deployment_budget
        )
        base = {
            "state_id": f'{trace["sample_id"]}__search1_natural',
            "sample_id": trace["sample_id"],
            "controller_split": sample.get("controller_split"),
            "state_origin": "natural_bocha_on_policy",
            "question": sample["question"],
            "previous_queries": [query],
            "documents": docs,
            "observation": observation,
            "research_memory": memory,
            "controller_input": controller_input,
            "label_candidate": candidate,
            "builder_audit": {
                "gold_answers": answers,
                "supporting_titles": sorted(gold_titles),
                "gold_answer_visible_document_ids": visible,
                "supporting_title_recall": title_recall,
                "yes_no_requires_checker": yes_no,
                "gold_fields_runtime_visible": False,
            },
        }
        natural.append(base)
        if candidate == "sufficient_candidate":
            kept = [d for d in docs if d["document_id"] not in set(visible)]
            if kept and not any(answer_in(str(d.get("text") or ""), answers) for d in kept):
                m_obs, m_mem, m_input = render_state(sample["question"], query, kept, args.deployment_budget)
                masked.append({
                    **{k: v for k, v in base.items() if k not in {"documents", "observation", "research_memory", "controller_input", "builder_audit"}},
                    "state_id": f'{trace["sample_id"]}__search1_masked',
                    "state_origin": "counterfactual_evidence_mask",
                    "documents": kept,
                    "observation": m_obs,
                    "research_memory": m_mem,
                    "controller_input": m_input,
                    "label_candidate": "insufficient_masked_candidate",
                    "paired_natural_state_id": base["state_id"],
                    "builder_audit": {
                        "gold_answers": answers,
                        "supporting_titles": sorted(gold_titles),
                        "masked_document_ids": visible,
                        "answer_visible_after_mask": False,
                        "gold_fields_runtime_visible": False,
                    },
                })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("natural_states.jsonl", natural), ("masked_siblings.jsonl", masked)):
        with (out / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in natural:
        key = row["label_candidate"]
        counts[key] = counts.get(key, 0) + 1
    summary = {
        "gate": "W5_STATE_CANDIDATE_SMOKE_PASS" if natural else "W5_STATE_CANDIDATE_SMOKE_FAIL",
        "trace_rows": sum(1 for _ in Path(args.trace).open()),
        "natural_states": len(natural),
        "masked_siblings": len(masked),
        "candidate_counts": counts,
        "skip_reasons": reasons,
        "deployment_budget": args.deployment_budget,
        "final_labels_assigned": False,
        "requires_grounded_checker": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

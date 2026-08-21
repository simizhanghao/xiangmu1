#!/usr/bin/env python3
"""Adjudicate W5 checker labels and enforce leakage/split/data gates."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import token_f1

STATE_DIR = Path("results/72_w5_state_dataset/checker_candidates")
LABELS = Path("results/73_w5_grounded_checker/full/labels.jsonl")
OUT = Path("results/74_w5_controller_data")


def read(path: Path):
    return [json.loads(x) for x in path.open() if x.strip()]


def main() -> None:
    states = {}
    for name in ("natural_states.jsonl", "masked_siblings.jsonl"):
        for row in read(STATE_DIR / name):
            states[row["state_id"]] = row
    labels = read(LABELS)
    unique_labels = {x["state_id"]: x for x in labels}
    rows, counts = [], Counter()
    train_ids, dev_ids = set(), set()
    for state_id, state in states.items():
        label = unique_labels[state_id]
        teacher = label["teacher"]
        decision = teacher["decision"]
        reason = "checker"
        if decision == "STOP":
            golds = (state.get("builder_audit") or {}).get("gold_answers") or []
            score = token_f1(str(teacher.get("grounded_answer") or ""), golds)
            valid_ids = all(
                str(x).startswith("D") and str(x)[1:].isdigit()
                and 1 <= int(str(x)[1:]) <= len(state.get("documents") or [])
                for x in teacher.get("source_ids") or []
            )
            if score < 0.5 or not valid_ids or not teacher.get("source_ids"):
                decision, reason = "CONTINUE", "conservative_stop_rejection"
                counts["rejected_stop"] += 1
            else:
                counts["accepted_stop"] += 1
        split = state["controller_split"]
        (train_ids if split == "train" else dev_ids).add(state["sample_id"])
        counts[(split, state["state_origin"], decision)] += 1
        # Export no gold, supporting facts, checker answer, or rationale.
        rows.append({
            "state_id": state_id,
            "sample_id": state["sample_id"],
            "controller_split": split,
            "state_origin": state["state_origin"],
            "decision": decision,
            "adjudication": reason,
            "controller_input": state["controller_input"],
            "previous_queries": state["previous_queries"],
            "documents": state["documents"],
        })
    masked = [x for x in rows if x["state_origin"] == "counterfactual_evidence_mask"]
    masked_continue = sum(x["decision"] == "CONTINUE" for x in masked) / len(masked)
    forbidden = ("gold_answers", "supporting_facts", "reference_answers", "teacher")
    structural_leaks = sum(any(key in row for key in forbidden) for row in rows)
    passed = (
        len(labels) == len(unique_labels) == len(states) == 6256
        and all(x["teacher"].get("ok") for x in labels)
        and not (train_ids & dev_ids) and structural_leaks == 0
        and masked_continue >= 0.90 and counts["accepted_stop"] >= 800
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "adjudicated_states.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "gate": "W5_GROUNDED_LABEL_GATE_PASS" if passed else "W5_GROUNDED_LABEL_GATE_FAIL",
        "states": len(states), "labels": len(labels), "unique_labels": len(unique_labels),
        "accepted_stop": counts["accepted_stop"], "rejected_stop": counts["rejected_stop"],
        "final_continue": sum(x["decision"] == "CONTINUE" for x in rows),
        "masked_continue_rate": masked_continue,
        "train_questions": len(train_ids), "dev_questions": len(dev_ids),
        "question_overlap": len(train_ids & dev_ids), "structural_gold_leaks": structural_leaks,
        "distribution": {"|".join(k): v for k, v in counts.items() if isinstance(k, tuple)},
        "output": str(OUT / "adjudicated_states.jsonl"),
    }
    (OUT / "grounded_label_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

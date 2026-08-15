#!/usr/bin/env python3
"""Select the unique GRPO checkpoint using the rule frozen before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = (200, 400, 600, 800)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    root = args.project_root
    rows = []
    missing = []
    for step in STEPS:
        path = root / "results" / "frozen_dev" / f"step{step}" / "summary.json"
        if not path.exists():
            missing.append(step)
            continue
        s = json.loads(path.read_text())
        health = (
            float(s["finish_rate"]) >= 0.95
            and float(s["parse_ok_rate"]) >= 0.95
            and float(s["observation_mask_ok_rate"]) == 1.0
        )
        rank = (
            int(health),
            float(s["mean_token_f1"]),
            float(s["mean_evidence_f1"]),
            float(s["mean_em"]),
            -float(s["mean_duplicate_query_count"]),
            -step,  # exact ties prefer the earlier checkpoint
        )
        rows.append({"step": step, "health_gate": health, "rank": rank, "summary": s, "path": str(path)})

    if missing and not args.allow_partial:
        raise SystemExit(f"missing frozen-dev summaries for steps {missing}")
    if not rows:
        raise SystemExit("no checkpoint summaries found")
    best = max(rows, key=lambda r: r["rank"])
    out = {
        "selection_rule": {
            "health_gate": "finish>=0.95 AND parse_ok>=0.95 AND observation_mask_ok==1.0",
            "ranking": ["health_gate", "answer_token_f1", "evidence_f1", "EM", "fewer_duplicate_queries", "earlier_step_on_exact_tie"],
            "sealed_test_used": False,
        },
        "partial": bool(missing),
        "missing_steps": missing,
        "best_step": best["step"],
        "best_summary": best["summary"],
        "candidates": [
            {
                "step": r["step"],
                "health_gate": r["health_gate"],
                "mean_token_f1": r["summary"]["mean_token_f1"],
                "mean_evidence_f1": r["summary"]["mean_evidence_f1"],
                "mean_em": r["summary"]["mean_em"],
                "finish_rate": r["summary"]["finish_rate"],
                "parse_ok_rate": r["summary"]["parse_ok_rate"],
                "mean_search_count": r["summary"]["mean_search_count"],
                "mean_duplicate_query_count": r["summary"]["mean_duplicate_query_count"],
            }
            for r in rows
        ],
    }
    target = root / "results" / "checkpoint_selection.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"SELECTION_WRITTEN={target}")


if __name__ == "__main__":
    main()


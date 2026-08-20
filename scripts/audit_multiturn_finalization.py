#!/usr/bin/env python3
"""CPU-only: search=1 vs search=2 finish rates + A/B/C unfinished classes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    REPO
    / "results/51_heldout_test/n500_grpo400"
    / "agent_rollout_n500_20260820_104546_heldout_n500_grpo400/metrics.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-turn finalization audit.")
    p.add_argument("--config", type=str, default=str(REPO / "config" / "harness_v1.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "results" / "51_heldout_test" / "multiturn_harness_audit"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS))
    return p.parse_args()


def classify(row: Dict[str, Any]) -> str:
    steps = row.get("steps") or []
    types = [str(s.get("step_type") or "") for s in steps]
    n_obs = types.count("observation")
    gens = row.get("raw_generations") or []
    last = gens[-1] if gens else ""
    last_l = last.lower()
    gen_tok = float((row.get("cost_info") or {}).get("generated_tokens") or 0)
    if n_obs < 2:
        return "C_no_second_obs"
    if "<answer>" in last_l or "</answer>" in last_l:
        return "D_had_answer_tag_not_parsed"
    if len(last.strip()) < 8:
        return "A_stop_after_obs_empty_gen"
    if gen_tok >= 900 or len(last) >= 800 or "<evidence>" in last_l:
        return "B_budget_after_second_obs"
    return "A_stop_after_obs_no_answer"


def main() -> None:
    args = parse_args()
    path = Path(args.metrics)
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if args.max_samples and len(rows) >= args.max_samples:
                break

    cell = {
        "s1_fin": 0,
        "s1_unf": 0,
        "s2_fin": 0,
        "s2_unf": 0,
        "other": 0,
    }
    unfinished: List[Dict[str, Any]] = []
    for r in rows:
        sc = int((r.get("metrics") or {}).get("search_count") or 0)
        fin = bool(r.get("finished"))
        if sc == 1 and fin:
            cell["s1_fin"] += 1
        elif sc == 1 and not fin:
            cell["s1_unf"] += 1
        elif sc == 2 and fin:
            cell["s2_fin"] += 1
        elif sc == 2 and not fin:
            cell["s2_unf"] += 1
        else:
            cell["other"] += 1
        if (not fin) and sc == 2:
            kinds = [str(s.get("step_type") or "") for s in (r.get("steps") or [])]
            gens = r.get("raw_generations") or []
            last = gens[-1] if gens else ""
            unfinished.append(
                {
                    "sample_id": r.get("sample_id"),
                    "class": classify(r),
                    "queries": r.get("search_queries"),
                    "dup": float((r.get("metrics") or {}).get("duplicate_query_count") or 0),
                    "hit_max_search_turns": r.get("hit_max_search_turns"),
                    "generated_tokens": (r.get("cost_info") or {}).get("generated_tokens"),
                    "step_types": kinds,
                    "n_obs": kinds.count("observation"),
                    "last_gen_chars": len(last),
                    "last_gen_has_evidence": "<evidence>" in last.lower(),
                    "last_gen_has_search": "<search>" in last.lower(),
                    "last_gen_has_answer": "<answer>" in last.lower(),
                    "last_gen_preview": last[:240].replace("\n", " "),
                }
            )

    n1 = cell["s1_fin"] + cell["s1_unf"]
    n2 = cell["s2_fin"] + cell["s2_unf"]
    cls = Counter(u["class"] for u in unfinished)
    report = {
        "gate": "MULTITURN_HARNESS_AUDIT",
        "n": len(rows),
        "cell": cell,
        "p_unfinished_search1": round(cell["s1_unf"] / n1, 4) if n1 else None,
        "p_unfinished_search2": round(cell["s2_unf"] / n2, 4) if n2 else None,
        "class_counts": dict(cls),
        "verdict": (
            "MULTITURN_FINALIZATION_BUG"
            if n2 and cell["s2_unf"] / n2 >= 0.10 and cell["s1_unf"] / max(n1, 1) < 0.02
            else "NO_CLEAR_HARNESS_GAP"
        ),
        "code_note": (
            "After search_turns>=max, loop allows ONE generate then break. "
            "Stop strings are </search></answer></internal>, not </evidence>. "
            "Evidence can consume max_new_tokens=512 with no forced answer turn."
        ),
        "cases": unfinished,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "audit_summary.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"search=1  finish={cell['s1_fin']} unfinished={cell['s1_unf']}  "
        f"P(unf)={report['p_unfinished_search1']}"
    )
    print(
        f"search=2  finish={cell['s2_fin']} unfinished={cell['s2_unf']}  "
        f"P(unf)={report['p_unfinished_search2']}"
    )
    print("classes", dict(cls))
    print(report["verdict"])
    print("wrote", dest)


if __name__ == "__main__":
    main()

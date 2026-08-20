#!/usr/bin/env python3
"""CPU-only: did 4550 search_format teach question-copy as the <search> target?"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL | re.IGNORECASE)
_Q_RE = re.compile(r"^Question:\s*(.*)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT search_format query supervision audit.")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "harness_v1.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "32_audit_sft_query_supervision"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--canonical",
        type=str,
        default=str(REPO_ROOT / "results" / "17_build_8b_coldstart_v2" / "canonical.jsonl"),
    )
    p.add_argument("--n-examples", type=int, default=20)
    return p.parse_args()


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def question_of(row: Dict[str, Any]) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            body = str(m.get("content") or "")
            hit = _Q_RE.match(body.strip())
            if hit:
                return hit.group(1).split("\n\nDocuments:", 1)[0].strip()
            return body.strip()
    return str(row.get("question") or "").strip()


def first_search_query(target: str) -> Optional[str]:
    hit = _SEARCH_RE.search(target or "")
    if not hit:
        return None
    return hit.group(1).strip()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.canonical)
    if not src.is_file():
        raise SystemExit(f"CANONICAL_MISSING {src}")

    n_all = 0
    cats: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_all += 1
            cat = str(rec.get("category") or "")
            cats[cat] = cats.get(cat, 0) + 1
            if cat != "search_format":
                continue
            q = question_of(rec)
            query = first_search_query(str(rec.get("target") or ""))
            src_flag = ((rec.get("provenance") or {}).get("query_source"))
            rows.append(
                {
                    "sample_id": rec.get("sample_id"),
                    "sft_id": rec.get("sft_id"),
                    "question": q,
                    "query": query,
                    "query_source_field": src_flag,
                    "exact_copy": bool(query is not None and query == q),
                    "normalized_copy": bool(query is not None and norm(query) == norm(q)),
                    "length_ratio": (len(query) / max(len(q), 1)) if query else None,
                }
            )
            if args.max_samples and len(rows) >= args.max_samples:
                break

    n = len(rows)
    if n == 0:
        raise SystemExit("NO_SEARCH_FORMAT")
    exact_n = sum(1 for r in rows if r["exact_copy"])
    norm_n = sum(1 for r in rows if r["normalized_copy"])
    missing = sum(1 for r in rows if r["query"] is None)
    ratios = [float(r["length_ratio"]) for r in rows if r["length_ratio"] is not None]
    rng = random.Random(args.seed)
    examples = rng.sample(rows, k=min(args.n_examples, n))
    exact_rate = exact_n / n
    verdict = (
        "QUERY_COPY_IS_SUPERVISION_INDUCED"
        if exact_rate >= 0.90
        else "QUERY_COPY_NOT_DOMINANT"
    )
    summary = {
        "n_canonical": n_all,
        "category_counts": cats,
        "n_search_format": n,
        "n_search_missing_tag": missing,
        "exact_copy_rate": round(exact_rate, 4),
        "normalized_copy_rate": round(norm_n / n, 4),
        "non_copy_rate": round(1.0 - exact_rate, 4),
        "mean_query_question_length_ratio": round(statistics.mean(ratios), 4) if ratios else None,
        "query_source_field_counts": {
            str(k): sum(1 for r in rows if r["query_source_field"] == k)
            for k in {r["query_source_field"] for r in rows}
        },
        "builder_hardcodes_question_copy": True,
        "verdict": verdict,
        "canonical": str(src),
        "gate35_implication": (
            "stop_sampling_sweep; demote query_diversity from GRPO hard door"
            if verdict == "QUERY_COPY_IS_SUPERVISION_INDUCED"
            else "keep investigating query supervision"
        ),
    }
    (out / "query_supervision_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "query_supervision_examples.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(verdict, flush=True)
    for e in examples[:5]:
        print(
            f"COPY={e['exact_copy']} Q={e['question'][:80]!r} QUERY={str(e['query'])[:80]!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tool-only depth candidate mining for Web-MultiTurn-v2.

Gold annotations are used only after retrieval to assign hidden builder labels.
They are never sent to the Web provider or a teacher model.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import load_jsonl
from src.tools.web_adapter import WebAdapter

DEFAULT_POOL = Path(
    "/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)
DEFAULT_OUT = ROOT / "results/56_web_multiturn_v2/depth_mining"
_TITLE_CLEAN = re.compile(r"[^a-z0-9]+")
_LEAK = re.compile(
    r"hotpot.?qa|huggingface\.co/datasets|datasets-server|kaggle\.com/datasets", re.I
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--web-timeout", type=float, default=30.0)
    p.add_argument("--web-retries", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="Mine only 20 examples.")
    return p.parse_args()


def norm(value: str) -> str:
    return normalize_answer(str(value))


def title_norm(value: str) -> str:
    return _TITLE_CLEAN.sub(" ", str(value).lower()).strip()


def answer_visible(text: str, answers: list[str]) -> bool:
    hay = norm(text)
    padded = f" {hay} "
    return any(norm(x) and f" {norm(x)} " in padded for x in answers)


def supporting_titles(sample: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(x.get("title") or "").strip()
            for x in sample.get("supporting_facts") or []
            if str(x.get("title") or "").strip()
        }
    )


def title_hits(text: str, titles: list[str]) -> list[str]:
    hay = title_norm(text)
    return [x for x in titles if title_norm(x) and title_norm(x) in hay]


def classify_candidate(sample: dict[str, Any], docs: list[dict[str, Any]]) -> tuple[int, str]:
    text = "\n".join(f"{x.get('title') or ''}\n{x.get('text') or ''}" for x in docs)
    answers = [str(x) for x in sample.get("gold_answers") or []]
    titles = supporting_titles(sample)
    hits = title_hits(text, titles)
    if answer_visible(text, answers):
        return 1, "answer_visible_after_search1"
    if titles and hits:
        return 2, "support_visible_answer_missing"
    return 0, "unresolved_or_no_bridge"


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = parse_args()
    if cfg.smoke:
        cfg.max_samples = min(cfg.max_samples, 20)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(str(cfg.pool))
    excluded: set[str] = set()
    for name in ("web_dev50_ids.txt", "web_final50_ids.txt"):
        path = ROOT / "data/web_v2" / name
        if path.exists():
            excluded.update(x.strip() for x in path.read_text().splitlines() if x.strip())
    rows = [x for x in pool if str(x["sample_id"]) not in excluded]
    random.Random(cfg.seed).shuffle(rows)
    rows = rows[: cfg.max_samples]
    web = WebAdapter(
        provider="brave_llm_context",
        cache_dir=cfg.output_dir / "web_cache",
        timeout_s=cfg.web_timeout,
        retries=cfg.web_retries,
        llm_context_tokens=4096,
    )
    mined: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for index, sample in enumerate(rows, 1):
        packed = web.retrieve(sample, str(sample["question"]), cfg.top_k)
        docs = list(packed.get("documents") or [])
        errors = list(packed.get("errors") or [])
        urls = [str((x.get("metadata") or {}).get("url") or "") for x in docs]
        text = "\n".join(
            f"{x.get('title') or ''}\n{x.get('text') or ''}" for x in docs
        )
        if errors:
            label, reason = 0, "web_error"
        elif not docs:
            label, reason = 0, "empty_observation"
        elif any(_LEAK.search(x) for x in urls):
            label, reason = 0, "benchmark_leak"
        else:
            label, reason = classify_candidate(sample, docs)
        reasons[reason] = reasons.get(reason, 0) + 1
        mined.append(
            {
                "sample_id": sample["sample_id"],
                "likely_depth": label,
                "mining_reason": reason,
                "search_query": sample["question"],
                "answer_visible_after_search1": reason == "answer_visible_after_search1",
                "supporting_title_count": len(supporting_titles(sample)),
                "supporting_title_hits": title_hits(text, supporting_titles(sample)),
                "document_count": len(docs),
                "search1_documents": docs,
                "gold_used_builder_side_only": True,
            }
        )
        print(f"[{index}/{len(rows)}] likely_depth={label} {sample['sample_id']}", flush=True)
    usable = [x for x in mined if x["likely_depth"] in (1, 2)]
    dump_jsonl(cfg.output_dir / "all_mined.jsonl", mined)
    dump_jsonl(cfg.output_dir / "candidate_manifest.jsonl", usable)
    dump_jsonl(cfg.output_dir / "depth1_candidates.jsonl", [x for x in usable if x["likely_depth"] == 1])
    dump_jsonl(cfg.output_dir / "depth2_candidates.jsonl", [x for x in usable if x["likely_depth"] == 2])
    counts = {str(d): sum(x["likely_depth"] == d for x in mined) for d in (0, 1, 2)}
    summary = {
        "gate": "W3_DEPTH_MINING_PASS" if counts["1"] and counts["2"] else "W3_DEPTH_MINING_INCOMPLETE",
        "attempted": len(mined),
        "likely_depth_counts": counts,
        "yield_rate": len(usable) / max(1, len(mined)),
        "reasons": reasons,
        "gold_visible_to_teacher_or_web": False,
        "provider": "brave_llm_context",
        "candidate_manifest": str(cfg.output_dir / "candidate_manifest.jsonl"),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["gate"] != "W3_DEPTH_MINING_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

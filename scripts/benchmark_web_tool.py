#!/usr/bin/env python3
"""Tool-only comparison across normalized WebAdapter providers."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.web_adapter import WebAdapter  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def normalized(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return [float(row.get(key) or 0.0) for row in rows]

    tool = values("tool_total_ms")
    return {
        "n": len(rows),
        "mean_tool_ms": round(statistics.fmean(tool), 2) if tool else 0.0,
        "p50_tool_ms": round(percentile(tool, 0.50), 2),
        "p95_tool_ms": round(percentile(tool, 0.95), 2),
        "mean_search_api_ms": round(statistics.fmean(values("search_api_ms")), 2),
        "mean_fetch_ms": round(statistics.fmean(values("fetch_total_ms")), 2),
        "mean_extract_ms": round(statistics.fmean(values("extract_ms")), 2),
        "mean_failed_urls": round(statistics.fmean(values("failed_urls")), 3),
        "mean_filtered_urls": round(statistics.fmean(values("filtered_urls")), 3),
        "nonempty_context_rate": round(
            sum(bool(row["document_count"]) for row in rows) / max(1, len(rows)), 4
        ),
        "mean_context_tokens_approx": round(
            statistics.fmean(values("context_tokens_approx")), 2
        ),
        "gold_answer_string_hit_rate": round(
            sum(bool(row["gold_answer_string_hit"]) for row in rows) / max(1, len(rows)), 4
        ),
        "mean_supporting_title_recall": round(
            statistics.fmean(values("supporting_title_recall")), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval/hotpotqa_200.jsonl")
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--raw-timeout", type=float, default=12.0)
    parser.add_argument("--raw-retries", type=int, default=0)
    parser.add_argument("--context-timeout", type=float, default=30.0)
    parser.add_argument("--context-retries", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("brave", "brave_llm_context", "bocha"),
        default=("brave", "brave_llm_context"),
    )
    parser.add_argument("--output", default="results/55_web_infra/tool_ab.json")
    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.is_absolute():
        eval_path = ROOT / eval_path
    samples = load_jsonl(eval_path)[: args.max_queries]
    available = {
        "brave": lambda: WebAdapter(
            provider="brave",
            cache_dir=ROOT / "results/55_web_infra/cache_local",
            timeout_s=args.raw_timeout,
            retries=args.raw_retries,
        ),
        "brave_llm_context": lambda: WebAdapter(
            provider="brave_llm_context",
            cache_dir=ROOT / "results/55_web_infra/cache_context",
            timeout_s=args.context_timeout,
            retries=args.context_retries,
            llm_context_tokens=args.context_tokens,
        ),
        "bocha": lambda: WebAdapter(
            provider="bocha",
            cache_dir=ROOT / "results/55_web_infra/cache_bocha",
            timeout_s=args.context_timeout,
            retries=args.context_retries,
            llm_context_tokens=args.context_tokens,
        ),
    }
    adapters = {name: available[name]() for name in args.providers}
    details: dict[str, list[dict[str, Any]]] = {name: [] for name in adapters}
    for index, sample in enumerate(samples, 1):
        query = sample["question"]
        gold = [normalized(x) for x in sample.get("gold_answers") or []]
        gold_titles = {normalized(x.get("title")) for x in sample.get("supporting_facts") or []}
        for name, adapter in adapters.items():
            call_started = time.monotonic()
            result = adapter.retrieve({}, query, args.top_k)
            timing = result.get("timing") or {}
            docs = result.get("documents") or []
            context = normalized("\n".join(str(x.get("text") or "") for x in docs))
            titles = {normalized(x.get("title")) for x in docs}
            title_hits = sum(
                any(gold_title in title or title in gold_title for title in titles if title)
                for gold_title in gold_titles
            )
            row = {
                "sample_id": sample.get("sample_id"),
                "query": query,
                "document_count": len(docs),
                "document_urls": [
                    str((document.get("metadata") or {}).get("url") or "") for document in docs
                ],
                "error_count": len(result.get("errors") or []),
                "errors": result.get("errors") or [],
                "context_tokens_approx": len(context.split()),
                "gold_answer_string_hit": any(answer and answer in context for answer in gold),
                "supporting_title_recall": title_hits / max(1, len(gold_titles)),
                **timing,
            }
            details[name].append(row)
            print(
                f"[{index}/{len(samples)}] {name} docs={len(docs)} "
                f"tool_ms={timing.get('tool_total_ms', 0)} errors={len(row['errors'])}",
                flush=True,
            )
            # Keep provider calls serialized and avoid burst-rate artifacts.
            remaining = 1.05 - (time.monotonic() - call_started)
            if remaining > 0:
                time.sleep(remaining)

    output = {
        "purpose": "web_tool_only_ab",
        "eval_file": str(eval_path),
        "top_k": args.top_k,
        "context_tokens": args.context_tokens,
        "summary": {name: summarize(rows) for name, rows in details.items()},
        "details": details,
    }
    target = Path(args.output)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"WEB_TOOL_AB_PASS output={target}")


if __name__ == "__main__":
    main()

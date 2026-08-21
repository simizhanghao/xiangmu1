#!/usr/bin/env python3
"""Tool-only depth candidate mining for Web-MultiTurn-v2.

Gold annotations are used only after retrieval to assign hidden builder labels.
They are never sent to the Web provider or a teacher model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
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
_LEAK_FILTER_VERSION = "webmt_v2_leak_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--web-timeout", type=float, default=30.0)
    p.add_argument("--web-retries", type=int, default=2)
    p.add_argument(
        "--min-request-interval",
        type=float,
        default=0.0,
        help="Minimum seconds between provider request starts (rate-limit guard).",
    )
    p.add_argument(
        "--provider", choices=("brave_llm_context", "bocha"), default="brave_llm_context"
    )
    p.add_argument(
        "--candidate-policy",
        choices=("support_bridge", "answer_absent"),
        default="support_bridge",
        help=(
            "support_bridge requires benchmark supporting titles; answer_absent marks a "
            "nonempty Search1 without the answer as a D2 candidate for strict downstream validation."
        ),
    )
    p.add_argument(
        "--max-consecutive-web-errors",
        type=int,
        default=5,
        help="Abort and preserve a partial artifact when the provider is unhealthy.",
    )
    p.add_argument("--smoke", action="store_true", help="Mine only 20 examples.")
    p.add_argument("--exclude-file", type=Path, action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
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


def classify_candidate(
    sample: dict[str, Any], docs: list[dict[str, Any]], candidate_policy: str = "support_bridge"
) -> tuple[int, str]:
    text = "\n".join(f"{x.get('title') or ''}\n{x.get('text') or ''}" for x in docs)
    answers = [str(x) for x in sample.get("gold_answers") or []]
    titles = supporting_titles(sample)
    hits = title_hits(text, titles)
    if answer_visible(text, answers):
        return 1, "answer_visible_after_search1"
    if candidate_policy == "answer_absent":
        return 2, "answer_absent_after_search1"
    if titles and hits:
        return 2, "support_visible_answer_missing"
    return 0, "unresolved_or_no_bridge"


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ids_from_path(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".txt":
        return {x.strip() for x in path.read_text().splitlines() if x.strip()}
    ids: set[str] = set()
    for row in load_jsonl(str(path)):
        value = row.get("sample_id") or row.get("question_id")
        if value:
            ids.add(str(value))
    return ids


def frozen_exclusions(extra: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    paths: list[Path] = []
    paths.extend(sorted((ROOT / "data/web_v2").glob("*_ids.txt")))
    paths.extend(sorted((ROOT / "data/eval").glob("*.jsonl")))
    paths.extend(sorted((ROOT / "data/sealed").glob("*.jsonl")))
    paths.extend(sorted((ROOT / "results/54_web_zero_shot").glob("**/*n40*/trace.jsonl")))
    # Development-smoke questions were used to tune the builder gates and are not
    # eligible for the formal mining pool.
    paths.extend(sorted((ROOT / "results/56_web_multiturn_v2/depth_mining_smoke").glob("all_mined.jsonl")))
    # The N=400 mixed Pilot froze the protocol and measured strict yield. Keep
    # every question seen in that phase out of the fresh formal-scale corpus.
    paths.extend(sorted((ROOT / "results/56_web_multiturn_v2/depth_mining_n400").glob("all_mined.jsonl")))
    paths.extend(sorted((ROOT / "results/56_web_multiturn_v2/mixed_pilot_n120").glob("trajectories.jsonl")))
    paths.extend(extra)
    excluded: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        found = ids_from_path(path)
        excluded.update(found)
        sources.append({"path": str(path), "ids": len(found)})
    return excluded, sources


def main() -> None:
    cfg = parse_args()
    if cfg.smoke:
        cfg.max_samples = min(cfg.max_samples, 20)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(str(cfg.pool))
    excluded, exclusion_sources = frozen_exclusions(cfg.exclude_file)
    rows = [x for x in pool if str(x["sample_id"]) not in excluded]
    random.Random(cfg.seed).shuffle(rows)
    rows = rows[: cfg.max_samples]
    exclusion_manifest = {
        "excluded_id_count": len(excluded),
        "sources": exclusion_sources,
        "eligible_pool_count": len(pool) - sum(str(x["sample_id"]) in excluded for x in pool),
        "selected_count": len(rows),
        "selected_overlap_with_exclusions": sum(str(x["sample_id"]) in excluded for x in rows),
    }
    (cfg.output_dir / "exclusion_manifest.json").write_text(
        json.dumps(exclusion_manifest, indent=2), encoding="utf-8"
    )
    if cfg.dry_run:
        print(json.dumps({"gate": "W3_MINING_PREFLIGHT_PASS", **exclusion_manifest}, indent=2))
        return
    web = WebAdapter(
        provider=cfg.provider,
        cache_dir=cfg.output_dir / "web_cache",
        timeout_s=cfg.web_timeout,
        retries=cfg.web_retries,
        llm_context_tokens=4096,
    )
    mined: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    consecutive_web_errors = 0
    run_timestamp = datetime.now(timezone.utc).isoformat()
    for index, sample in enumerate(rows, 1):
        request_started = time.monotonic()
        packed = web.retrieve(sample, str(sample["question"]), cfg.top_k)
        remaining = cfg.min_request_interval - (time.monotonic() - request_started)
        if remaining > 0:
            time.sleep(remaining)
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
            label, reason = classify_candidate(sample, docs, cfg.candidate_policy)
        reasons[reason] = reasons.get(reason, 0) + 1
        consecutive_web_errors = consecutive_web_errors + 1 if reason == "web_error" else 0
        frozen_docs = json.dumps(docs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
                "retrieval_errors": errors,
                "search1_documents": docs,
                "search1_provenance": {
                    "question_id": sample["sample_id"],
                    "query1": sample["question"],
                    "provider": cfg.provider,
                    "top_k": cfg.top_k,
                    "context_tokens": 4096,
                    "leak_filter_version": _LEAK_FILTER_VERSION,
                    "obs1_sha256": hashlib.sha256(frozen_docs.encode("utf-8")).hexdigest(),
                    "source_ids": [str(x.get("document_id") or "") for x in docs],
                    "urls": [str((x.get("metadata") or {}).get("url") or "") for x in docs],
                    "timestamp_utc": run_timestamp,
                },
                "gold_used_builder_side_only": True,
            }
        )
        print(
            f"[{index}/{len(rows)}] likely_depth={label} reason={reason} "
            f"{sample['sample_id']}",
            flush=True,
        )
        if consecutive_web_errors >= cfg.max_consecutive_web_errors:
            dump_jsonl(cfg.output_dir / "all_mined.partial.jsonl", mined)
            failure = {
                "gate": "W3_PROVIDER_UNHEALTHY",
                "attempted": len(mined),
                "consecutive_web_errors": consecutive_web_errors,
                "reasons": reasons,
                "last_errors": errors,
                "partial_artifact": str(cfg.output_dir / "all_mined.partial.jsonl"),
            }
            (cfg.output_dir / "failure_summary.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            print(json.dumps(failure, indent=2), flush=True)
            raise SystemExit(2)
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
        "provider": cfg.provider,
        "candidate_manifest": str(cfg.output_dir / "candidate_manifest.jsonl"),
        "excluded_id_count": len(excluded),
        "exclusion_sources": exclusion_sources,
        "search1_provenance_frozen": True,
        "candidate_policy": cfg.candidate_policy,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["gate"] != "W3_DEPTH_MINING_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create Bocha-native D2 candidates from two independently retrieved sources.

The synthesizer first discovers a bridge in Source A, retrieves Source B using
that bridge, and only then writes a question whose bridge and answer are hidden.
All resulting questions must still pass the ordinary frozen Search1/Builder
counterfactual pipeline; synthesis provenance is never runtime model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import load_jsonl
from src.tools.web_adapter import WebAdapter

DEFAULT_POOL = Path(
    "/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--target", type=int, default=12)
    p.add_argument("--max-seeds", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--teacher-base-url", default=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-flash"))
    p.add_argument("--teacher-api-key", default=os.environ.get("TEACHER_API_KEY", ""))
    return p.parse_args()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_docs(docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "source_id": f"D{i}",
            "title": str(x.get("title") or ""),
            "text": str(x.get("text") or "")[:1800],
            "url": str((x.get("metadata") or {}).get("url") or ""),
        }
        for i, x in enumerate(docs, 1)
    ]


def llm_json(cfg: argparse.Namespace, system: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.teacher_api_key}"}
    body: dict[str, Any] = {
        "model": cfg.teacher_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": cfg.seed,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                cfg.teacher_base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json=body,
                timeout=180,
            )
            if response.status_code >= 400 and attempt == 0:
                body.pop("thinking", None)
                body.pop("response_format", None)
                continue
            response.raise_for_status()
            content = str(response.json()["choices"][0]["message"]["content"]).strip()
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("not a JSON object")
            return value
        except Exception as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"teacher failure: {type(error).__name__}")


def visible(value: str, text: str) -> bool:
    needle = normalize_answer(value)
    return bool(needle) and f" {needle} " in f" {normalize_answer(text)} "


def main() -> None:
    cfg = parse_args()
    if not cfg.teacher_api_key:
        raise SystemExit("TEACHER_API_KEY is required")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = cfg.output_dir / "synthesis_provenance.jsonl"
    rejected_path = cfg.output_dir / "rejected.jsonl"
    pool_path = cfg.output_dir / "builder_pool.jsonl"
    candidate_path = cfg.output_dir / "candidate_manifest.jsonl"
    for path in (provenance_path, rejected_path, pool_path, candidate_path):
        path.unlink(missing_ok=True)

    seeds = load_jsonl(str(cfg.seed_pool))
    random.Random(cfg.seed).shuffle(seeds)
    seeds = seeds[: cfg.max_seeds]
    web = WebAdapter(
        provider="bocha",
        cache_dir=cfg.output_dir / "web_cache",
        timeout_s=30,
        retries=2,
        llm_context_tokens=4096,
    )
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    bridge_system = (
        "You design two-hop Web research tasks. Using Source A only, select one concrete "
        "bridge entity explicitly present in a cited source and one factual attribute that "
        "is not answered by Source A. Return JSON: bridge_entity, source_a_id, bridge_fact, "
        "missing_relation, search2_query. search2_query must contain the bridge entity."
    )
    question_system = (
        "You design shortcut-resistant two-hop questions from supplied Web sources. Return "
        "JSON: answer, source_b_id, answer_fact, generated_question. The short answer and "
        "answer_fact must be explicit in Source B. The question must require resolving the "
        "Source-A bridge and then its Source-B attribute, but must not mention either the "
        "bridge entity or answer. Do not use facts outside the supplied sources."
    )
    for index, seed_row in enumerate(seeds, 1):
        seed_query = str(seed_row["question"])
        try:
            packed_a = web.retrieve({"sample_id": f"seed_{index}"}, seed_query, cfg.top_k)
            docs_a = list(packed_a.get("documents") or [])
            if packed_a.get("errors") or not docs_a:
                raise ValueError("source_a_web_error")
            compact_a = compact_docs(docs_a)
            bridge = llm_json(cfg, bridge_system, {"discovery_query": seed_query, "source_a": compact_a})
            bridge_entity = " ".join(str(bridge.get("bridge_entity") or "").split())
            source_a_id = str(bridge.get("source_a_id") or "")
            query2 = " ".join(str(bridge.get("search2_query") or "").split())
            source_a = next((x for x in compact_a if x["source_id"] == source_a_id), None)
            if not bridge_entity or not query2 or source_a is None or not visible(bridge_entity, source_a["text"]):
                raise ValueError("invalid_grounded_bridge")
            if not visible(bridge_entity, query2):
                raise ValueError("query2_missing_bridge")
            packed_b = web.retrieve({"sample_id": f"seed_{index}_b"}, query2, cfg.top_k)
            docs_b = list(packed_b.get("documents") or [])
            if packed_b.get("errors") or not docs_b:
                raise ValueError("source_b_web_error")
            compact_b = compact_docs(docs_b)
            final = llm_json(
                cfg,
                question_system,
                {"source_a": [source_a], "bridge": bridge, "source_b": compact_b},
            )
            answer = " ".join(str(final.get("answer") or "").split())
            question = " ".join(str(final.get("generated_question") or "").split())
            source_b_id = str(final.get("source_b_id") or "")
            source_b = next((x for x in compact_b if x["source_id"] == source_b_id), None)
            if not answer or not question or source_b is None or not visible(answer, source_b["text"]):
                raise ValueError("invalid_grounded_answer")
            if visible(answer, source_a["text"]):
                raise ValueError("answer_visible_in_source_a")
            if visible(answer, question) or visible(bridge_entity, question):
                raise ValueError("question_leaks_bridge_or_answer")
            if source_a["url"] == source_b["url"]:
                raise ValueError("same_source_url")

            # Revalidate from the generated question, not from the discovery
            # query. A genuine D2 candidate must recover the bridge but not the
            # answer in this exact frozen Search1 state.
            fresh = web.retrieve({"sample_id": f"seed_{index}_fresh"}, question, cfg.top_k)
            fresh_docs = list(fresh.get("documents") or [])
            if fresh.get("errors") or not fresh_docs:
                raise ValueError("fresh_search1_web_error")
            fresh_text = "\n".join(
                f"{x.get('title') or ''}\n{x.get('text') or ''}" for x in fresh_docs
            )
            if not visible(bridge_entity, fresh_text):
                raise ValueError("fresh_search1_bridge_missing")
            if visible(answer, fresh_text):
                raise ValueError("fresh_search1_answer_visible")
            digest = hashlib.sha256(
                f"{cfg.seed}:{question}:{answer}:{source_a['url']}:{source_b['url']}".encode()
            ).hexdigest()[:20]
            sample_id = f"sourcefirst_{digest}"
            runtime_row = {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": [answer],
                "supporting_facts": [],
                "gold_builder_side_only": True,
            }
            provenance = {
                "sample_id": sample_id,
                "seed_sample_id": seed_row.get("sample_id"),
                "seed_query": seed_query,
                "bridge": bridge,
                "answer": answer,
                "question": question,
                "source_a": source_a,
                "source_b": source_b,
                "provider": "bocha",
                "teacher_model": cfg.teacher_model,
                "requires_fresh_causal_revalidation": True,
            }
            append_jsonl(pool_path, runtime_row)
            append_jsonl(provenance_path, provenance)
            frozen_docs = json.dumps(
                fresh_docs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            append_jsonl(
                candidate_path,
                {
                    "sample_id": sample_id,
                    "likely_depth": 2,
                    "mining_reason": "fresh_search1_bridge_visible_answer_absent",
                    "search_query": question,
                    "answer_visible_after_search1": False,
                    "supporting_title_count": 0,
                    "supporting_title_hits": [],
                    "document_count": len(fresh_docs),
                    "retrieval_errors": [],
                    "search1_documents": fresh_docs,
                    "search1_provenance": {
                        "question_id": sample_id,
                        "query1": question,
                        "provider": "bocha",
                        "top_k": cfg.top_k,
                        "context_tokens": 4096,
                        "leak_filter_version": "webmt_v2_leak_v1",
                        "obs1_sha256": hashlib.sha256(frozen_docs.encode()).hexdigest(),
                        "source_ids": [str(x.get("document_id") or "") for x in fresh_docs],
                        "urls": [str((x.get("metadata") or {}).get("url") or "") for x in fresh_docs],
                    },
                    "gold_used_builder_side_only": True,
                    "bridge_visible_builder_side": True,
                },
            )
            accepted.append(runtime_row)
            print(f"[{len(accepted)}/{cfg.target}] {sample_id}", flush=True)
            if len(accepted) >= cfg.target:
                break
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ValueError) else f"exception_{type(exc).__name__}"
            reasons[reason] = reasons.get(reason, 0) + 1
            append_jsonl(rejected_path, {"seed_sample_id": seed_row.get("sample_id"), "reason": reason})

    summary = {
        "gate": "SOURCE_FIRST_SYNTHESIS_PASS" if len(accepted) == cfg.target else "SOURCE_FIRST_SYNTHESIS_INCOMPLETE",
        "target": cfg.target,
        "accepted": len(accepted),
        "attempted": min(len(seeds), len(accepted) + sum(reasons.values())),
        "rejection_reasons": reasons,
        "provider": "bocha",
        "teacher_model": cfg.teacher_model,
        "runtime_contains_provenance": False,
        "requires_fresh_causal_revalidation": True,
        "fresh_search1_bridge_visible": True,
        "fresh_search1_answer_visible": False,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(accepted) != cfg.target:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

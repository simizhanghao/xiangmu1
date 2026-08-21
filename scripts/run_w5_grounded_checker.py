#!/usr/bin/env python3
"""Offline grounded STOP/CONTINUE checker for frozen W5 Web states.

Gold fields are available only to this labeler. They are never copied into the
Controller input and this script does not generate the next query.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


SYSTEM = """You are an offline grounded-sufficiency labeler. Given a question,
reference answers, reference supporting-fact titles, and retrieved Web sources, decide
whether the CURRENT sources are sufficient to answer the question correctly.
Return one JSON object only:
{"decision":"STOP|CONTINUE","grounded_answer":"short answer or empty",
 "source_ids":["D1"],"missing":"specific unresolved fact or empty","rationale":"short"}.
STOP only when the short reference answer is directly supported by the supplied sources
and the necessary multi-hop relation is grounded. Otherwise CONTINUE. Never invent a
source, never propose a search query, and do not use outside knowledge."""


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-dir", type=Path, default=Path("results/72_w5_state_dataset/checker_candidates"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--per-stratum", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com"))
    p.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-flash"))
    p.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY", ""))
    return p.parse_args()


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def compact(row: dict[str, Any]) -> dict[str, Any]:
    docs = []
    for i, doc in enumerate(row.get("documents") or [], 1):
        docs.append({"source_id": f"D{i}", "title": doc.get("title", ""), "text": str(doc.get("text") or "")[:2200]})
    audit = row.get("builder_audit") or {}
    return {
        "question": row["question"],
        "reference_answers": audit.get("gold_answers") or row.get("gold_answers") or [],
        "reference_supporting_titles": audit.get("supporting_titles") or [],
        "sources": docs,
    }


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    value = json.loads(text)
    if value.get("decision") not in {"STOP", "CONTINUE"}:
        raise ValueError("invalid decision")
    ids = value.get("source_ids") or []
    if not isinstance(ids, list):
        raise ValueError("source_ids is not a list")
    return value


def call(cfg: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
    body = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(compact(row), ensure_ascii=False)}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
    }
    error: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            response = requests.post(cfg.base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, timeout=cfg.timeout)
            if response.status_code >= 400 and attempt == 0:
                body.pop("thinking", None)
                body.pop("response_format", None)
                continue
            response.raise_for_status()
            value = parse_json(str(response.json()["choices"][0]["message"]["content"]))
            n_docs = len(row.get("documents") or [])
            if any(not re.fullmatch(r"D[1-9][0-9]*", str(x)) or int(str(x)[1:]) > n_docs for x in value["source_ids"]):
                raise ValueError("invalid source id")
            if value["decision"] == "STOP" and (not value.get("grounded_answer") or not value["source_ids"]):
                raise ValueError("ungrounded STOP")
            return {**value, "ok": True}
        except Exception as exc:
            error = exc
            time.sleep(2**attempt)
    return {"ok": False, "error_type": type(error).__name__, "error": str(error)[:300]}


def main() -> None:
    cfg = args()
    if not cfg.api_key:
        raise SystemExit("MISSING_TEACHER_API_KEY")
    natural = load(cfg.candidate_dir / "natural_states.jsonl")
    masked = load(cfg.candidate_dir / "masked_siblings.jsonl")
    if cfg.per_stratum:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in natural + masked:
            groups[row["label_candidate"]].append(row)
        rows = [row for key in sorted(groups) for row in groups[key][: cfg.per_stratum]]
    else:
        rows = natural + masked
        if cfg.max_samples:
            rows = rows[: cfg.max_samples]
    if cfg.max_samples:
        rows = rows[: cfg.max_samples]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "labels.jsonl"
    cached = {x["state_id"]: x for x in load(out_path)} if out_path.exists() else {}
    todo = [row for row in rows if row["state_id"] not in cached]
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futures = {pool.submit(call, cfg, row): row for row in todo}
        for future in as_completed(futures):
            row = futures[future]
            result = {"state_id": row["state_id"], "sample_id": row["sample_id"], "controller_split": row.get("controller_split"), "state_origin": row["state_origin"], "label_candidate": row["label_candidate"], "teacher": future.result()}
            cached[row["state_id"]] = result
            with out_path.open("a") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    selected = [cached[row["state_id"]] for row in rows]
    ok = [x for x in selected if x["teacher"].get("ok")]
    decisions = Counter(x["teacher"].get("decision") for x in ok)
    masked_ok = [x for x in ok if x["state_origin"] == "counterfactual_evidence_mask"]
    summary = {
        "gate": ("W5_GROUNDED_CHECKER_FULL_PASS" if len(rows) > 100 else "W5_GROUNDED_CHECKER_SMOKE_PASS") if len(ok) == len(rows) and decisions["STOP"] > 0 and decisions["CONTINUE"] > 0 and (not masked_ok or sum(x["teacher"]["decision"] == "CONTINUE" for x in masked_ok) / len(masked_ok) >= 0.8) else ("W5_GROUNDED_CHECKER_FULL_FAIL" if len(rows) > 100 else "W5_GROUNDED_CHECKER_SMOKE_FAIL"),
        "requested": len(rows), "successful": len(ok), "failed": len(rows) - len(ok),
        "decisions": dict(decisions), "masked_states": len(masked_ok),
        "masked_continue_rate": (sum(x["teacher"]["decision"] == "CONTINUE" for x in masked_ok) / len(masked_ok)) if masked_ok else None,
        "teacher_model": cfg.model, "gold_visible_only_to_offline_checker": True,
        "query_generation_in_this_stage": False, "labels": str(out_path),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["gate"].endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Hard data gate for Web-MultiTurn-v2 trajectories."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.research_memory import ResearchMemory, serialize_research_memory

_Q = re.compile(r"[^a-z0-9]+")
_W = re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
_LEAK = re.compile(r"hotpot.?qa|huggingface\.co/datasets|datasets-server|kaggle\.com/datasets", re.I)
_STOP = {"the", "and", "for", "was", "what", "which", "who", "where", "with", "from", "that", "this", "about"}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def qnorm(x: str) -> str:
    return _Q.sub(" ", x.lower()).strip()


def words(x: str) -> set[str]:
    return {w for w in _W.findall(x.lower()) if w not in _STOP}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--allow-incomplete", action="store_true")
    a = p.parse_args()
    rows = load(a.run_dir / "trajectories.jsonl")
    share = load(a.run_dir / "sharegpt.jsonl")
    parity = 0
    extra = duplicate = obs_conditioned = new_source = new_evidence = 0
    empty = leaks = invalid_evidence_refs = 0
    depth: dict[int, int] = {}
    for row in rows:
        memory = ResearchMemory(row["question"], max_searches=row["target_depth"], evidence_limit=8, char_budget=4500)
        queries: list[str] = []
        prior_obs = ""
        for turn in row["turns"]:
            memory.update_from_internal(turn["internal"])
            query = turn["query"]
            if queries:
                extra += 1
                duplicate += int(qnorm(query) in {qnorm(x) for x in queries})
                novel = words(query) - words(row["question"])
                obs_conditioned += int(bool(novel & words(prior_obs)))
                new_source += int(int(turn["new_urls"]) > 0)
                new_evidence += int(int(turn["new_evidence"]) > 0)
            queries.append(query)
            docs = turn["documents"]
            empty += int(not docs)
            leaks += sum(bool(_LEAK.search(str((x.get("metadata") or {}).get("url") or ""))) for x in docs)
            memory.add_search(query, docs)
            if serialize_research_memory(memory) != turn["memory_rendered"]:
                raise SystemExit(f"MEMORY_PARITY_FAIL {row['sample_id']} turn={turn['search_turn']}")
            parity += 1
            prior_obs = "\n".join(str(x.get("text") or "") for x in docs)
        available_doc_ids = {
            str(doc.get("document_id") or "")
            for turn in row["turns"]
            for doc in turn["documents"]
        }
        cited_doc_ids = set(re.findall(r"\[document_id=([^|\]]+)", row["actions"][-1]))
        invalid_evidence_refs += len({x.strip() for x in cited_doc_ids} - available_doc_ids)
        depth[row["actual_depth"]] = depth.get(row["actual_depth"], 0) + 1
    forbidden = sum(
        any(k in example for k in ("gold_answers", "supporting_facts", "contexts"))
        for example in share
    )
    summary_file = json.loads((a.run_dir / "summary.json").read_text())
    metrics = {
        "gate": "WEB_MEMORY_PROTOCOL_PARITY_PASS",
        "trajectories": len(rows),
        "decision_examples": len(share),
        "depth_counts": {str(k): v for k, v in sorted(depth.items())},
        "replayed_memory_states": parity,
        "duplicate_extra_query_rate": duplicate / max(1, extra),
        "obs_conditioned_extra_query_rate": obs_conditioned / max(1, extra),
        "new_source_extra_query_rate": new_source / max(1, extra),
        "new_evidence_extra_query_rate": new_evidence / max(1, extra),
        "empty_observations": empty,
        "retained_leak_urls": leaks,
        "invalid_evidence_document_refs": invalid_evidence_refs,
        "training_examples_with_oracle_fields": forbidden,
    }
    quality_pass = (
        rows
        and metrics["duplicate_extra_query_rate"] < 0.05
        and (extra == 0 or metrics["obs_conditioned_extra_query_rate"] > 0.80)
        and (extra == 0 or metrics["new_source_extra_query_rate"] > 0.80)
        and empty == leaks == forbidden == invalid_evidence_refs == 0
    )
    complete = summary_file.get("gate") == "W3_PILOT_BUILT"
    hard_pass = quality_pass and (a.allow_incomplete or complete)
    if quality_pass and complete:
        metrics["gate"] = "W3_DATA_GATE_PASS"
    elif quality_pass and a.allow_incomplete:
        metrics["gate"] = "W3_ACCEPTED_SUBSET_GATE_PASS"
    else:
        metrics["gate"] = "W3_DATA_GATE_FAIL"
    (a.run_dir / "audit_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not hard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

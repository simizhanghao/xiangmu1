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
    decision_path = a.run_dir / "decision_sft.jsonl"
    share = load(decision_path if decision_path.exists() else a.run_dir / "sharegpt.jsonl")
    balanced_path = a.run_dir / "decision_sft_balanced.jsonl"
    balanced = load(balanced_path) if balanced_path.exists() else []
    parity = 0
    extra = duplicate = obs_conditioned = new_source = new_evidence = 0
    empty = leaks = invalid_evidence_refs = 0
    depth: dict[int, int] = {}
    memory_token_values: list[int] = []
    d1_stop_correct = d2_forced1_insufficient = d2_positive_delta = 0
    d1_total = d2_total = causal_records = 0
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
                conditioning = str(turn.get("query_conditioning") or "")
                if conditioning:
                    obs_conditioned += int(conditioning in {"observation_entity", "missing_state_refinement"})
                else:
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
            if "memory_tokens" in turn:
                memory_token_values.append(int(turn["memory_tokens"]))
            prior_obs = "\n".join(str(x.get("text") or "") for x in docs)
        available_doc_ids = {
            str(doc.get("document_id") or "")
            for turn in row["turns"]
            for doc in turn["documents"]
        }
        cited_doc_ids = set(re.findall(r"\[document_id=([^|\]]+)", row["actions"][-1]))
        invalid_evidence_refs += len({x.strip() for x in cited_doc_ids} - available_doc_ids)
        depth[row["actual_depth"]] = depth.get(row["actual_depth"], 0) + 1
        causal = row.get("minimal_depth_audit") or {}
        if causal:
            causal_records += 1
        if row["actual_depth"] == 1:
            d1_total += 1
            d1_stop_correct += int(
                causal.get("minimal_depth") == 1
                and causal.get("search1_sufficient") is True
                and causal.get("stop_after_search1_correct") is True
                and float(causal.get("final_f1") or 0) >= 0.8
            )
        elif row["actual_depth"] == 2:
            d2_total += 1
            d2_forced1_insufficient += int(
                causal.get("minimal_depth") == 2
                and causal.get("search1_sufficient") is False
                and causal.get("forced1_sufficient") is False
            )
            d2_positive_delta += int(
                causal.get("search2_useful") is True
                and float(causal.get("delta_grounded_acceptance") or 0) > 0
                and float(causal.get("final_f1") or 0) >= 0.8
            )
    forbidden = sum(
        any(k in example for k in ("gold_answers", "supporting_facts", "contexts"))
        for example in share + balanced
    )
    decision_distribution = {
        kind: sum((x.get("metadata") or {}).get("decision_type") == kind for x in share)
        for kind in ("initial_search", "post_obs_stop", "post_obs_continue")
    }

    def percentile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[round((len(ordered) - 1) * q)]
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
        "decision_distribution": decision_distribution,
        "balanced_decision_examples": len(balanced),
        "memory_tokens_p50": percentile(memory_token_values, 0.50),
        "memory_tokens_p95": percentile(memory_token_values, 0.95),
        "minimal_depth_audit_coverage": causal_records / max(1, len(rows)),
        "depth1_stop_correct_rate": d1_stop_correct / d1_total if d1_total else None,
        "depth2_forced1_insufficient_rate": d2_forced1_insufficient / d2_total if d2_total else None,
        "depth2_positive_delta_rate": d2_positive_delta / d2_total if d2_total else None,
    }
    quality_pass = (
        rows
        and metrics["duplicate_extra_query_rate"] < 0.05
        and (extra == 0 or metrics["obs_conditioned_extra_query_rate"] > 0.80)
        and (extra == 0 or metrics["new_source_extra_query_rate"] > 0.80)
        and empty == leaks == forbidden == invalid_evidence_refs == 0
        and metrics["minimal_depth_audit_coverage"] == 1.0
        and (d1_total == 0 or metrics["depth1_stop_correct_rate"] == 1.0)
        and (d2_total == 0 or metrics["depth2_forced1_insufficient_rate"] == 1.0)
        and (d2_total == 0 or metrics["depth2_positive_delta_rate"] == 1.0)
    )
    quotas = {int(k): int(v) for k, v in (summary_file.get("quotas") or {}).items()}
    required_depths = {d for d, quota in quotas.items() if quota > 0}
    represented_depths = {int(d) for d, count in depth.items() if count > 0}
    representation_pass = required_depths.issubset(represented_depths)
    metrics["required_depths_represented"] = representation_pass
    complete = len(rows) >= 100 and d1_total >= 20 and d2_total >= 20
    hard_pass = quality_pass and representation_pass and (a.allow_incomplete or complete)
    if quality_pass and complete:
        metrics["gate"] = "W3_DATA_GATE_PASS"
    elif quality_pass and representation_pass and a.allow_incomplete:
        metrics["gate"] = "W3_ACCEPTED_SUBSET_GATE_PASS"
    elif quality_pass and not representation_pass:
        metrics["gate"] = "W3_PARTIAL_DEPTH_SUBSET_FAIL"
    else:
        metrics["gate"] = "W3_DATA_GATE_FAIL"
    (a.run_dir / "audit_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not hard_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

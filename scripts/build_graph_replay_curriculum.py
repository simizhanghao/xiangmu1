#!/usr/bin/env python3
"""Compile train-only Hotpot supporting graphs into provider-decoupled D1/D2 replay states."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_web_multiturn_v2 import (
    action_text,
    build_decision_example,
    docs_serializable,
    internal_text,
)
from src.agents.research_memory import ResearchMemory, serialize_research_memory
from src.eval.metrics import normalize_answer
from src.sft.prototype_builder import load_jsonl
from src.tools.candidate_bm25 import format_observation_text

DEFAULT_POOL = Path(
    "/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)
STOP = {"what", "which", "where", "when", "who", "was", "were", "the", "that", "this", "does", "did", "are", "and", "for", "with", "from", "into", "about"}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--d1", type=int, default=8)
    p.add_argument("--d2", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer-path", type=Path, default=ROOT / "model")
    p.add_argument("--exclude-id-file", type=Path, action="append", default=[])
    p.add_argument("--split-name", default="smoke")
    return p.parse_args()


def visible(value: str, text: str) -> bool:
    needle = normalize_answer(value)
    return bool(needle) and f" {needle} " in f" {normalize_answer(text)} "


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def excluded_ids(extra: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in [
        *sorted((ROOT / "data/eval").glob("*.jsonl")),
        *sorted((ROOT / "data/sealed").glob("*.jsonl")),
        *sorted((ROOT / "data/web_v2").glob("*.jsonl")),
    ]:
        for row in load_jsonl(str(path)):
            value = row.get("sample_id") or row.get("question_id")
            if value:
                ids.add(str(value))
    for path in extra:
        if path.suffix == ".txt":
            ids.update(x.strip() for x in path.read_text().splitlines() if x.strip())
        else:
            for row in load_jsonl(str(path)):
                value = row.get("sample_id") or row.get("source_sample_id")
                if value:
                    ids.add(str(value).split("__graph_d", 1)[0])
    return ids


def replay_doc(sample_id: str, context: dict[str, Any], rank: int) -> dict[str, Any]:
    digest = hashlib.sha256(f"{sample_id}:{context['document_id']}".encode()).hexdigest()[:16]
    return {
        "document_id": f"replay_{digest}",
        "title": str(context["title"]),
        "text": str(context["text"]),
        "score": float(10 - rank),
        "metadata": {"url": f"replay://graph/{digest}"},
    }


def graph_candidate(sample: dict[str, Any]) -> dict[str, Any] | None:
    question = str(sample["question"])
    answer = str((sample.get("gold_answers") or [""])[0])
    contexts = {str(x["title"]): x for x in sample.get("contexts") or []}
    titles = list(dict.fromkeys(str(x["title"]) for x in sample.get("supporting_facts") or []))
    if len(titles) != 2 or any(x not in contexts for x in titles):
        return None
    answer_titles = [t for t in titles if visible(answer, contexts[t]["text"])]
    if len(answer_titles) != 1:
        return None
    title_b = answer_titles[0]
    title_a = titles[0] if titles[1] == title_b else titles[1]
    text_a, text_b = str(contexts[title_a]["text"]), str(contexts[title_b]["text"])
    if not visible(title_b, text_a) or visible(answer, text_a) or visible(title_b, question):
        return None
    facts_a = [str(x["sentence"]).strip() for x in sample["supporting_facts"] if str(x["title"]) == title_a]
    facts_b = [str(x["sentence"]).strip() for x in sample["supporting_facts"] if str(x["title"]) == title_b]
    if not facts_a or not facts_b or not any(visible(answer, x) for x in facts_b):
        return None
    distractors = [x for x in sample["contexts"] if str(x["title"]) not in {title_a, title_b} and not visible(answer, x["text"])]
    if len(distractors) < 8:
        return None
    return {
        "question": question,
        "answer": answer,
        "bridge": title_b,
        "a": contexts[title_a],
        "b": contexts[title_b],
        "fact_a": " ".join(facts_a),
        "fact_b": " ".join(facts_b),
        "distractors": distractors,
    }


def query2(bridge: str, question: str) -> str:
    words = [x for x in re.findall(r"[A-Za-z0-9'-]+", question) if x.lower() not in STOP]
    return " ".join([bridge, *words[-5:]])


def compile_row(sample: dict[str, Any], graph: dict[str, Any], depth: int, tokenizer: Any) -> dict[str, Any]:
    sid, question, answer, bridge = str(sample["sample_id"]), graph["question"], graph["answer"], graph["bridge"]
    a = replay_doc(sid, graph["a"], 0)
    b = replay_doc(sid, graph["b"], 0)
    distractors = [replay_doc(sid, x, i + 1) for i, x in enumerate(graph["distractors"])]
    obs_docs = [[a, *distractors[:4]], [b, *distractors[4:8]]] if depth == 2 else [[a, b, *distractors[:3]]]
    queries = [question] if depth == 1 else [question, query2(bridge, question)]
    memory = ResearchMemory(question, max_searches=depth, evidence_limit=8, char_budget=4500)
    actions: list[str] = []
    observations: list[str] = []
    turns: list[dict[str, Any]] = []
    decisions = [{
        "known": [], "missing": ["Evidence needed to answer the question"], "decision": "SEARCH",
        "next_query": queries[0], "answer": "", "source_ids": [], "rationale": "graph replay initial search",
    }]
    for turn_index, docs in enumerate(obs_docs):
        decision = decisions[turn_index]
        internal = internal_text(decision)
        memory.update_from_internal(internal)
        actions.append(action_text(decision, memory))
        novelty = memory.add_search(queries[turn_index], docs)
        rendered = serialize_research_memory(memory)
        observations.append(f"{format_observation_text(docs)}\n\n{rendered}")
        turns.append({
            "search_turn": turn_index + 1,
            "internal": internal,
            "query": queries[turn_index],
            "documents": docs_serializable(docs),
            "memory_rendered": rendered,
            "new_urls": novelty["new_urls"],
            "new_evidence": novelty["new_evidence"],
            "query_conditioning": "initial_query" if turn_index == 0 else "observation_entity",
            "missing_at_decision": list(decision["missing"]),
            "memory_tokens": len(tokenizer.encode(rendered, add_special_tokens=False)),
        })
        if depth == 2 and turn_index == 0:
            source_a = next(x.source_id for x in memory.evidence if x.document_id == a["document_id"])
            decisions.append({
                "known": [f"{graph['fact_a']} [{source_a}]"],
                "missing": [f"The fact about {bridge} needed to answer the question"],
                "decision": "SEARCH", "next_query": queries[1], "answer": "", "source_ids": [source_a],
                "rationale": "bridge visible; answer absent",
            })
    source_ids = [x.source_id for x in memory.evidence if x.document_id in {a["document_id"], b["document_id"]}]
    final = {
        "known": [graph["fact_a"], graph["fact_b"]], "missing": [], "decision": "ANSWER",
        "next_query": "", "answer": answer, "source_ids": source_ids, "rationale": "minimal evidence complete",
    }
    memory.update_from_internal(internal_text(final))
    actions.append(action_text(final, memory))
    examples = [build_decision_example(sid, question, actions, observations, i) for i in range(len(actions))]
    for example in examples:
        example["metadata"]["origin"] = "graph_replay_hotpot"
        example["metadata"]["replay_depth"] = depth
    causal = ({
        "minimal_depth": 1, "search1_sufficient": True, "stop_after_search1_correct": True, "final_f1": 1.0,
    } if depth == 1 else {
        "minimal_depth": 2, "search1_sufficient": False, "forced1_answer": "", "forced1_f1": 0.0,
        "forced1_sufficient": False, "forced1_missing": [f"Fact about {bridge}"], "forced1_source_ids": [],
        "final_f1": 1.0, "delta_f1_search2": 1.0, "delta_grounded_acceptance": 1.0,
        "search2_useful": True, "stop_after_search2_correct": True,
    })
    return {
        "sample_id": f"{sid}__graph_d{depth}", "question": question, "target_depth": depth,
        "actual_depth": depth, "actions": actions, "turns": turns, "final_answer": answer,
        "gold_accept_score": 1.0, "minimal_depth_audit": causal,
        "missing_after_search1": [] if depth == 1 else [f"Fact about {bridge}"],
        "decision_examples": examples, "origin": "graph_replay_hotpot",
        "graph_audit": {"bridge": bridge, "bridge_visible_obs1": True, "answer_hidden_obs1": depth == 2, "answer_visible_final": True},
    }


def main() -> None:
    cfg = args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.tokenizer_path), trust_remote_code=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    blocked = excluded_ids(cfg.exclude_id_file)
    rows = [x for x in load_jsonl(str(cfg.pool)) if str(x["sample_id"]) not in blocked]
    rows.sort(key=lambda x: hashlib.sha256(f"{cfg.seed}:{x['sample_id']}".encode()).hexdigest())
    graphs = [(x, graph_candidate(x)) for x in rows]
    graphs = [(x, g) for x, g in graphs if g is not None]
    need = cfg.d1 + cfg.d2
    if len(graphs) < need:
        raise SystemExit(f"GRAPH_REPLAY_SUPPLY_FAIL valid={len(graphs)} need={need}")
    trajectories = [compile_row(x, g, 1 if i < cfg.d1 else 2, tokenizer) for i, (x, g) in enumerate(graphs[:need])]
    examples = [e for row in trajectories for e in row["decision_examples"]]
    by_type = {k: [x for x in examples if x["metadata"]["decision_type"] == k] for k in ("initial_search", "post_obs_stop", "post_obs_continue")}
    n = min(len(by_type["post_obs_stop"]), len(by_type["post_obs_continue"]))
    stop_d1 = [x for x in by_type["post_obs_stop"] if x["metadata"]["replay_depth"] == 1]
    stop_d2 = [x for x in by_type["post_obs_stop"] if x["metadata"]["replay_depth"] == 2]
    chosen_stop: list[dict[str, Any]] = []
    for i in range(max(len(stop_d1), len(stop_d2))):
        if i < len(stop_d1): chosen_stop.append(stop_d1[i])
        if i < len(stop_d2): chosen_stop.append(stop_d2[i])
        if len(chosen_stop) >= n: break
    balanced = by_type["initial_search"][:n] + chosen_stop[:n] + by_type["post_obs_continue"][:n]
    dump(cfg.output_dir / "full_trajectories.jsonl", trajectories)
    dump(cfg.output_dir / "trajectories.jsonl", trajectories)
    dump(cfg.output_dir / "decision_sft.jsonl", examples)
    dump(cfg.output_dir / "decision_sft_balanced.jsonl", balanced)
    (cfg.output_dir / "selected_source_ids.txt").write_text(
        "\n".join(str(x[0]["sample_id"]) for x in graphs[:need]) + "\n", encoding="utf-8"
    )
    dump(
        cfg.output_dir / "graph_manifest.jsonl",
        [
            {
                "source_sample_id": str(x["sample_id"]), "assigned_depth": 1 if i < cfg.d1 else 2,
                "bridge": g["bridge"], "answer": g["answer"], "builder_oracle_only": True,
            }
            for i, (x, g) in enumerate(graphs[:need])
        ],
    )
    summary = {
        "gate": "GRAPH_REPLAY_BUILD_PASS", "trajectories": len(trajectories),
        "depth_counts": {"1": cfg.d1, "2": cfg.d2}, "decision_examples": len(examples),
        "decision_distribution": {k: len(v) for k, v in by_type.items()},
        "balanced_decision_examples": len(balanced), "origin": "graph_replay_hotpot",
        "eval_overlap": 0, "tool_contract": "ResearchMemory+candidate_bm25_observation+XML_action",
        "split_name": cfg.split_name,
        "excluded_source_ids": len(blocked),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

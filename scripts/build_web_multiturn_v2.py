#!/usr/bin/env python3
"""Build leak-safe variable-depth Web trajectories for Web-MultiTurn-v2.

The teacher never sees gold answers or supporting facts. Gold is used only by
deterministic builder-side accept/reject checks. Teacher JSON is compiled into
the frozen XML action protocol by code.
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

from src.agents.react_loop import WEB_MULTITURN_V2_SYSTEM_PROMPT
from src.agents.research_memory import ResearchMemory, serialize_research_memory
from src.eval.metrics import normalize_answer, token_f1
from src.sft.prototype_builder import load_jsonl
from src.tools.candidate_bm25 import format_observation_text
from src.tools.web_adapter import WebAdapter

DEFAULT_POOL = Path(
    "/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)
DEFAULT_OUT = ROOT / "results/56_web_multiturn_v2/pilot"
_QUERY_NORM = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]{2,}")
_LEAK = re.compile(r"hotpot.?qa|huggingface\.co/datasets|datasets-server|kaggle\.com/datasets", re.I)
_STOP = {
    "the", "and", "for", "was", "were", "what", "which", "who", "where", "when",
    "how", "does", "did", "with", "from", "that", "this", "into", "about", "their",
    "his", "her", "has", "have", "had", "are", "not", "but", "all", "between",
}
_ORDINAL_WORDS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    p.add_argument(
        "--candidate-manifest",
        type=Path,
        help="Depth-aware JSONL from mine_web_depth_candidates.py; required for formal pilot.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=120)
    p.add_argument("--max-candidates", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher-base-url", default=os.environ.get("TEACHER_BASE_URL", "http://10.16.137.2:8000/v1"))
    p.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "Kimi-K2.6-CT-FP8KV"))
    p.add_argument("--teacher-api-key", default=os.environ.get("TEACHER_API_KEY", "EMPTY"))
    p.add_argument("--teacher-timeout", type=float, default=180.0)
    p.add_argument("--teacher-temperature", type=float, default=0.0)
    p.add_argument("--teacher-seed", type=int, default=42)
    p.add_argument(
        "--diagnostic-log",
        type=Path,
        help="Optional JSONL of gold-blind teacher inputs/outputs for rejected-path diagnosis.",
    )
    p.add_argument("--web-timeout", type=float, default=30.0)
    p.add_argument("--web-retries", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--smoke", action="store_true", help="Build target=6 with quotas 2/2/2.")
    p.add_argument("--quota-depth1", type=int)
    p.add_argument("--quota-depth2", type=int)
    p.add_argument("--quota-depth3", type=int)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze_splits(pool: list[dict[str, Any]], seed: int) -> tuple[set[str], set[str]]:
    out = ROOT / "data/web_v2"
    out.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        pool,
        key=lambda x: hashlib.sha256(f"{seed}:{x['sample_id']}".encode()).hexdigest(),
    )
    dev, final = ranked[:50], ranked[50:100]
    paths = [("web_dev50", dev), ("web_final50", final)]
    for name, rows in paths:
        path = out / f"{name}.jsonl"
        ids_path = out / f"{name}_ids.txt"
        expected = [str(x["sample_id"]) for x in rows]
        if path.exists():
            existing = [str(x["sample_id"]) for x in load_jsonl(str(path))]
            if existing != expected:
                raise SystemExit(f"FROZEN_SPLIT_MISMATCH {path}")
        else:
            write_jsonl(path, rows)
            ids_path.write_text("\n".join(expected) + "\n", encoding="utf-8")
    return {str(x["sample_id"]) for x in dev}, {str(x["sample_id"]) for x in final}


def qnorm(value: str) -> str:
    return _QUERY_NORM.sub(" ", str(value).lower()).strip()


def words(value: str) -> set[str]:
    return {x for x in _WORD.findall(str(value).lower()) if x not in _STOP}


def acceptance_f1(prediction: str, golds: list[str]) -> float:
    def expand(value: str) -> str:
        tokens = str(value).split()
        return " ".join(_ORDINAL_WORDS.get(x.lower(), x) for x in tokens)

    prediction_expanded = expand(prediction)
    golds_expanded = [expand(x) for x in golds]
    score = token_f1(prediction_expanded, golds_expanded)
    pred_tokens = normalize_answer(prediction_expanded).split()
    if 0 < len(pred_tokens) <= 2:
        for gold in golds_expanded:
            gold_tokens = normalize_answer(gold).split()
            if set(pred_tokens).issubset(set(gold_tokens)):
                score = 1.0
    return score


def query_conditioning(
    query: str, question: str, prior_observation: str, missing: list[str]
) -> str:
    """Return the state signal used by a non-duplicate refinement query."""
    novel_from_observation = (words(query) - words(question)) & words(prior_observation)
    if novel_from_observation:
        return "observation_entity"
    missing_words = words(" ".join(missing))
    if qnorm(query) != qnorm(question) and words(query) & missing_words:
        return "missing_state_refinement"
    return "none"


def answer_hit(text: str, golds: list[str]) -> bool:
    hay = normalize_answer(text)
    padded = f" {hay} "
    return any(
        f" {normalize_answer(g)} " in padded for g in golds if normalize_answer(g)
    )


def teacher_call(cfg: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    system = """You are a Web research trajectory teacher. You never see reference answers.
Return one JSON object with keys: known (array of concise source-grounded claims), missing
(array of specific unresolved facts), decision (SEARCH or ANSWER), next_query (string),
answer (string), source_ids (array like S1), and rationale (short string).
Use only supplied Web evidence. For SEARCH, next_query must target Missing, incorporate a
concrete entity learned from observations when possible, and never repeat Previous Queries.
For ANSWER, missing must be empty and answer/source_ids must be grounded. The answer must
be the shortest directly answering span, not an explanation or sentence. Do not output XML."""
    headers = {"Content-Type": "application/json"}
    if cfg.teacher_api_key and cfg.teacher_api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {cfg.teacher_api_key}"
    body: dict[str, Any] = {
        "model": cfg.teacher_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": float(getattr(cfg, "teacher_temperature", 0.0)),
        "seed": int(getattr(cfg, "teacher_seed", 42)),
        "max_tokens": 1024,
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
                timeout=cfg.teacher_timeout,
            )
            if response.status_code >= 400 and attempt == 0:
                body.pop("thinking", None)
                body.pop("response_format", None)
                continue
            response.raise_for_status()
            message = (response.json().get("choices") or [{}])[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I | re.S)
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("teacher output is not an object")
            diagnostic_log = getattr(cfg, "diagnostic_log", None)
            if diagnostic_log:
                diagnostic_log.parent.mkdir(parents=True, exist_ok=True)
                append_jsonl(
                    diagnostic_log,
                    {
                        "question": payload.get("question"),
                        "decision_index": payload.get("decision_index"),
                        "instruction": payload.get("instruction"),
                        "research_memory": payload.get("research_memory"),
                        "latest_observation": payload.get("latest_observation"),
                        "teacher_output": value,
                        "gold_visible": False,
                    },
                )
            return value
        except Exception as exc:  # network/API failures are recorded by caller
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"teacher failed: {type(error).__name__}: {str(error)[:300]}")


def forced_answer_call(
    cfg: argparse.Namespace, question: str, memory: ResearchMemory, latest_observation: str
) -> dict[str, Any]:
    """Counterfactual A1: force ANSWER at the post-search1 state, without gold input."""
    return validate_decision(
        teacher_call(
            cfg,
            {
                "question": question,
                "decision_index": 1,
                "instruction": (
                    "Counterfactual evaluation: you must ANSWER now without another search. "
                    "Use only the supplied evidence. Return Decision=ANSWER even when uncertain."
                ),
                "research_memory": serialize_research_memory(memory),
                "latest_observation": latest_observation,
            },
        )
    )


def validate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    decision = str(raw.get("decision") or "").upper().strip()
    if decision not in {"SEARCH", "ANSWER"}:
        raise ValueError("invalid decision")
    known = [str(x).strip() for x in (raw.get("known") or []) if str(x).strip()][:8]
    missing = [str(x).strip() for x in (raw.get("missing") or []) if str(x).strip()][:3]
    query = " ".join(str(raw.get("next_query") or "").split())
    answer = " ".join(str(raw.get("answer") or "").split())
    source_ids = [str(x).strip().upper() for x in (raw.get("source_ids") or [])]
    if decision == "SEARCH" and not query:
        raise ValueError("SEARCH without next_query")
    if decision == "ANSWER" and not answer:
        raise ValueError("ANSWER without answer")
    return {
        "known": known,
        "missing": missing,
        "decision": decision,
        "next_query": query,
        "answer": answer,
        "source_ids": source_ids,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def internal_text(d: dict[str, Any]) -> str:
    known = "\n".join(f"- {x}" for x in d["known"]) or "- None"
    missing = "\n".join(f"- {x}" for x in d["missing"]) or "- None"
    next_line = f"\nNext Query: {d['next_query']}" if d["decision"] == "SEARCH" else ""
    return f"Known:\n{known}\nMissing:\n{missing}\nDecision: {d['decision']}{next_line}"


def action_text(d: dict[str, Any], memory: ResearchMemory) -> str:
    internal = internal_text(d)
    if d["decision"] == "SEARCH":
        return f"<internal>\n{internal}\n</internal>\n<search>\n{d['next_query']}\n</search>"
    chosen = [x for x in memory.evidence if x.source_id in set(d["source_ids"])]
    if not chosen:
        chosen = sorted(memory.evidence, key=lambda x: -x.score)[:2]
    evidence = "\n\n".join(
        f"[document_id={x.document_id} | title={x.title} | sentence_id=0]\n{x.snippet[:700]}"
        for x in chosen[:4]
    )
    return (
        f"<internal>\n{internal}\n</internal>\n"
        f"<evidence>\n{evidence}\n</evidence>\n<answer>\n{d['answer']}\n</answer>"
    )


def docs_serializable(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": str(x.get("document_id") or ""),
            "title": str(x.get("title") or ""),
            "text": str(x.get("text") or ""),
            "score": float(x.get("score") or 0.0),
            "metadata": {"url": str((x.get("metadata") or {}).get("url") or "")},
        }
        for x in docs
    ]


def build_decision_example(
    sample_id: str,
    question: str,
    actions: list[str],
    observations: list[str],
    target_index: int,
) -> dict[str, Any]:
    conv: list[dict[str, str]] = [{"from": "human", "value": f"Question: {question}"}]
    for i in range(target_index):
        conv.append({"from": "gpt", "value": actions[i]})
        obs = observations[i] if i == target_index - 1 else "[Earlier raw observation compressed into Research Memory.]"
        conv.append({"from": "observation", "value": obs})
    conv.append({"from": "gpt", "value": actions[target_index]})
    return {
        "conversations": conv,
        "system": WEB_MULTITURN_V2_SYSTEM_PROMPT,
        "sft_id": f"{sample_id}__webmt_v2_decision_{target_index}",
        "sample_id": sample_id,
        "category": "web_multiturn_v2",
        "metadata": {"decision_index": target_index, "observation_role": "sharegpt_observation"},
    }


def build_one(
    sample: dict[str, Any],
    target_depth: int,
    cfg: argparse.Namespace,
    web: WebAdapter,
    initial_query: str | None = None,
    initial_documents: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    question = str(sample["question"])
    initial_query = " ".join(str(initial_query or question).split())
    golds = [str(x) for x in sample.get("gold_answers") or []]
    memory = ResearchMemory(question, max_searches=target_depth, evidence_limit=8, char_budget=4500)
    actions: list[str] = []
    observations: list[str] = []
    turns: list[dict[str, Any]] = []
    accumulated = ""
    previous_observation = ""
    forced1: dict[str, Any] | None = None
    missing_after_search1: list[str] = []
    for decision_index in range(target_depth + 1):
        if decision_index == 0:
            # Causal alignment: candidate mining and trajectory construction must
            # evaluate the identical Search1/Obs1 state. W2 already establishes
            # full-question Search1 as the frozen policy baseline.
            decision = {
                "known": [],
                "missing": ["Evidence needed to answer the question"],
                "decision": "SEARCH",
                "next_query": initial_query,
                "answer": "",
                "source_ids": [],
                "rationale": "fixed mined Search1",
            }
        else:
            payload = {
                "question": question,
                "desired_max_depth": target_depth,
                "decision_index": decision_index,
                "instruction": (
                    "Use the minimum sufficient sequential searches. For depth>1, decompose dependencies; "
                    "do not copy the full question as every query. ANSWER immediately when evidence is sufficient."
                ),
                "research_memory": serialize_research_memory(memory),
                "latest_observation": previous_observation,
            }
            decision = validate_decision(teacher_call(cfg, payload))
        if len(memory.searches) == 1:
            missing_after_search1 = list(decision["missing"])
        internal = internal_text(decision)
        memory.update_from_internal(internal)
        action = action_text(decision, memory)
        actions.append(action)
        if decision["decision"] == "ANSWER":
            if not memory.searches:
                return None, "answer_without_search"
            if len(memory.searches) != target_depth:
                return None, "depth_mismatch"
            if acceptance_f1(decision["answer"], golds) < 0.8:
                return None, "teacher_answer_mismatch"
            available_sources = {x.source_id for x in memory.evidence}
            if decision["missing"]:
                return None, "answer_with_unresolved_missing"
            if not decision["source_ids"] or not set(decision["source_ids"]).issubset(available_sources):
                return None, "answer_not_grounded_in_available_sources"
            examples = [
                build_decision_example(str(sample["sample_id"]), question, actions, observations, i)
                for i in range(len(actions))
            ]
            final_f1 = acceptance_f1(decision["answer"], golds)
            if target_depth == 1:
                minimal_depth_audit = {
                    "minimal_depth": 1,
                    "search1_sufficient": True,
                    "stop_after_search1_correct": True,
                    "final_f1": final_f1,
                }
            else:
                if forced1 is None:
                    return None, "missing_forced1_counterfactual"
                forced1_f1 = acceptance_f1(forced1["answer"], golds)
                delta_f1 = final_f1 - forced1_f1
                forced1_sources = set(forced1["source_ids"])
                forced1_sufficient = (
                    forced1_f1 >= 0.8
                    and not forced1["missing"]
                    and bool(forced1_sources)
                    and forced1_sources.issubset(available_sources)
                )
                if forced1_sufficient:
                    return None, "forced1_already_correct"
                delta_grounded_acceptance = 1.0 - float(forced1_sufficient)
                if delta_grounded_acceptance <= 0:
                    return None, "search2_no_positive_grounded_delta"
                minimal_depth_audit = {
                    "minimal_depth": 2,
                    "search1_sufficient": False,
                    "forced1_answer": forced1["answer"],
                    "forced1_f1": forced1_f1,
                    "forced1_sufficient": forced1_sufficient,
                    "forced1_missing": list(forced1["missing"]),
                    "forced1_source_ids": list(forced1["source_ids"]),
                    "final_f1": final_f1,
                    "delta_f1_search2": delta_f1,
                    "delta_grounded_acceptance": delta_grounded_acceptance,
                    "search2_useful": True,
                    "stop_after_search2_correct": True,
                }
            return {
                "sample_id": sample["sample_id"],
                "question": question,
                "target_depth": target_depth,
                "actual_depth": len(memory.searches),
                "actions": actions,
                "turns": turns,
                "final_answer": decision["answer"],
                "gold_accept_score": acceptance_f1(decision["answer"], golds),
                "minimal_depth_audit": minimal_depth_audit,
                "missing_after_search1": missing_after_search1,
                "decision_examples": examples,
                "counterfactual_pairs": [
                    {
                        "state_after_search": i + 1,
                        "preferred": "ANSWER" if i + 1 == target_depth else "SEARCH_REFINED",
                        "rejected": "SEARCH_DUPLICATE" if i + 1 == target_depth else "ANSWER_PREMATURE",
                    }
                    for i in range(target_depth)
                ],
            }, "accepted"
        if decision_index >= target_depth:
            return None, "teacher_over_depth"
        query = decision["next_query"]
        if any(qnorm(query) == qnorm(x.query) for x in memory.searches):
            return None, "duplicate_query"
        if memory.searches:
            conditioning = query_conditioning(
                query, question, previous_observation, decision["missing"]
            )
            if conditioning == "none":
                return None, "query_not_observation_conditioned"
        else:
            conditioning = "initial_query"
        if not memory.searches and initial_documents is not None:
            # Reuse the exact mined Obs1. Re-querying a live provider would make
            # the minimal-depth label and forced-answer counterfactual refer to
            # different evidence even when Query1 is identical.
            packed = {"documents": initial_documents, "errors": []}
        else:
            packed = web.retrieve(sample, query, cfg.top_k)
        docs = list(packed.get("documents") or [])
        if packed.get("errors"):
            return None, "web_error"
        if not docs:
            return None, "empty_observation"
        if any(_LEAK.search(str((x.get("metadata") or {}).get("url") or "")) for x in docs):
            return None, "benchmark_leak"
        before_urls = set(memory.visited_urls)
        before_evidence = len(memory.evidence)
        novelty = memory.add_search(query, docs)
        if len(memory.searches) > 1 and (novelty["new_urls"] <= 0 or novelty["new_evidence"] <= 0):
            return None, "no_new_information"
        raw = format_observation_text(docs)
        rendered = serialize_research_memory(memory)
        observation = f"{raw}\n\n{rendered}"
        observations.append(observation)
        previous_observation = raw
        accumulated += "\n" + raw
        if target_depth == 2 and len(memory.searches) == 1:
            try:
                forced1 = forced_answer_call(cfg, question, memory, raw)
            except Exception:
                return None, "forced1_teacher_error"
            if forced1["decision"] != "ANSWER":
                return None, "forced1_protocol_error"
            forced1_sources = set(forced1["source_ids"])
            available_sources = {x.source_id for x in memory.evidence}
            forced1_sufficient = (
                acceptance_f1(forced1["answer"], golds) >= 0.8
                and not forced1["missing"]
                and bool(forced1_sources)
                and forced1_sources.issubset(available_sources)
            )
            if forced1_sufficient:
                return None, "forced1_already_correct"
        # A deeper teacher path is invalid if the answer was already explicit.
        if len(memory.searches) < target_depth and answer_hit(accumulated, golds):
            return None, "prematurely_sufficient"
        turns.append(
            {
                "search_turn": len(memory.searches),
                "internal": internal,
                "query": query,
                "documents": docs_serializable(docs),
                "memory_rendered": rendered,
                "new_urls": len(memory.visited_urls - before_urls),
                "new_evidence": len(memory.evidence) - before_evidence,
                "query_conditioning": conditioning,
                "missing_at_decision": list(decision["missing"]),
            }
        )
    return None, "no_final_answer"


def main() -> None:
    cfg = args()
    if cfg.smoke:
        cfg.target = 6
        cfg.max_candidates = min(cfg.max_candidates, 60)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_jsonl(str(cfg.pool))
    dev_ids, final_ids = freeze_splits(pool, cfg.seed)
    candidates = [x for x in pool if str(x["sample_id"]) not in dev_ids | final_ids]
    mined_depth: dict[str, int] = {}
    mined_query: dict[str, str] = {}
    mined_documents: dict[str, list[dict[str, Any]]] = {}
    if cfg.candidate_manifest:
        manifest = load_jsonl(str(cfg.candidate_manifest))
        mined_depth = {
            str(x["sample_id"]): int(x["likely_depth"])
            for x in manifest
            if int(x.get("likely_depth") or 0) in (1, 2)
        }
        mined_query = {
            str(x["sample_id"]): str(x.get("search_query") or "").strip()
            for x in manifest
        }
        mined_documents = {
            str(x["sample_id"]): list(x.get("search1_documents") or [])
            for x in manifest
        }
        candidates = [x for x in candidates if str(x["sample_id"]) in mined_depth]
    rng = random.Random(cfg.seed)
    rng.shuffle(candidates)
    candidates = candidates[: cfg.max_candidates]
    explicit_quotas = (cfg.quota_depth1, cfg.quota_depth2, cfg.quota_depth3)
    if any(x is not None for x in explicit_quotas):
        if any(x is None or x < 0 for x in explicit_quotas):
            raise SystemExit("QUOTA_CONFIG_FAIL: set all three quotas to non-negative integers")
        quotas = {1: cfg.quota_depth1, 2: cfg.quota_depth2, 3: cfg.quota_depth3}
        if sum(quotas.values()) != cfg.target:
            raise SystemExit(f"QUOTA_CONFIG_FAIL: quota sum {sum(quotas.values())} != target {cfg.target}")
    elif cfg.smoke:
        quotas = {1: 2, 2: 2, 3: 2}
    else:
        # HotpotQA + Brave LLM Context rarely needs a genuine third request. Keep
        # the formal pilot minimal-depth by default; depth-3 can be supplied
        # explicitly when a genuinely three-hop source pool is available.
        depth1 = round(cfg.target * 0.45)
        quotas = {1: depth1, 2: cfg.target - depth1, 3: 0}
    accepted_path = cfg.output_dir / "trajectories.jsonl"
    rejected_path = cfg.output_dir / "rejected.jsonl"
    if not cfg.resume:
        for path in (accepted_path, rejected_path):
            path.unlink(missing_ok=True)
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    web = WebAdapter(
        provider="brave_llm_context",
        cache_dir=cfg.output_dir / "web_cache",
        timeout_s=cfg.web_timeout,
        retries=cfg.web_retries,
        llm_context_tokens=4096,
    )
    attempt = 0
    for sample in candidates:
        open_depths = [d for d in (2, 1, 3) if sum(x["actual_depth"] == d for x in accepted) < quotas[d]]
        if not open_depths:
            break
        if mined_depth:
            target_depth = mined_depth[str(sample["sample_id"])]
            if target_depth not in open_depths:
                continue
        else:
            target_depth = open_depths[attempt % len(open_depths)]
        attempt += 1
        try:
            row, reason = build_one(
                sample,
                target_depth,
                cfg,
                web,
                initial_query=mined_query.get(str(sample["sample_id"])) or str(sample["question"]),
                initial_documents=mined_documents.get(str(sample["sample_id"])),
            )
        except Exception as exc:
            row, reason = None, f"exception_{type(exc).__name__}"
        if row is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            append_jsonl(rejected_path, {"sample_id": sample["sample_id"], "target_depth": target_depth, "reason": reason})
        else:
            accepted.append(row)
            append_jsonl(accepted_path, row)
            print(f"[{len(accepted)}/{cfg.target}] depth={target_depth} {sample['sample_id']}", flush=True)
    examples = [e for row in accepted for e in row["decision_examples"]]
    write_jsonl(cfg.output_dir / "sharegpt.jsonl", examples)
    write_jsonl(cfg.output_dir / "decision_sft.jsonl", examples)
    counts = {str(d): sum(x["actual_depth"] == d for x in accepted) for d in (1, 2, 3)}
    summary = {
        "gate": "W3_PILOT_BUILT" if len(accepted) == cfg.target else "W3_PILOT_INCOMPLETE",
        "target": cfg.target,
        "accepted": len(accepted),
        "decision_examples": len(examples),
        "depth_counts": counts,
        "quotas": {str(k): v for k, v in quotas.items()},
        "attempted": attempt,
        "rejection_reasons": reasons,
        "teacher_model": cfg.teacher_model,
        "teacher_gold_visible": False,
        "web_provider": "brave_llm_context",
        "candidate_manifest": str(cfg.candidate_manifest) if cfg.candidate_manifest else None,
        "minimal_depth_counterfactual": True,
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if len(accepted) != cfg.target:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

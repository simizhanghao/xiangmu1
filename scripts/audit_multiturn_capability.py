#!/usr/bin/env python3
"""L3 audit: are search=2 traces adaptive multi-turn, or duplicate retries?

Offline (default): query novelty, obs1 dependence, new evidence.
--counterfactual: replay search1 then forbid search2 (needs vLLM).
No training. Does not pick a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

REPO = Path(__file__).resolve().parents[1]
V2B = (
    REPO
    / "results/51_heldout_test/n500_grpo400_finalize_v2b"
    / "agent_rollout_n500_20260820_125356_heldout_n500_grpo400_finalize_v2b"
)
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "at", "as", "is", "was", "were", "be", "been", "are",
    "that", "this", "these", "those", "who", "whom", "which", "what",
    "when", "where", "why", "how", "did", "does", "do", "had", "has",
    "have", "its", "his", "her", "their", "our", "your", "also",
}
_TITLE_RE = re.compile(r"\]\s*([^:\n]{1,120}):")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-turn capability audit.")
    p.add_argument("--config", type=str, default=str(REPO / "config" / "harness_v1.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "results" / "52_multiturn_capability" / "n54_offline"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--metrics", type=str, default=str(V2B / "metrics.jsonl"))
    p.add_argument("--traces", type=str, default=str(V2B / "trace.jsonl"))
    p.add_argument(
        "--eval-file",
        type=str,
        default=str(REPO / "data" / "sealed" / "hotpotqa_test500.jsonl"),
    )
    p.add_argument("--counterfactual", action="store_true")
    p.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:18120/v1")
    p.add_argument("--vllm-model-name", type=str, default="sft8b")
    p.add_argument(
        "--model-path",
        type=str,
        default=str(REPO / "results" / "44_hf_formal_grpo_step400" / "model_view"),
    )
    return p.parse_args()


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def tokens(text: str) -> Set[str]:
    return {
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOP and len(t) >= 3
    }


def norm_query(text: str) -> str:
    raw = (text or "").lower()
    raw = "".join(ch if ch not in string.punctuation else " " for ch in raw)
    return " ".join(raw.split())


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def titles_from_obs(obs: str) -> List[str]:
    return [m.group(1).strip() for m in _TITLE_RE.finditer(obs or "")]


def obs_list(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in (trace.get("steps") or []) if s.get("step_type") == "observation"]


def novelty(q1: str, q2: str) -> str:
    if (q1 or "").strip() == (q2 or "").strip():
        return "exact_duplicate"
    if norm_query(q1) == norm_query(q2):
        return "normalized_duplicate"
    if jaccard(tokens(q1), tokens(q2)) >= 0.85:
        return "semantic_duplicate"
    return "rewrite"


def gold_titles(sample: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for sf in sample.get("supporting_facts") or []:
        if isinstance(sf, dict) and sf.get("title"):
            out.append(str(sf["title"]).strip())
    return out


def title_hit(title: str, obs: str) -> bool:
    t = (title or "").strip().lower()
    return bool(t) and t in (obs or "").lower()


def audit_one(
    row: Dict[str, Any],
    trace: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    qs = [str(q) for q in (row.get("search_queries") or [])]
    q1 = qs[0] if qs else ""
    q2 = qs[1] if len(qs) > 1 else ""
    question = str(sample.get("question") or trace.get("question") or "")
    obs_steps = obs_list(trace)
    obs1 = str(obs_steps[0].get("content") or "") if obs_steps else ""
    obs2 = str(obs_steps[1].get("content") or "") if len(obs_steps) > 1 else ""
    ids1 = [str(x) for x in (obs_steps[0].get("document_ids") or [])] if obs_steps else []
    ids2 = [str(x) for x in (obs_steps[1].get("document_ids") or [])] if len(obs_steps) > 1 else []
    titles1 = titles_from_obs(obs1)
    titles2 = titles_from_obs(obs2)
    nov = novelty(q1, q2)
    novel_q2 = tokens(q2) - tokens(question)
    obs_overlap = sorted(novel_q2 & tokens(obs1))
    sf = gold_titles(sample)
    sf_in_1 = [t for t in sf if title_hit(t, obs1)]
    sf_in_2 = [t for t in sf if title_hit(t, obs2)]
    new_sf = [t for t in sf_in_2 if t not in sf_in_1]
    m = row.get("metrics") or {}
    return {
        "sample_id": row.get("sample_id"),
        "question": question,
        "q1": q1,
        "q2": q2,
        "novelty": nov,
        "query_jaccard": round(jaccard(tokens(q1), tokens(q2)), 4),
        "obs_conditioned": bool(obs_overlap),
        "obs_conditioned_tokens": obs_overlap[:12],
        "new_doc_ids": sorted(set(ids2) - set(ids1)),
        "new_doc_count": len(set(ids2) - set(ids1)),
        "new_titles": [t for t in titles2 if t not in titles1],
        "new_title_count": len([t for t in titles2 if t not in titles1]),
        "new_supporting_fact_titles": new_sf,
        "new_supporting_fact_count": len(new_sf),
        "two_search_f1": float(m.get("token_f1") or 0.0),
        "two_search_em": float(m.get("exact_match") or 0.0),
        "two_search_evidence_f1": float(m.get("evidence_f1") or 0.0),
        "finished": bool(row.get("finished")),
    }


def label_row(r: Dict[str, Any]) -> str:
    if r["novelty"] in {"exact_duplicate", "normalized_duplicate", "semantic_duplicate"}:
        if r["new_doc_count"] == 0 and r["new_supporting_fact_count"] == 0:
            return "duplicate_retry"
        return "duplicate_but_new_docs"
    if r["obs_conditioned"] and (r["new_doc_count"] > 0 or r["new_supporting_fact_count"] > 0):
        return "obs_conditioned_hop"
    if r["obs_conditioned"]:
        return "obs_conditioned_no_new_docs"
    return "rewrite_not_obs_conditioned"


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    nov = Counter(r["novelty"] for r in rows)
    lab = Counter(r["label"] for r in rows)
    return {
        "n_search2": n,
        "novelty": dict(nov),
        "novelty_rate": {k: round(v / n, 4) if n else 0.0 for k, v in nov.items()},
        "labels": dict(lab),
        "label_rate": {k: round(v / n, 4) if n else 0.0 for k, v in lab.items()},
        "obs_conditioned_rate": round(sum(1 for r in rows if r["obs_conditioned"]) / n, 4) if n else 0.0,
        "new_doc_rate": round(sum(1 for r in rows if r["new_doc_count"] > 0) / n, 4) if n else 0.0,
        "new_sf_rate": round(sum(1 for r in rows if r["new_supporting_fact_count"] > 0) / n, 4) if n else 0.0,
        "mean_two_search_f1": round(sum(r["two_search_f1"] for r in rows) / n, 4) if n else 0.0,
        "note": (
            "L3 is adaptive only if rewrite + obs-conditioned + new evidence "
            "and counterfactual ΔF1>0. Duplicate retry is tool re-use, not DeepResearch."
        ),
    }


def run_counterfactual(
    rows: List[Dict[str, Any]],
    samples: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import sys

    sys.path.insert(0, str(REPO))
    from transformers import AutoTokenizer

    from src.agents.react_loop import (
        RolloutConfig,
        make_openai_completions_fn,
        run_search_agent_rollout,
    )

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    gen = make_openai_completions_fn(args.vllm_base_url, args.vllm_model_name)
    cfg = RolloutConfig(top_k=5, max_search_turns=1, temperature=0.0)
    for i, r in enumerate(rows, 1):
        sid = str(r["sample_id"])
        sample = samples[sid]
        result = run_search_agent_rollout(
            sample,
            None,
            tok,
            cfg,
            generate_fn=gen,
            prefix_search_queries=[r["q1"]],
            finalize_after_prefix=True,
        )
        r["forced1_f1"] = float(result.metrics.get("token_f1") or 0.0)
        r["forced1_em"] = float(result.metrics.get("exact_match") or 0.0)
        r["forced1_finished"] = bool(result.finished)
        r["delta_f1_search2"] = round(r["two_search_f1"] - r["forced1_f1"], 4)
        print(
            f"[cf {i}/{len(rows)}] {sid} 2hopF1={r['two_search_f1']:.3f} "
            f"forced1={r['forced1_f1']:.3f} dF1={r['delta_f1_search2']:+.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_jsonl(Path(args.metrics))
    traces = {str(t.get("sample_id")): t for t in load_jsonl(Path(args.traces))}
    samples = {str(s.get("sample_id")): s for s in load_jsonl(Path(args.eval_file))}
    picked: List[Dict[str, Any]] = []
    for row in metrics:
        sc = int((row.get("metrics") or {}).get("search_count") or 0)
        if sc != 2:
            continue
        sid = str(row.get("sample_id"))
        if sid not in traces or sid not in samples:
            continue
        rec = audit_one(row, traces[sid], samples[sid])
        rec["label"] = label_row(rec)
        picked.append(rec)
        if args.max_samples and len(picked) >= args.max_samples:
            break

    if args.counterfactual:
        run_counterfactual(picked, samples, args)
        n = len(picked)
        d = [r.get("delta_f1_search2") for r in picked if r.get("delta_f1_search2") is not None]
        extra = {
            "counterfactual_n": n,
            "mean_forced1_f1": round(sum(float(r.get("forced1_f1") or 0) for r in picked) / n, 4) if n else 0.0,
            "mean_delta_f1_search2": round(sum(d) / len(d), 4) if d else 0.0,
            "share_search2_helps": round(sum(1 for x in d if x > 1e-9) / len(d), 4) if d else 0.0,
            "share_search2_hurts": round(sum(1 for x in d if x < -1e-9) / len(d), 4) if d else 0.0,
        }
    else:
        extra = {"counterfactual": False}

    summary = summarize(picked)
    summary.update(extra)
    summary.update(
        {
            "metrics": args.metrics,
            "traces": args.traces,
            "eval_file": args.eval_file,
            "seed": args.seed,
            "config": args.config,
            "debug": bool(args.debug),
        }
    )
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked),
        encoding="utf-8",
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("CAPABILITY_AUDIT_OK" if picked else "CAPABILITY_AUDIT_EMPTY")


if __name__ == "__main__":
    main()

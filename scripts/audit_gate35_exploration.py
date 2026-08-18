#!/usr/bin/env python3
"""Gate 3.5: rollout-only exploration audit. No backward.

Uses RL smoke parquet (not frozen-dev@200). Reward is the frozen Evidence GRPO
formula: R = R_answer + 0.5 R_evidence + 0.1 R_format, cost λ = 0.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

_EMPTY_THINK = re.compile(r"<think>\s*</think>", re.IGNORECASE)
_OBS_TAG = re.compile(r"<observation\b", re.IGNORECASE)
_EXTRA_CONTINUE = re.compile(
    r"Continue\.\s*Prefer|<search> again only if necessary", re.IGNORECASE
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.react_loop import (  # noqa: E402
    RolloutConfig,
    make_openai_completions_fn,
    run_search_agent_rollout,
)
from src.rl.rewards_evidence import compute_score  # noqa: E402

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 3.5 rollout-only exploration.")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "Dee" / "config" / "harness_v1.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "Dee" / "results" / "28_gate35_exploration_32x8"),
    )
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--n-rollouts", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--train-parquet",
        type=str,
        default=str(REPO_ROOT / "data" / "rl" / "train_smoke_128" / "train.parquet"),
    )
    p.add_argument(
        "--contexts-index",
        type=str,
        default=str(REPO_ROOT / "data" / "rl" / "train_smoke_128" / "contexts_index.jsonl"),
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=str(REPO_ROOT / "Dee" / "outputs" / "22_sft_qwen3_8b_merged"),
    )
    p.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:18000/v1")
    p.add_argument("--vllm-model-name", type=str, default="sft8b")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-search-turns", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def golds_from_rm(rm: Dict[str, Any]) -> List[str]:
    gt = rm.get("ground_truth")
    if isinstance(gt, dict):
        t = gt.get("target", gt.get("gold_answers"))
        if isinstance(t, list):
            return [str(x) for x in t]
        if t is not None:
            return [str(t)]
    if isinstance(gt, list):
        return [str(x) for x in gt]
    if gt is not None:
        return [str(gt)]
    return []


def load_contexts(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


def load_smoke_samples(
    parquet_path: Path, contexts: Dict[str, Dict[str, Any]], max_samples: int
) -> List[Dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ei = r["extra_info"]
        if not isinstance(ei, dict):
            ei = dict(ei)
        rm = r["reward_model"]
        if not isinstance(rm, dict):
            rm = dict(rm)
        sid = str(ei.get("sample_id") or "")
        if not sid:
            raise SystemExit("missing sample_id in smoke parquet")
        if sid.startswith("hotpotqa_distractor_validation_"):
            continue
        ctx_row = contexts.get(sid) or {}
        sf = ei.get("supporting_facts")
        if sf is None:
            sf = []
        elif hasattr(sf, "tolist"):
            sf = list(sf.tolist())
        else:
            sf = list(sf)
        rows.append(
            {
                "sample_id": sid,
                "question": str(ei.get("question") or ctx_row.get("question") or ""),
                "gold_answers": golds_from_rm(rm),
                "contexts": list(ctx_row.get("contexts") or []),
                "supporting_facts": list(sf),
            }
        )
        if max_samples and len(rows) >= max_samples:
            break
    if not rows:
        raise SystemExit(f"no smoke prompts loaded from {parquet_path}")
    return rows


def group_std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def advantages(xs: Sequence[float]) -> List[float]:
    if not xs:
        return []
    mu = float(statistics.mean(xs))
    sd = group_std(xs)
    denom = sd if sd > EPS else 1.0
    return [(float(x) - mu) / denom for x in xs]


def protocol_flags(raw_generations: Sequence[str]) -> Dict[str, int]:
    text = "\n".join(raw_generations or [])
    return {
        "empty_think": len(_EMPTY_THINK.findall(text)),
        "observation_tag": len(_OBS_TAG.findall(text)),
        "extra_continue": len(_EXTRA_CONTINUE.findall(text)),
    }


def judge(summary: Dict[str, Any]) -> str:
    n_ok = int(summary["n_trajectories"]) == int(summary["n_expected"])
    finish_ok = float(summary["finish_rate"]) >= 0.95
    parse_ok = float(summary["parse_rate"]) >= 0.95
    proto_ok = (
        int(summary.get("empty_think_n") or 0) == 0
        and int(summary.get("observation_tag_n") or 0) == 0
        and int(summary.get("extra_continue_n") or 0) == 0
    )
    var_ok = float(summary["group_reward_std_nonzero_rate"]) >= 0.30
    adv_ok = float(summary["nonzero_advantage_rate"]) >= 0.20
    cq = float(summary.get("conditional_query_diversity_rate") or 0.0)
    rdiv = float(summary["route_diversity_rate"])
    reward_std = float(summary["reward_std"])
    if not (n_ok and finish_ok and parse_ok and proto_ok):
        return "GATE35_EXPLORATION_FAIL"
    if reward_std <= 0 or not var_ok or not adv_ok:
        if cq >= 0.20 or rdiv >= 0.30:
            return "GATE35_REWARD_FLAT"
        return "GATE35_UNDEREXPLORE"
    if cq >= 0.20:
        return "GATE35_EXPLORATION_PASS"
    return "GATE35_UNDEREXPLORE"


def trial_a_branch(summary: Dict[str, Any]) -> str:
    if summary.get("gate") == "GATE35_EXPLORATION_FAIL":
        return "PROTOCOL_FAIL"
    cq = float(summary.get("conditional_query_diversity_rate") or 0.0)
    if cq >= 0.20:
        return "STOP_TUNE"
    return "TRIAL_B"


def trial_b_branch(summary: Dict[str, Any]) -> str:
    if summary.get("gate") == "GATE35_EXPLORATION_FAIL":
        return "PROTOCOL_FAIL"
    cq = float(summary.get("conditional_query_diversity_rate") or 0.0)
    if cq >= 0.20:
        return "STOP_TUNE"
    return "TRIAL_C"


def trial_c_branch(summary: Dict[str, Any]) -> str:
    if summary.get("gate") == "GATE35_EXPLORATION_FAIL":
        return "PROTOCOL_FAIL"
    cq = float(summary.get("conditional_query_diversity_rate") or 0.0)
    if cq >= 0.20:
        return "STOP_TUNE"
    return "STOP_SWEEP"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    samples = load_smoke_samples(
        Path(args.train_parquet), load_contexts(Path(args.contexts_index)), args.max_samples
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    generate_fn = make_openai_completions_fn(args.vllm_base_url, args.vllm_model_name)
    cfg = RolloutConfig(
        top_k=args.top_k,
        max_search_turns=args.max_search_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"[gate35] n_prompts={len(samples)} n_rollouts={args.n_rollouts} "
        f"T={args.temperature} top_p={args.top_p} url={args.vllm_base_url}",
        flush=True,
    )

    traj_path = out_dir / "rollouts.jsonl"
    group_path = out_dir / "groups.jsonl"
    all_traj: List[Dict[str, Any]] = []
    groups: List[Dict[str, Any]] = []
    with traj_path.open("w", encoding="utf-8") as tf, group_path.open(
        "w", encoding="utf-8"
    ) as gf:
        for pi, sample in enumerate(samples):
            group_rows: List[Dict[str, Any]] = []
            for ri in range(args.n_rollouts):
                seed = args.seed * 100_000 + pi * 100 + ri
                result = run_search_agent_rollout(
                    sample,
                    None,
                    tokenizer,
                    cfg,
                    generate_fn=generate_fn,
                    generation_seed=seed,
                )
                solution = "\n".join(result.raw_generations or [])
                scored = compute_score(
                    solution_str=solution,
                    ground_truth={
                        "target": sample.get("gold_answers") or [],
                        "supporting_facts": sample.get("supporting_facts") or [],
                    },
                    extra_info={
                        "sample_id": sample["sample_id"],
                        "supporting_facts": sample.get("supporting_facts") or [],
                        "search_count": result.metrics.get("search_count"),
                        "reward_weights": {
                            "answer_weight": 1.0,
                            "evidence_weight": 0.5,
                            "format_weight": 0.1,
                            "search_cost_weight": 0.0,
                        },
                    },
                )
                evid = ""
                for st in result.trace.steps:
                    if st.step_type == "evidence":
                        evid = st.content
                        break
                pred = scored.get("pred") or ""
                proto = protocol_flags(result.raw_generations or [])
                question = str(sample.get("question") or "")
                row = {
                    "sample_id": sample["sample_id"],
                    "question": question,
                    "rollout_id": ri,
                    "route": result.route_first,
                    "queries": list(result.search_queries),
                    "search_count": int(result.metrics.get("search_count") or 0),
                    "answer": pred,
                    "evidence": evid,
                    "finished": bool(result.finished),
                    "parse_ok": float(result.metrics.get("format_valid") or 0.0) >= 1.0,
                    "R_answer": float(scored.get("answer_reward") or 0.0),
                    "R_evidence": float(scored.get("evidence_reward") or 0.0),
                    "R_format": float(scored.get("format_reward") or 0.0),
                    "R_total": float(scored.get("total_reward") or 0.0),
                    "em": float(scored.get("em") or 0.0),
                    "token_f1": float(result.metrics.get("token_f1") or 0.0),
                    "duplicate_query_count": int(
                        result.metrics.get("duplicate_query_count") or 0
                    ),
                    "exact_question_copy": any(q == question for q in result.search_queries),
                    "empty_think_n": proto["empty_think"],
                    "observation_tag_n": proto["observation_tag"],
                    "extra_continue_n": proto["extra_continue"],
                    "generation_seed": seed,
                }
                tf.write(json.dumps(row, ensure_ascii=False) + "\n")
                tf.flush()
                group_rows.append(row)
                all_traj.append(row)
                print(
                    f"[{pi+1}/{len(samples)} r{ri}] {sample['sample_id']} "
                    f"route={row['route']} search={row['search_count']} "
                    f"R={row['R_total']:.3f} fin={row['finished']}",
                    flush=True,
                )
            rewards = [r["R_total"] for r in group_rows]
            advs = advantages(rewards)
            routes = {r["route"] for r in group_rows}
            queries = {q for r in group_rows for q in r["queries"]}
            answers = {r["answer"] for r in group_rows}
            keys = {
                (r["route"], tuple(r["queries"]), r["answer"]) for r in group_rows
            }
            n_search_traj = sum(1 for r in group_rows if r["search_count"] >= 1)
            g = {
                "sample_id": sample["sample_id"],
                "question": str(sample.get("question") or ""),
                "n": len(group_rows),
                "n_search_traj": n_search_traj,
                "reward_mean": round(float(statistics.mean(rewards)), 4),
                "reward_std": round(group_std(rewards), 4),
                "reward_std_nonzero": group_std(rewards) > EPS,
                "nonzero_advantage_n": sum(1 for a in advs if abs(a) > EPS),
                "route_diversity": len(routes) > 1,
                "query_diversity": len(queries) > 1,
                "unique_query_n": len(queries),
                "unique_trajectory_n": len(keys),
                "unique_answer_n": len(answers),
                "search_hist": dict(Counter(r["search_count"] for r in group_rows)),
                "has_internal_and_search": ("internal" in routes and "search" in routes),
                "searchable": n_search_traj >= 2,
            }
            gf.write(json.dumps(g, ensure_ascii=False) + "\n")
            gf.flush()
            groups.append(g)

    n_traj = len(all_traj)
    n_group = max(len(groups), 1)
    rewards = [r["R_total"] for r in all_traj]
    search_counts = [r["search_count"] for r in all_traj]
    searchable = [g for g in groups if g.get("searchable")]
    search_queries = [q for r in all_traj for q in r["queries"]]
    copy_n = sum(
        1
        for r in all_traj
        for q in r["queries"]
        if q == (r.get("question") or "")
    )
    forks = [g for g in searchable if g.get("query_diversity")]
    (out_dir / "query_forks.jsonl").write_text(
        "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in forks),
        encoding="utf-8",
    )
    summary: Dict[str, Any] = {
        "n_prompts": len(samples),
        "n_rollouts": args.n_rollouts,
        "n_expected": len(samples) * args.n_rollouts,
        "n_trajectories": n_traj,
        "eval_source": "train_smoke_128",
        "not_frozen_dev_200": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward": "R_answer + 0.5 R_evidence + 0.1 R_format",
        "search_cost_lambda": 0.0,
        "finish_rate": round(sum(1 for r in all_traj if r["finished"]) / max(n_traj, 1), 4),
        "parse_rate": round(sum(1 for r in all_traj if r["parse_ok"]) / max(n_traj, 1), 4),
        "reward_mean": round(float(statistics.mean(rewards)) if rewards else 0.0, 4),
        "reward_std": round(group_std(rewards), 4),
        "group_reward_std_nonzero_rate": round(
            sum(1 for g in groups if g["reward_std_nonzero"]) / n_group, 4
        ),
        "nonzero_advantage_rate": round(
            sum(g["nonzero_advantage_n"] for g in groups) / max(n_traj, 1), 4
        ),
        "unique_trajectory_rate": round(
            sum(g["unique_trajectory_n"] for g in groups) / max(n_traj, 1), 4
        ),
        "route_diversity_rate": round(
            sum(1 for g in groups if g["route_diversity"]) / n_group, 4
        ),
        "query_diversity_rate": round(
            sum(1 for g in groups if g["query_diversity"]) / n_group, 4
        ),
        "n_searchable_groups": len(searchable),
        "conditional_query_diversity_rate": round(
            (sum(1 for g in searchable if g["query_diversity"]) / len(searchable))
            if searchable
            else 0.0,
            4,
        ),
        "mean_unique_queries_per_searchable_group": round(
            (
                sum(int(g.get("unique_query_n") or 0) for g in searchable)
                / len(searchable)
            )
            if searchable
            else 0.0,
            4,
        ),
        "exact_question_copy_rate": round(copy_n / max(len(search_queries), 1), 4),
        "empty_think_n": sum(int(r.get("empty_think_n") or 0) for r in all_traj),
        "observation_tag_n": sum(int(r.get("observation_tag_n") or 0) for r in all_traj),
        "extra_continue_n": sum(int(r.get("extra_continue_n") or 0) for r in all_traj),
        "sample_ids": [s["sample_id"] for s in samples],
        "internal_and_search_rate": round(
            sum(1 for g in groups if g["has_internal_and_search"]) / n_group, 4
        ),
        "answer_correct_rate": round(sum(r["em"] for r in all_traj) / max(n_traj, 1), 4),
        "evidence_reward_mean": round(
            sum(r["R_evidence"] for r in all_traj) / max(n_traj, 1), 4
        ),
        "p_search_0": round(sum(1 for c in search_counts if c == 0) / max(n_traj, 1), 4),
        "p_search_1": round(sum(1 for c in search_counts if c == 1) / max(n_traj, 1), 4),
        "p_search_2": round(sum(1 for c in search_counts if c == 2) / max(n_traj, 1), 4),
        "duplicate_query_rate": round(
            sum(1 for r in all_traj if r["duplicate_query_count"] > 0) / max(n_traj, 1), 4
        ),
        "train_parquet": args.train_parquet,
        "model_path": args.model_path,
    }
    summary["gate"] = judge(summary)
    summary["trial_a_branch"] = trial_a_branch(summary)
    summary["trial_b_branch"] = trial_b_branch(summary)
    summary["trial_c_branch"] = trial_c_branch(summary)
    (out_dir / "gate35_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(summary["gate"], flush=True)
    print(summary["trial_a_branch"], flush=True)
    print(summary["trial_b_branch"], flush=True)
    print(summary["trial_c_branch"], flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""W4-0: score atomic SEARCH vs ANSWER on frozen behavior-dev states."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ATOMIC_PREFILL = "<internal>\nDecision:"
CANDIDATES = {"SEARCH": " SEARCH", "ANSWER": " ANSWER"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fixed-remaining-budget", type=int, default=None)
    return p.parse_args()


def chat_messages(example: dict[str, Any], fixed_remaining_budget: int | None = None) -> list[dict[str, str]]:
    out = [{"role": "system", "content": str(example["system"])}]
    roles = {"human": "user", "gpt": "assistant", "observation": "tool"}
    for item in example["conversations"][:-1]:
        content = str(item["value"])
        if item["from"] == "observation" and fixed_remaining_budget is not None:
            content = re.sub(r"Remaining Budget:\s*\d+", f"Remaining Budget: {fixed_remaining_budget}", content)
        out.append({"role": roles[item["from"]], "content": content})
    return out


def quantile(values: list[float], q: float) -> float:
    x = sorted(values)
    pos = (len(x) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return x[lo] if lo == hi else x[lo] * (hi - pos) + x[hi] * (pos - lo)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values), "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50), "p75": quantile(values, 0.75),
        "min": min(values), "max": max(values),
    }


def auroc(negative: list[float], positive: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def auprc(labels_scores: list[tuple[int, float]]) -> float:
    ranked = sorted(labels_scores, key=lambda x: x[1], reverse=True)
    positives = sum(y for y, _ in ranked)
    tp, precisions = 0, []
    for rank, (label, _) in enumerate(ranked, 1):
        if label:
            tp += 1
            precisions.append(tp / rank)
    return sum(precisions) / positives


def threshold_curve(d1: list[float], d2: list[float]) -> list[dict[str, float]]:
    unique = sorted(set(d1 + d2))
    thresholds = [unique[0] - 1e-6]
    thresholds += [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    thresholds += [unique[-1] + 1e-6]
    rows = []
    for tau in thresholds:
        stop = sum(x <= tau for x in d1) / len(d1)
        cont = sum(x > tau for x in d2) / len(d2)
        rows.append({
            "threshold": tau, "stop_at_d1": stop, "continue_at_d2": cont,
            "balanced_accuracy": (stop + cont) / 2,
            "original_gate_pass": stop >= 0.70 and cont >= 0.60,
        })
    return rows


def score_candidates(model: Any, tok: Any, prefixes: list[list[int]], candidate: list[int]) -> list[float]:
    sequences = [prefix + candidate for prefix in prefixes]
    max_len = max(map(len, sequences))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids = torch.full((len(sequences), max_len), pad, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    for i, seq in enumerate(sequences):
        input_ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
        attention[i, :len(seq)] = 1
    with torch.inference_mode():
        logp = torch.log_softmax(model(input_ids=input_ids, attention_mask=attention).logits.float(), dim=-1)
    scores = []
    for i, prefix in enumerate(prefixes):
        total = sum(logp[i, len(prefix) + j - 1, token_id].item() for j, token_id in enumerate(candidate))
        scores.append(total)
    return scores


def main() -> None:
    cfg = parse_args()
    rows = [json.loads(x) for x in (cfg.data_dir / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    cases = []
    for row in rows:
        depth = int(row["actual_depth"])
        kind = "post_obs_stop" if depth == 1 else "post_obs_continue"
        example = next(x for x in row["decision_examples"] if x["metadata"]["decision_type"] == kind)
        cases.append((row, example, depth))

    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    candidate_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in CANDIDATES.items()}
    details = []
    for start in range(0, len(cases), cfg.batch_size):
        batch = cases[start:start + cfg.batch_size]
        prefixes = []
        for _, example, _ in batch:
            try:
                prompt = tok.apply_chat_template(chat_messages(example, cfg.fixed_remaining_budget), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(chat_messages(example, cfg.fixed_remaining_budget), tokenize=False, add_generation_prompt=True)
            prefixes.append(tok.encode(prompt + ATOMIC_PREFILL, add_special_tokens=False))
        search_scores = score_candidates(model, tok, prefixes, candidate_ids["SEARCH"])
        answer_scores = score_candidates(model, tok, prefixes, candidate_ids["ANSWER"])
        for (row, _, depth), search, answer, prefix in zip(batch, search_scores, answer_scores, prefixes):
            details.append({
                "sample_id": row["sample_id"], "depth": depth,
                "label": "ANSWER" if depth == 1 else "SEARCH",
                "logp_search": search, "logp_answer": answer,
                "margin_search_minus_answer": search - answer, "prefix_tokens": len(prefix),
            })
    d1 = [x["margin_search_minus_answer"] for x in details if x["depth"] == 1]
    d2 = [x["margin_search_minus_answer"] for x in details if x["depth"] == 2]
    curve = threshold_curve(d1, d2)
    feasible = [x for x in curve if x["original_gate_pass"]]
    # When balanced accuracy ties, prefer recall on insufficient states: an
    # unnecessary search is cheaper than prematurely answering without evidence.
    best = max(curve, key=lambda x: (round(x["balanced_accuracy"], 12), x["continue_at_d2"]))
    summary = {
        "gate": "W4_0_THRESHOLD_GATE_PASS" if feasible else "W4_0_THRESHOLD_GATE_FAIL",
        "purpose": "atomic_decision_score_audit", "model_path": cfg.model_path,
        "data_dir": str(cfg.data_dir), "n_d1": len(d1), "n_d2": len(d2),
        "atomic_prefill": ATOMIC_PREFILL, "candidate_text": CANDIDATES,
        "candidate_token_ids": candidate_ids, "gold_known_missing_used": False,
        "fixed_remaining_budget": cfg.fixed_remaining_budget,
        "d1_margin": distribution(d1), "d2_margin": distribution(d2),
        "auroc": auroc(d1, d2),
        "auprc": auprc([(0, x) for x in d1] + [(1, x) for x in d2]),
        "best_balanced_threshold": best, "feasible_threshold_count": len(feasible),
        "feasible_thresholds": feasible,
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (cfg.output_dir / "threshold_curve.jsonl").open("w", encoding="utf-8") as f:
        for row in curve:
            f.write(json.dumps(row) + "\n")
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

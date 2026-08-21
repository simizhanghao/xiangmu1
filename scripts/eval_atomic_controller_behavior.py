#!/usr/bin/env python3
"""Evaluate forced atomic routing plus free Query/Answer generation on behavior-dev40."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_atomic_decision_scores import chat_messages
from build_web_multiturn_v2 import qnorm, query_conditioning

SEARCH = re.compile(r"<search>\s*(.*?)\s*</search>", re.I | re.S)
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
NEXT_QUERY = re.compile(r"Next Query\s*:\s*(.+?)(?:\n|</internal>)", re.I | re.S)
FORCED = {
    "SEARCH": "<internal>\nDecision: SEARCH\nNext Query:",
    "ANSWER": "<internal>\nDecision: ANSWER\n</internal>\n",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--score-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--search-max-new-tokens", type=int, default=192)
    p.add_argument("--answer-max-new-tokens", type=int, default=768)
    p.add_argument("--fixed-remaining-budget", type=int, default=4)
    return p.parse_args()


def base_prompt(tok: Any, example: dict[str, Any], fixed_remaining_budget: int) -> str:
    try:
        return tok.apply_chat_template(chat_messages(example, fixed_remaining_budget), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(chat_messages(example, fixed_remaining_budget), tokenize=False, add_generation_prompt=True)


def generate(model: Any, tok: Any, prompts: list[str], batch_size: int, max_new_tokens: int) -> list[str]:
    outputs = []
    tok.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        encoded = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        outputs.extend(tok.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True))
    return outputs


def main() -> None:
    cfg = parse_args()
    summary = json.loads((cfg.score_dir / "summary.json").read_text())
    if summary["gate"] != "W4_0_THRESHOLD_GATE_PASS":
        raise RuntimeError("atomic score audit has no feasible threshold")
    tau = float(summary["best_balanced_threshold"]["threshold"])
    scores = {x["sample_id"]: x for x in map(json.loads, (cfg.score_dir / "scores.jsonl").read_text().splitlines())}
    rows = [json.loads(x) for x in (cfg.data_dir / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    cases = []
    for row in rows:
        depth = int(row["actual_depth"])
        kind = "post_obs_stop" if depth == 1 else "post_obs_continue"
        example = next(x for x in row["decision_examples"] if x["metadata"]["decision_type"] == kind)
        margin = float(scores[row["sample_id"]]["margin_search_minus_answer"])
        route = "SEARCH" if margin > tau else "ANSWER"
        cases.append({"row": row, "example": example, "depth": depth, "margin": margin, "route": route})

    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    for route, cap in (("SEARCH", cfg.search_max_new_tokens), ("ANSWER", cfg.answer_max_new_tokens)):
        selected = [x for x in cases if x["route"] == route]
        prompts = [base_prompt(tok, x["example"], cfg.fixed_remaining_budget) + FORCED[route] for x in selected]
        completions = generate(model, tok, prompts, cfg.batch_size, cap)
        for case, continuation in zip(selected, completions):
            case["text"] = FORCED[route] + continuation

    details = []
    for case in cases:
        row, example, route, text = case["row"], case["example"], case["route"], case["text"]
        searches, answers = SEARCH.findall(text), ANSWER.findall(text)
        next_queries = NEXT_QUERY.findall(text)
        query = searches[0].strip() if searches else (next_queries[0].strip() if next_queries else "")
        previous = SEARCH.findall("\n".join(x["value"] for x in example["conversations"][:-1] if x["from"] == "gpt"))
        duplicate = bool(query and qnorm(query) in {qnorm(x) for x in previous})
        latest_obs = next((x["value"] for x in reversed(example["conversations"][:-1]) if x["from"] == "observation"), "")
        conditioned = bool(query and query_conditioning(query, row["question"], latest_obs, []) != "none")
        complete = bool(searches) if route == "SEARCH" else bool(answers)
        details.append({
            "sample_id": row["sample_id"], "depth": case["depth"], "margin": case["margin"],
            "threshold": tau, "route": route, "query": query, "duplicate": duplicate,
            "observation_conditioned": conditioned, "complete_action": complete, "text": text,
        })
    d1 = [x for x in details if x["depth"] == 1]
    d2 = [x for x in details if x["depth"] == 2]
    result = {
        "purpose": "forced_atomic_controller_behavior", "model_path": cfg.model_path,
        "threshold": tau, "n_d1": len(d1), "n_d2": len(d2),
        "fixed_remaining_budget": cfg.fixed_remaining_budget,
        "stop_at_d1": sum(x["route"] == "ANSWER" for x in d1) / len(d1),
        "continue_at_d2": sum(x["route"] == "SEARCH" for x in d2) / len(d2),
        "duplicate_q2_rate": sum(x["duplicate"] for x in d2) / len(d2),
        "obs_conditioned_q2_rate": sum(x["observation_conditioned"] for x in d2) / len(d2),
        "finish_rate": sum(x["complete_action"] for x in details) / len(details),
    }
    result["gate"] = "W4_BEHAVIOR_GATE_PASS" if (
        result["stop_at_d1"] >= 0.70 and result["continue_at_d2"] >= 0.60
        and result["duplicate_q2_rate"] <= 0.25 and result["obs_conditioned_q2_rate"] >= 0.60
        and result["finish_rate"] >= 0.95
    ) else "W4_BEHAVIOR_GATE_FAIL"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (cfg.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate STOP@D1 and CONTINUE/Q2@D2 on frozen graph-replay behavior states."""

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_web_multiturn_v2 import qnorm, query_conditioning

SEARCH = re.compile(r"<search>\s*(.*?)\s*</search>", re.I | re.S)
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
DECISION = re.compile(r"Decision\s*:\s*(SEARCH|ANSWER)", re.I)
NEXT_QUERY = re.compile(r"Next Query\s*:\s*(.+?)(?:\n|</internal>)", re.I | re.S)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=192)
    return p.parse_args()


def messages(example: dict[str, Any]) -> list[dict[str, str]]:
    out = [{"role": "system", "content": str(example["system"])}]
    role = {"human": "user", "gpt": "assistant", "observation": "tool"}
    for item in example["conversations"][:-1]:
        out.append({"role": role[item["from"]], "content": str(item["value"])})
    return out


def main() -> None:
    cfg = args()
    rows = [json.loads(x) for x in (cfg.data_dir / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    cases = []
    for row in rows:
        wanted = "post_obs_stop" if int(row["actual_depth"]) == 1 else "post_obs_continue"
        example = next(x for x in row["decision_examples"] if x["metadata"]["decision_type"] == wanted)
        cases.append((row, example, wanted))
    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    details = []
    for start in range(0, len(cases), cfg.batch_size):
        batch_cases = cases[start:start + cfg.batch_size]
        prompts = []
        for _, example, _ in batch_cases:
            try:
                prompt = tok.apply_chat_template(messages(example), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tok.apply_chat_template(messages(example), tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
        encoded = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **encoded, do_sample=False, max_new_tokens=cfg.max_new_tokens,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        prefix = encoded["input_ids"].shape[1]
        texts = tok.batch_decode(generated[:, prefix:], skip_special_tokens=True)
        for (row, example, wanted), text in zip(batch_cases, texts):
            searches, answers = SEARCH.findall(text), ANSWER.findall(text)
            decisions = DECISION.findall(text)
            # The behavior gate measures the first routing decision, not whether a
            # long evidence+answer payload fits inside the diagnostic token cap.
            prediction = (
                "SEARCH" if searches else "ANSWER" if answers else
                decisions[0].upper() if decisions else "INVALID"
            )
            previous = SEARCH.findall("\n".join(x["value"] for x in example["conversations"][:-1] if x["from"] == "gpt"))
            next_queries = NEXT_QUERY.findall(text)
            query = searches[0].strip() if searches else (next_queries[0].strip() if next_queries else "")
            duplicate = bool(query and qnorm(query) in {qnorm(x) for x in previous})
            latest_obs = next((x["value"] for x in reversed(example["conversations"][:-1]) if x["from"] == "observation"), "")
            conditioned = bool(query and query_conditioning(query, row["question"], latest_obs, []) != "none")
            decoded_tokens = len(tok.encode(text, add_special_tokens=False))
            truncated = prediction == "INVALID" and decoded_tokens >= cfg.max_new_tokens - 1
            details.append({
                "sample_id": row["sample_id"], "expected": "ANSWER" if wanted == "post_obs_stop" else "SEARCH",
                "prediction": prediction, "query": query, "duplicate": duplicate,
                "observation_conditioned": conditioned, "complete_action": prediction != "INVALID",
                "decoded_tokens": decoded_tokens, "truncated": truncated, "text": text,
            })
    d1 = [x for x in details if x["expected"] == "ANSWER"]
    d2 = [x for x in details if x["expected"] == "SEARCH"]
    summary = {
        "model_path": cfg.model_path, "cases": len(details),
        "stop_at_d1": sum(x["prediction"] == "ANSWER" for x in d1) / max(1, len(d1)),
        "continue_at_d2": sum(x["prediction"] == "SEARCH" for x in d2) / max(1, len(d2)),
        "duplicate_q2_rate": sum(x["duplicate"] for x in d2) / max(1, len(d2)),
        "obs_conditioned_q2_rate": sum(x["observation_conditioned"] for x in d2) / max(1, len(d2)),
        "finish_rate": sum(x["complete_action"] for x in details) / max(1, len(details)),
        "truncated_invalid_rate": sum(x["truncated"] for x in details) / max(1, len(details)),
        "diagnostic_max_new_tokens": cfg.max_new_tokens,
        "gates": {"stop_at_d1": 0.70, "continue_at_d2": 0.60, "duplicate_q2_max": 0.25, "obs_conditioned_q2": 0.60, "finish": 0.95},
    }
    summary["gate"] = "WEBMT_BEHAVIOR_PASS" if (
        summary["stop_at_d1"] >= 0.70 and summary["continue_at_d2"] >= 0.60
        and summary["duplicate_q2_rate"] <= 0.25 and summary["obs_conditioned_q2_rate"] >= 0.60
        and summary["finish_rate"] >= 0.95
    ) else "WEBMT_BEHAVIOR_FAIL"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

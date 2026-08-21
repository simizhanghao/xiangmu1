#!/usr/bin/env python3
"""W4-0b: repeat atomic score audit through the deployed vLLM endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

from audit_atomic_decision_scores import (
    ATOMIC_PREFILL, CANDIDATES, auprc, auroc, chat_messages, distribution, threshold_curve,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--served-model", required=True)
    p.add_argument("--fixed-remaining-budget", type=int, default=None)
    return p.parse_args()


def main() -> None:
    cfg = args()
    tok = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True, trust_remote_code=True)
    candidate_ids = {k: tok.encode(v, add_special_tokens=False) for k, v in CANDIDATES.items()}
    url = cfg.base_url.rstrip("/") + "/completions"

    def score(prompt: str, name: str) -> float:
        payload = {
            "model": cfg.served_model, "prompt": prompt + ATOMIC_PREFILL + CANDIDATES[name],
            "max_tokens": 1, "temperature": 0.0, "echo": True, "logprobs": 1,
        }
        req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        values = data["choices"][0]["logprobs"]["token_logprobs"]
        n = len(candidate_ids[name])
        chosen = values[-(n + 1):-1]
        if len(chosen) != n or any(x is None for x in chosen):
            raise RuntimeError(f"missing vLLM prompt logprobs for {name}: {chosen}")
        return float(sum(chosen))

    rows = [json.loads(x) for x in (cfg.data_dir / "trajectories.jsonl").read_text().splitlines() if x.strip()]
    details = []
    for row in rows:
        depth = int(row["actual_depth"])
        kind = "post_obs_stop" if depth == 1 else "post_obs_continue"
        example = next(x for x in row["decision_examples"] if x["metadata"]["decision_type"] == kind)
        try:
            prompt = tok.apply_chat_template(chat_messages(example, cfg.fixed_remaining_budget), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(chat_messages(example, cfg.fixed_remaining_budget), tokenize=False, add_generation_prompt=True)
        search, answer = score(prompt, "SEARCH"), score(prompt, "ANSWER")
        details.append({
            "sample_id": row["sample_id"], "depth": depth,
            "logp_search": search, "logp_answer": answer,
            "margin_search_minus_answer": search - answer,
        })
    d1 = [x["margin_search_minus_answer"] for x in details if x["depth"] == 1]
    d2 = [x["margin_search_minus_answer"] for x in details if x["depth"] == 2]
    curve = threshold_curve(d1, d2)
    feasible = [x for x in curve if x["original_gate_pass"]]
    best = max(curve, key=lambda x: (round(x["balanced_accuracy"], 12), x["continue_at_d2"]))
    summary = {
        "gate": "W4_0_VLLM_THRESHOLD_GATE_PASS" if feasible else "W4_0_VLLM_THRESHOLD_GATE_FAIL",
        "backend": "vllm_openai", "base_url": cfg.base_url, "served_model": cfg.served_model,
        "model_path": cfg.model_path, "n_d1": len(d1), "n_d2": len(d2),
        "candidate_token_ids": candidate_ids, "gold_known_missing_used": False,
        "fixed_remaining_budget": cfg.fixed_remaining_budget,
        "d1_margin": distribution(d1), "d2_margin": distribution(d2),
        "auroc": auroc(d1, d2), "auprc": auprc([(0, x) for x in d1] + [(1, x) for x in d2]),
        "best_balanced_threshold": best, "feasible_threshold_count": len(feasible),
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "scores.jsonl").open("w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")
    with (cfg.output_dir / "threshold_curve.jsonl").open("w") as f:
        for row in curve:
            f.write(json.dumps(row) + "\n")
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

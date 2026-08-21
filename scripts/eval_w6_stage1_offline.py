#!/usr/bin/env python3
"""One-shot frozen natural-dev500 Gate for W6 Stage-1 decision-only SFT."""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are a research sufficiency controller. Given the question, ResearchMemory, "
    "current observation, previous queries, sources, and remaining budget, decide only "
    "whether the available evidence is sufficient. Output exactly DECISION: STOP if it "
    "is sufficient; otherwise output exactly DECISION: CONTINUE. Do not output a query, "
    "missing fact, evidence, explanation, or answer."
)


def auc_rank(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    n1 = sum(labels)
    n0 = len(labels) - n1
    return (sum(r for r, y in zip(ranks, labels) if y) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="model/Qwen3-1.7B")
    parser.add_argument("--adapter", default="outputs/81_w6_stage1_decision_lora")
    parser.add_argument("--data", default="results/76_w5_controller_dataset/behavior_dev500.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("results/82_w6_stage1_offline"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise RuntimeError("frozen dev500 was already evaluated; refusing a rerun")

    rows = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    assert len(rows) == 500
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    stop_ids = tokenizer.encode(" STOP", add_special_tokens=False)
    continue_ids = tokenizer.encode(" CONTINUE", add_special_tokens=False)
    # Qwen3 tokenizes STOP as one token but CONTINUE as CONT + INUE.  The action
    # branches at the first label token, so AUROC uses that first-token margin;
    # generated text below still has to parse as the complete label.
    assert len(stop_ids) >= 1 and len(continue_ids) >= 1
    stop_id, continue_id = stop_ids[0], continue_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    marker_ids = tokenizer.encode("DECISION:", add_special_tokens=False)

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["controller_input"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    raw_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    sequences = [ids[: 4096 - len(marker_ids)] + marker_ids for ids in raw_ids]
    truncated = sum(len(ids) + len(marker_ids) > 4096 for ids in raw_ids)
    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_sequences = sequences[start : start + args.batch_size]
        encoded = tokenizer.pad(
            {"input_ids": batch_sequences}, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :].float()
            generated = model.generate(
                **encoded,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        scores = (logits[:, continue_id] - logits[:, stop_id]).cpu().tolist()
        for row, text, score in zip(batch_rows, texts, scores):
            match = re.search(r"\b(STOP|CONTINUE)\b", text, re.I)
            prediction = match.group(1).upper() if match else "INVALID"
            predictions.append(
                {
                    "state_id": row["state_id"],
                    "sample_id": row["sample_id"],
                    "gold_decision": row["decision"],
                    "prediction": prediction,
                    "continue_score": score,
                    "raw_completion_after_decision_prefix": text,
                }
            )
        print(f"[{len(predictions)}/500]", flush=True)

    stop = [row for row in predictions if row["gold_decision"] == "STOP"]
    cont = [row for row in predictions if row["gold_decision"] == "CONTINUE"]
    stop_recall = sum(row["prediction"] == "STOP" for row in stop) / len(stop)
    continue_recall = sum(row["prediction"] == "CONTINUE" for row in cont) / len(cont)
    balanced_accuracy = (stop_recall + continue_recall) / 2
    parse_valid = sum(row["prediction"] != "INVALID" for row in predictions) / len(predictions)
    labels = [int(row["gold_decision"] == "CONTINUE") for row in predictions]
    scores = [row["continue_score"] for row in predictions]
    passed = stop_recall >= 0.80 and continue_recall >= 0.80 and balanced_accuracy >= 0.80
    summary = {
        "gate": "W6_STAGE1_DECISION_GATE_PASS" if passed else "W6_STAGE1_DECISION_GATE_FAIL",
        "n": len(predictions),
        "gold_stop": len(stop),
        "gold_continue": len(cont),
        "auroc_diagnostic": auc_rank(labels, scores),
        "stop_recall": stop_recall,
        "continue_recall": continue_recall,
        "balanced_accuracy": balanced_accuracy,
        "parse_valid_rate": parse_valid,
        "truncated_prompts": truncated,
        "truncation_policy": "keep_source_prefix_then_append_decision_prefix",
        "hard_gate_uses_auroc": False,
        "score_definition": "first_divergent_label_token_logit_margin",
        "stop_label_token_ids": stop_ids,
        "continue_label_token_ids": continue_ids,
        "dev_evaluation_count": 1,
        "api_calls": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

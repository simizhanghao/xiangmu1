#!/usr/bin/env python3
"""One-shot frozen natural-dev500 Gate for the final W7 S2G Judge."""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You are an evidence sufficiency and gap judge. Given a question and a compact "
    "Evidence Context, output exactly SUFFICIENT: YES followed by GAPS: NONE when the "
    "evidence is sufficient. Otherwise output SUFFICIENT: NO followed by GAPS: and one "
    "concise missing fact. Do not produce a search query, explanation, or answer."
)
WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def words(text):
    return {token.lower() for token in WORD.findall(str(text)) if len(token) > 1}


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
    return (sum(rank for rank, label in zip(ranks, labels) if label) - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="model/Qwen3-1.7B")
    parser.add_argument("--adapter", default="outputs/84_w7_s2g_lora")
    parser.add_argument("--data", default="results/83_w7_s2g_dataset/dev500.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("results/85_w7_s2g_offline"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists():
        raise RuntimeError("frozen dev500 was already evaluated; refusing a rerun")

    rows = [json.loads(line) for line in Path(args.data).open() if line.strip()]
    assert len(rows) == 500
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    yes_ids = tokenizer.encode(" YES", add_special_tokens=False)
    no_ids = tokenizer.encode(" NO", add_special_tokens=False)
    assert len(yes_ids) == len(no_ids) == 1
    yes_id, no_id = yes_ids[0], no_ids[0]
    marker_ids = tokenizer.encode("SUFFICIENT:", add_special_tokens=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["compact_input"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in rows
    ]
    raw_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    assert max(len(ids) + len(marker_ids) for ids in raw_ids) <= 2048
    sequences = [ids + marker_ids for ids in raw_ids]
    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        encoded = tokenizer.pad(
            {"input_ids": sequences[start : start + args.batch_size]},
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :].float()
            generated = model.generate(
                **encoded,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        scores = (logits[:, no_id] - logits[:, yes_id]).cpu().tolist()
        for row, text, score in zip(batch_rows, texts, scores):
            decision_match = re.search(r"\b(YES|NO)\b", text, re.I)
            sufficient = decision_match.group(1).upper() if decision_match else "INVALID"
            prediction = "STOP" if sufficient == "YES" else "CONTINUE" if sufficient == "NO" else "INVALID"
            gap_match = re.search(r"GAPS?\s*:\s*(.*)", text, re.I | re.S)
            gap = gap_match.group(1).strip().lstrip("- ") if gap_match else ""
            gap_valid = prediction == "STOP" and gap.upper().startswith("NONE") or prediction == "CONTINUE" and bool(gap) and not gap.upper().startswith("NONE")
            target_words = words(row.get("gap_target", ""))
            gap_overlap = len(words(gap) & target_words) / max(1, len(target_words)) if prediction == "CONTINUE" else None
            predictions.append({
                "state_id": row["state_id"],
                "sample_id": row["sample_id"],
                "gold_decision": row["decision"],
                "prediction": prediction,
                "continue_score": score,
                "gap": gap,
                "gap_valid": bool(gap_valid),
                "gap_target_word_recall": gap_overlap,
                "raw_completion_after_sufficient_prefix": text,
            })
        print(f"[{len(predictions)}/500]", flush=True)

    stop = [row for row in predictions if row["gold_decision"] == "STOP"]
    cont = [row for row in predictions if row["gold_decision"] == "CONTINUE"]
    stop_recall = sum(row["prediction"] == "STOP" for row in stop) / len(stop)
    continue_recall = sum(row["prediction"] == "CONTINUE" for row in cont) / len(cont)
    balanced_accuracy = (stop_recall + continue_recall) / 2
    labels = [int(row["gold_decision"] == "CONTINUE") for row in predictions]
    scores = [row["continue_score"] for row in predictions]
    predicted_continue = [row for row in predictions if row["prediction"] == "CONTINUE"]
    passed = stop_recall >= 0.80 and continue_recall >= 0.80 and balanced_accuracy >= 0.80
    summary = {
        "gate": "W7_S2G_STATIC_GATE_PASS" if passed else "W7_S2G_STATIC_GATE_FAIL",
        "n": len(predictions),
        "gold_stop": len(stop),
        "gold_continue": len(cont),
        "auroc_diagnostic": auc_rank(labels, scores),
        "stop_recall": stop_recall,
        "continue_recall": continue_recall,
        "balanced_accuracy": balanced_accuracy,
        "parse_valid_rate": sum(row["prediction"] != "INVALID" for row in predictions) / len(predictions),
        "structured_gap_valid_rate": sum(row["gap_valid"] for row in predictions) / len(predictions),
        "predicted_continue": len(predicted_continue),
        "mean_gap_target_word_recall": sum(row["gap_target_word_recall"] or 0 for row in predicted_continue) / max(1, len(predicted_continue)),
        "max_prompt_tokens": max(len(ids) + len(marker_ids) for ids in raw_ids),
        "truncated_prompts": 0,
        "hard_gate_uses_auroc": False,
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

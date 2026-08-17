#!/usr/bin/env python3
"""Frozen single-pass controlled baselines: Direct, one-shot RAG, Oracle RAG."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.metrics import exact_match, token_f1
from src.eval.protocol import score_evidence_use
from src.sft.prototype_builder import (
    AGENT_SYSTEM_PROMPT,
    format_documents_for_user,
    load_jsonl,
    oracle_documents,
)
from src.tools.candidate_bm25 import retrieve_candidate_bm25


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["direct", "rag", "oracle"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--eval-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, int, int]:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(model.device) for key, value in batch.items()}
    prompt_tokens = int(batch["input_ids"].shape[-1])
    output = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0, prompt_tokens:]
    return tokenizer.decode(generated, skip_special_tokens=True), prompt_tokens, int(generated.numel())


def main() -> None:
    cfg = args()
    torch.manual_seed(cfg.seed)
    all_samples = load_jsonl(cfg.eval_file)
    end = cfg.offset + cfg.max_samples if cfg.max_samples > 0 else len(all_samples)
    samples = all_samples[cfg.offset : end]
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=cfg.exist_ok)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval()

    rows = []
    started = time.time()
    with (out / "metrics.jsonl").open("w") as handle:
        for index, sample in enumerate(samples, 1):
            retrieved = []
            if cfg.mode == "direct":
                messages = [
                    {
                        "role": "system",
                        "content": "Answer the question directly and concisely. Put only the final answer inside <answer>...</answer>.",
                    },
                    {"role": "user", "content": f"Question: {sample['question']}"},
                ]
            else:
                if cfg.mode == "rag":
                    retrieved = list(
                        retrieve_candidate_bm25(sample, sample["question"], top_k=cfg.top_k).get("documents") or []
                    )
                else:
                    retrieved = oracle_documents(sample)
                messages = [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {sample['question']}\n\nDocuments:\n"
                            f"{format_documents_for_user(retrieved)}\n\n"
                            "Select supporting sentences in <evidence>, reason briefly in <think>, then answer in <answer>."
                        ),
                    },
                ]

            t0 = time.time()
            text, prompt_tokens, generated_tokens = generate(model, tokenizer, messages, cfg.max_new_tokens)
            match = ANSWER_RE.search(text)
            prediction = match.group(1).strip() if match else text.strip()
            em = exact_match(prediction, sample.get("gold_answers") or [])
            gold_titles = {str(x["title"]) for x in sample.get("supporting_facts") or []}
            retrieved_titles = {str(x.get("title") or "") for x in retrieved}
            title_recall = len(gold_titles & retrieved_titles) / max(len(gold_titles), 1)
            evidence_f1 = 0.0
            if cfg.mode != "direct":
                try:
                    evidence_f1 = float(score_evidence_use(text, sample)["evidence_f1"])
                except Exception:
                    evidence_f1 = 0.0
            row = {
                "sample_id": sample["sample_id"],
                "mode": cfg.mode,
                "prediction": prediction,
                "generation": text,
                "exact_match": em,
                "direct_correct": em >= 1.0 - 1e-9,
                "token_f1": token_f1(prediction, sample.get("gold_answers") or []),
                "evidence_f1": evidence_f1,
                "format_valid": bool(match),
                "finish": bool(match),
                "search_count": 0 if cfg.mode == "direct" else 1,
                "retrieval_title_recall": title_recall,
                "retrieval_hit_all": float(bool(gold_titles) and gold_titles <= retrieved_titles),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "latency_ms": round((time.time() - t0) * 1000, 1),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(row)
            print(f"[{index}/{len(samples)}] mode={cfg.mode} EM={row['exact_match']} F1={row['token_f1']:.3f}", flush=True)

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / max(len(rows), 1)

    summary = {
        "purpose": "frozen_controlled_baseline",
        "mode": cfg.mode,
        "model_path": str(Path(cfg.model_path).resolve()),
        "eval_file": str(Path(cfg.eval_file).resolve()),
        "num_samples": len(rows),
        "mean_em": mean("exact_match"),
        "mean_token_f1": mean("token_f1"),
        "mean_evidence_f1": mean("evidence_f1"),
        "finish_rate": mean("finish"),
        "parse_ok_rate": mean("format_valid"),
        "mean_search_count": mean("search_count"),
        "retrieval_title_recall": mean("retrieval_title_recall"),
        "retrieval_hit_all_rate": mean("retrieval_hit_all"),
        "mean_generated_tokens": mean("generated_tokens"),
        "mean_latency_ms": mean("latency_ms"),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("CONTROLLED_BASELINE_PASS")


if __name__ == "__main__":
    main()

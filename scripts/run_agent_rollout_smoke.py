"""Phase 3A: Search Agent rollout smoke (SFT-v1 + Candidate-BM25, no training).

Example:
  CUDA_VISIBLE_DEVICES=4 python scripts/run_agent_rollout_smoke.py \
    --model-path outputs/00_sft_v1_merged \
    --eval-file data/eval/hotpotqa_200.jsonl \
    --max-samples 8 --top-k 5 --max-search-turns 2 \
    --run-tag phase3a_n8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.react_loop import (  # noqa: E402
    RolloutConfig,
    make_openai_completions_fn,
    make_vllm_generate_fn,
    run_search_agent_rollout,
)
from src.sft.prototype_builder import load_jsonl  # noqa: E402

DEFAULT_MODEL = str(REPO_ROOT / "outputs" / "00_sft_v1_merged")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3A agent rollout smoke.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--eval-file", type=str, default="data/eval/hotpotqa_200.jsonl")
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-search-turns", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default="phase3a")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=("hf", "vllm", "vllm_openai"),
    )
    p.add_argument("--vllm-base-url", type=str, default="http://127.0.0.1:18000/v1")
    p.add_argument("--vllm-model-name", type=str, default="sft8b")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=8192)
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = max(len(rows), 1)

    def mean(key: str) -> float:
        return sum(float(r["metrics"].get(key) or 0) for r in rows) / n

    n_fin = sum(1 for r in rows if r["finished"])
    n_search = sum(1 for r in rows if r["route_first"] == "search")
    n_internal = sum(1 for r in rows if r["route_first"] == "internal")
    n_hit_cap = sum(1 for r in rows if r.get("hit_max_search_turns"))
    obs_ok = sum(
        1
        for r in rows
        if any(s.get("step_type") == "observation" and s.get("loss_mask") is False for s in r["steps"])
        or r["metrics"].get("search_count", 0) == 0
    )
    counts = [int(r["metrics"].get("search_count") or 0) for r in rows]
    hist = {
        "p_search_0": round(sum(1 for c in counts if c == 0) / n, 4),
        "p_search_1": round(sum(1 for c in counts if c == 1) / n, 4),
        "p_search_2": round(sum(1 for c in counts if c == 2) / n, 4),
        "p_search_ge3": round(sum(1 for c in counts if c >= 3) / n, 4),
    }
    return {
        "num_samples": len(rows),
        "finish_rate": round(n_fin / n, 4),
        "parse_ok_rate": round(mean("format_valid"), 4),
        "mean_em": round(mean("exact_match"), 4),
        "mean_token_f1": round(mean("token_f1"), 4),
        "mean_evidence_f1": round(mean("evidence_f1"), 4),
        "mean_search_count": round(mean("search_count"), 4),
        "mean_duplicate_query_count": round(mean("duplicate_query_count"), 4),
        "internal_rate": round(n_internal / n, 4),
        "search_rate": round(n_search / n, 4),
        "max_search_turn_hit_rate": round(n_hit_cap / n, 4),
        "observation_mask_ok_rate": round(obs_ok / n, 4),
        "mean_latency_ms": round(
            sum(float((r.get("cost_info") or {}).get("latency_ms") or 0) for r in rows) / n,
            1,
        ),
        "mean_observation_tokens": round(
            sum(float((r.get("cost_info") or {}).get("observation_tokens") or 0) for r in rows)
            / n,
            1,
        ),
        "mean_generated_tokens": round(
            sum(float((r.get("cost_info") or {}).get("generated_tokens") or 0) for r in rows) / n,
            1,
        ),
        **hist,
    }


def main() -> None:
    args = parse_args()
    import random

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    eval_path = resolve(args.eval_file)
    samples = load_jsonl(str(eval_path))[: args.max_samples]
    if not samples:
        raise SystemExit(f"no samples in {eval_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(args.output_dir) / (
        f"agent_rollout_n{len(samples)}_{stamp}_{args.run_tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase3a] model={args.model_path} device={device} backend={args.backend}", flush=True)
    print(f"[phase3a] n={len(samples)} top_k={args.top_k} "
          f"max_search_turns={args.max_search_turns}", flush=True)
    print(f"[phase3a] run_dir={run_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = None
    generate_fn = None
    if args.backend == "vllm_openai":
        generate_fn = make_openai_completions_fn(args.vllm_base_url, args.vllm_model_name)
        print(f"[phase3a] vllm_openai {args.vllm_base_url} model={args.vllm_model_name}", flush=True)
    elif args.backend == "vllm":
        from vllm import LLM

        llm = LLM(
            model=args.model_path,
            tokenizer=args.model_path,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=True,
        )
        generate_fn = make_vllm_generate_fn(llm)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device).eval()

    cfg = RolloutConfig(
        top_k=args.top_k,
        max_search_turns=args.max_search_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    rows_out: List[Dict[str, Any]] = []
    t_all = time.time()
    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as tf, (
        run_dir / "metrics.jsonl"
    ).open("w", encoding="utf-8") as mf:
        for i, sample in enumerate(samples, 1):
            result = run_search_agent_rollout(
                sample, model, tokenizer, cfg, generate_fn=generate_fn
            )
            tr = result.trace.to_jsonl_dict()
            tf.write(json.dumps(tr, ensure_ascii=False) + "\n")
            tf.flush()
            row = {
                "sample_id": sample["sample_id"],
                "finished": result.finished,
                "route_first": result.route_first,
                "search_queries": result.search_queries,
                "hit_max_search_turns": bool(
                    (result.trace.metadata or {}).get("hit_max_search_turns")
                ),
                "metrics": result.metrics,
                "validation_errors": result.validation_errors[:8],
                "cost_info": tr.get("cost_info"),
                "steps": [
                    {
                        "step_type": s.step_type,
                        "loss_mask": s.loss_mask,
                        "content_preview": (s.content or "")[:160],
                    }
                    for s in result.trace.steps
                ],
                "raw_generations": result.raw_generations,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            rows_out.append(row)
            print(
                f"[{i}/{len(samples)}] {sample['sample_id']} "
                f"fin={result.finished} route={result.route_first} "
                f"search={result.metrics.get('search_count')} "
                f"EM={result.metrics.get('exact_match')} "
                f"err={result.validation_errors[:1]}",
                flush=True,
            )

    summary = aggregate(rows_out)
    summary.update(
        {
            "phase": "3A",
            "purpose": "search_agent_rollout_smoke",
            "backend": args.backend,
            "model_path": args.model_path,
            "git_commit": git_commit(),
            "eval_file": str(eval_path),
            "top_k": args.top_k,
            "max_search_turns": args.max_search_turns,
            "elapsed_seconds": round(time.time() - t_all, 2),
            "run_dir": str(run_dir),
            "gates_hint": {
                "finish_rate_target": ">=0.8",
                "observation_mask_ok_target": 1.0,
                "note": "EM secondary; health first",
            },
        }
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[phase3a] artifacts -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()

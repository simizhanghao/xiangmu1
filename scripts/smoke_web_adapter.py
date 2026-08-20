#!/usr/bin/env python3
"""Search/fetch smoke before connecting the frozen policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.candidate_bm25 import format_observation_text
from src.tools.web_adapter import WebAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Who developed the theory of relativity?")
    parser.add_argument(
        "--provider",
        choices=("duckduckgo", "brave", "brave_llm_context", "searxng"),
        default="duckduckgo",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--output", default="results/54_web_zero_shot/adapter_smoke.json")
    args = parser.parse_args()
    adapter = WebAdapter(
        provider=args.provider,
        cache_dir="results/54_web_zero_shot/cache",
        timeout_s=args.timeout,
        retries=args.retries,
        llm_context_tokens=args.context_tokens,
    )
    result = adapter.retrieve({}, args.query, args.top_k)
    result["observation"] = format_observation_text(result["documents"])
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not result["documents"]:
        raise SystemExit(f"WEB_ADAPTER_SMOKE_FAIL errors={result['errors'][:3]}")
    print(f"WEB_ADAPTER_SMOKE_PASS docs={len(result['documents'])} provider={args.provider}")
    print(json.dumps(result.get("timing") or {}, indent=2))
    print(f"artifact={target}")


if __name__ == "__main__":
    main()

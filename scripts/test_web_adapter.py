#!/usr/bin/env python3
"""Offline contracts for Web timing and Brave LLM Context normalization."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.web_adapter import WebAdapter  # noqa: E402


def main() -> None:
    context = WebAdapter(provider="brave_llm_context", cache_dir="/tmp/dee_web_context_test")
    context._brave_llm_context = lambda query, top_k: [  # type: ignore[method-assign]
        {
            "title": "Ada Example",
            "url": "https://example.org/ada",
            "snippets": ["Ada Example was born in Example City."],
            "source_metadata": {"hostname": "example.org"},
        }
    ]
    out = context.retrieve({}, "Where was Ada Example born?", 5)
    assert out["documents"][0]["metadata"]["url"] == "https://example.org/ada"
    assert out["timing"]["fetch_total_ms"] == 0.0
    assert out["timing"]["success_urls"] == 1

    bocha = WebAdapter(provider="bocha", cache_dir="/tmp/dee_web_bocha_test")
    bocha._bocha_context = lambda query, top_k: [  # type: ignore[method-assign]
        {
            "title": "Grace Example",
            "url": "https://example.org/grace",
            "snippets": ["Grace Example designed an early compiler."],
            "source_metadata": {"site_name": "Example"},
        }
    ]
    bocha_out = bocha.retrieve({}, "Who designed an early compiler?", 5)
    assert bocha_out["documents"][0]["metadata"]["source"] == "bocha"
    assert bocha_out["documents"][0]["metadata"]["url"] == "https://example.org/grace"
    assert bocha_out["timing"]["fetch_total_ms"] == 0.0

    failed = WebAdapter(provider="duckduckgo", cache_dir="/tmp/dee_web_failure_test")

    def timeout(query: str, count: int):
        raise requests.ReadTimeout("fixture timeout")

    failed._search = timeout  # type: ignore[method-assign]
    failure = failed.retrieve({}, "fixture", 5)
    assert not failure["retriever"]["search_ok"]
    assert failure["errors"][0]["stage"] == "search"
    assert "tool_total_ms" in failure["timing"]
    print("WEB_ADAPTER_CONTRACT_PASS")


if __name__ == "__main__":
    main()

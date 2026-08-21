#!/usr/bin/env python3
"""Interactive terminal client for the final research API."""
from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MOCK_RESULT = {
    "request_id": "offline-demo",
    "question": "Who directed Troll 2 and where was he born?",
    "answer": "Troll 2 was directed by Claudio Fragasso, who was born in Rome, Italy.",
    "evidence": [{
        "text": "Claudio Fragasso is an Italian film director born in Rome.",
        "source_ids": ["S1"],
    }],
    "sources": [{
        "id": "S1",
        "title": "Claudio Fragasso",
        "url": "https://en.wikipedia.org/wiki/Claudio_Fragasso",
    }],
    "search_queries": ["Troll 2 director birthplace"],
    "search_count": 1,
    "finished": True,
    "format_valid": True,
    "warnings": [],
    "latency_ms": 0.0,
    "policy": "GRPO@400 (cached mock)",
    "adaptive_controller": False,
    "memory": {"mode": "provenance_only", "injected_into_policy_prompt": False},
}


def ask(api_url: str, question: str, api_key: str = "") -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        api_url.rstrip("/") + "/v1/research",
        data=json.dumps({"question": question}).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach API at {api_url}: {exc}") from exc


def render(result: dict) -> None:
    for index, query in enumerate(result.get("search_queries") or [], 1):
        print(f"\n[Search {index}]\n{query}")
    evidence = result.get("evidence") or []
    if evidence:
        print("\n[Evidence]")
        for index, item in enumerate(evidence, 1):
            print(f"{index}. {item.get('text', '').strip()}")
    print("\n[Answer]")
    print(result.get("answer") or "(No final answer was produced.)")
    sources = result.get("sources") or []
    if sources:
        print("\n[Sources]")
        for index, source in enumerate(sources, 1):
            title = source.get("title") or source.get("id")
            print(f"{index}. {title}\n   {source.get('url') or '(URL unavailable)'}")
    warnings = result.get("warnings") or []
    print(
        f"\n[Run] searches={result.get('search_count', 0)} "
        f"finished={result.get('finished')} latency={result.get('latency_ms')}ms"
    )
    if warnings:
        print("[Warnings] " + "; ".join(warnings))


def render_verbose(result: dict) -> None:
    usage = result.get("usage") or {}
    memory = result.get("memory") or {}
    print(
        "[Trace] "
        f"policy={result.get('policy')} provider={result.get('web_provider', 'cached')} "
        f"tokens={usage or 'n/a'} memory={memory.get('mode', 'n/a')} "
        f"prompt_injection={memory.get('injected_into_policy_prompt', False)}"
    )
    for source in result.get("sources") or []:
        if source.get("snippet"):
            print(f"[Source {source.get('id')}] {source['snippet']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-Aware Web Search Agent CLI")
    parser.add_argument("--question", "-q", default="")
    parser.add_argument("--api-url", default=os.environ.get("DEE_API_URL", "http://127.0.0.1:8010"))
    parser.add_argument("--api-key", default=os.environ.get("DEE_API_KEY", ""))
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    parser.add_argument(
        "--mock", action="store_true",
        help="run a cached offline demo without model, GPU, Web, or API keys",
    )
    parser.add_argument("--verbose", action="store_true", help="show trace and token details")
    args = parser.parse_args()

    def run(question: str) -> None:
        result = dict(MOCK_RESULT) if args.mock else ask(args.api_url, question, args.api_key)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            render(result)
            if args.verbose:
                render_verbose(result)

    if args.question.strip():
        run(args.question.strip())
        return
    print("Evidence-Aware DeepResearch Agent. Type /exit to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not question:
            continue
        try:
            run(question)
        except Exception as exc:
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()

"""Thin production wrapper around the frozen GRPO@400 AgentLoop."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.agents.react_loop import (
    RolloutConfig,
    make_openai_completions_fn,
    protocol_stop_strings,
    run_search_agent_rollout,
)
from src.tools.web_adapter import WebAdapter

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppSettings:
    model_path: str = os.environ.get(
        "DEE_MODEL_PATH", str(ROOT / "results/44_hf_formal_grpo_step400/model_view")
    )
    vllm_url: str = os.environ.get("DEE_VLLM_URL", "http://127.0.0.1:18120/v1")
    vllm_model: str = os.environ.get("DEE_VLLM_MODEL", "sft8b")
    web_provider: str = os.environ.get("DEE_WEB_PROVIDER", "bocha")
    cache_dir: str = os.environ.get("DEE_WEB_CACHE", str(ROOT / "artifacts/final_web_cache"))
    top_k: int = int(os.environ.get("DEE_TOP_K", "5"))
    max_search_turns: int = int(os.environ.get("DEE_MAX_SEARCH_TURNS", "5"))
    max_new_tokens: int = int(os.environ.get("DEE_MAX_NEW_TOKENS", "512"))
    web_timeout: float = float(os.environ.get("DEE_WEB_TIMEOUT", "30"))
    web_retries: int = int(os.environ.get("DEE_WEB_RETRIES", "2"))
    web_context_tokens: int = int(os.environ.get("DEE_WEB_CONTEXT_TOKENS", "4096"))


class ResearchService:
    """One frozen policy + one Web adapter; no Controller or training code."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings()
        if not (Path(self.settings.model_path) / "config.json").is_file():
            raise FileNotFoundError(f"missing frozen model: {self.settings.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_path, local_files_only=True, trust_remote_code=True
        )
        self.generate_fn = make_openai_completions_fn(
            self.settings.vllm_url,
            self.settings.vllm_model,
            protocol_stop_strings("research"),
        )
        self.web = WebAdapter(
            provider=self.settings.web_provider,
            cache_dir=self.settings.cache_dir,
            timeout_s=self.settings.web_timeout,
            retries=self.settings.web_retries,
            llm_context_tokens=self.settings.web_context_tokens,
        )
        self.config = RolloutConfig(
            top_k=self.settings.top_k,
            max_search_turns=self.settings.max_search_turns,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=0.0,
            # Frozen W2 showed that injecting ResearchMemory changes the policy's
            # input distribution and hurts finish/F1.  Keep provenance outside the
            # prompt instead: tracked_retrieve records queries and source documents.
            memory_mode="none",
        )

    def ask(self, question: str, request_id: str | None = None) -> dict[str, Any]:
        question = " ".join(str(question).split()).strip()
        if not question:
            raise ValueError("question must be non-empty")
        request_id = request_id or str(uuid.uuid4())
        documents: dict[str, dict[str, Any]] = {}

        def tracked_retrieve(sample: dict[str, Any], query: str, top_k: int):
            packed = self.web.retrieve(sample, query, top_k)
            for doc in packed.get("documents") or []:
                documents.setdefault(str(doc.get("document_id")), doc)
            return packed

        started = time.perf_counter()
        sample = {
            "sample_id": request_id,
            "question": question,
            "gold_answers": [],
        }
        result = run_search_agent_rollout(
            sample,
            None,
            self.tokenizer,
            self.config,
            generate_fn=self.generate_fn,
            retrieve_fn=tracked_retrieve,
            retriever_scope="web",
        )
        evidence = [
            {
                "source_ids": list(step.document_ids),
                "text": step.content,
            }
            for step in result.trace.steps
            if step.step_type == "evidence"
        ]
        sources = []
        for doc in documents.values():
            metadata = doc.get("metadata") or {}
            sources.append(
                {
                    "id": str(doc.get("document_id") or ""),
                    "title": str(doc.get("title") or ""),
                    "url": str(metadata.get("url") or ""),
                    "snippet": str(doc.get("text") or "")[:500],
                    "rank": doc.get("rank"),
                }
            )
        sources.sort(key=lambda item: (item["rank"] or 9999, item["id"]))
        cost = result.trace.cost_info
        return {
            "request_id": request_id,
            "question": question,
            "answer": result.trace.final_answer,
            "evidence": evidence,
            "sources": sources,
            "search_queries": list(result.search_queries),
            "search_count": int(result.metrics.get("search_count") or 0),
            "finished": bool(result.finished),
            "format_valid": bool(result.metrics.get("format_valid")),
            "warnings": list(result.validation_errors),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "usage": {
                "prompt_tokens": cost.prompt_tokens,
                "generated_tokens": cost.generated_tokens,
                "observation_tokens": cost.observation_tokens,
            },
            "policy": "GRPO@400",
            "web_provider": self.settings.web_provider,
            "adaptive_controller": False,
            "memory": {
                "mode": "provenance_only",
                "injected_into_policy_prompt": False,
            },
            "trace": {
                "queries": list(result.search_queries),
                "search_count": int(result.metrics.get("search_count") or 0),
            },
        }

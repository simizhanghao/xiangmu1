"""Thin production wrapper around the frozen GRPO@400 AgentLoop."""
from __future__ import annotations

import os
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

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
    assistant_base_url: str = os.environ.get(
        "DEE_ASSISTANT_BASE_URL", os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com")
    )
    assistant_model: str = os.environ.get(
        "DEE_ASSISTANT_MODEL", os.environ.get("TEACHER_MODEL", "deepseek-v4-flash")
    )


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

    def _assistant_chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        key = os.environ.get("DEE_ASSISTANT_API_KEY") or os.environ.get("TEACHER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("hybrid mode requires DEE_ASSISTANT_API_KEY (or TEACHER_API_KEY)")
        payload: dict[str, Any] = {
            "model": self.settings.assistant_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1200,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(
            self.settings.assistant_base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 180),
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise RuntimeError("planner returned no JSON object")
        return json.loads(match.group(0))

    def ask_hybrid(self, question: str, request_id: str, history: list[dict[str, str]]) -> dict[str, Any]:
        started = time.perf_counter()
        history_text = "\n".join(
            f"{x.get('role', 'user')}: {str(x.get('content', ''))[:800]}" for x in history[-6:]
        ) or "(none)"
        plan_text = self._assistant_chat([
            {"role": "system", "content": (
                "You are a research query planner. Resolve follow-ups using conversation history. "
                "Return JSON only: {needs_search:boolean,direct_answer:string,queries:[string]}. "
                "Use 1-3 precise queries, preserving requested year/entity/metric; prefer official sources. "
                "For greetings or casual chat set needs_search=false and answer naturally."
            )},
            {"role": "user", "content": f"History:\n{history_text}\n\nCurrent request:\n{question}"},
        ], json_mode=True)
        plan = self._json_object(plan_text)
        queries = [" ".join(str(q).split()) for q in plan.get("queries", []) if str(q).strip()][:3]
        if not plan.get("needs_search", True):
            return {
                "request_id": request_id, "question": question,
                "answer": str(plan.get("direct_answer") or "你好！有什么需要研究的问题吗？"),
                "evidence": [], "sources": [], "search_queries": [], "search_count": 0,
                "finished": True, "answer_success": True, "warnings": [],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "policy": "Hybrid Assistant", "web_provider": self.settings.web_provider,
                "adaptive_controller": False, "trace": {"queries": [], "search_count": 0},
            }
        documents: dict[str, dict[str, Any]] = {}
        warnings: list[str] = []
        for query in queries:
            packed = self.web.retrieve({"sample_id": request_id}, query, self.settings.top_k)
            warnings.extend(str(x.get("error") or x) for x in (packed.get("errors") or []))
            for doc in packed.get("documents") or []:
                documents.setdefault(str(doc.get("document_id")), doc)
        sources = []
        context = []
        for index, doc in enumerate(list(documents.values())[:12], 1):
            meta = doc.get("metadata") or {}
            source = {
                "id": f"S{index}", "title": str(doc.get("title") or ""),
                "url": str(meta.get("url") or ""), "snippet": str(doc.get("text") or "")[:1200],
            }
            sources.append(source)
            context.append(f"[{source['id']}] {source['title']}\nURL: {source['url']}\n{source['snippet']}")
        if not sources:
            answer = "没有检索到足够证据，暂时无法可靠回答。请稍后重试或补充更具体的年份、对象和榜单。"
            success = False
            warnings.append("web search returned no usable evidence")
        else:
            answer = self._assistant_chat([
                {"role": "system", "content": (
                    "You are a careful Chinese research assistant. Answer the exact question using only the "
                    "provided sources. Preserve requested years and metrics. Cite factual claims as [S1]. "
                    "If evidence is insufficient or conflicting, say so explicitly; never guess a rank or number."
                )},
                {"role": "user", "content": f"Question:\n{question}\n\nSources:\n" + "\n\n".join(context)},
            ])
            success = bool(answer.strip())
        evidence = [{"source_ids": [s["id"]], "text": s["snippet"]} for s in sources]
        return {
            "request_id": request_id, "question": question, "answer": answer,
            "evidence": evidence, "sources": sources, "search_queries": queries,
            "search_count": len(queries), "finished": True, "answer_success": success,
            "format_valid": True, "warnings": warnings,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "policy": "Hybrid Assistant (DeepSeek planner/synthesizer + Bocha)",
            "web_provider": self.settings.web_provider, "adaptive_controller": False,
            "memory": {"mode": "conversation_and_provenance", "injected_into_grpo_prompt": False},
            "trace": {"queries": queries, "search_count": len(queries)},
        }

    def ask(self, question: str, request_id: str | None = None, *, mode: str = "hybrid", history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        question = " ".join(str(question).split()).strip()
        if not question:
            raise ValueError("question must be non-empty")
        request_id = request_id or str(uuid.uuid4())
        if mode == "hybrid":
            return self.ask_hybrid(question, request_id, history or [])
        if mode != "frozen":
            raise ValueError("mode must be hybrid or frozen")
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
        warnings = list(result.validation_errors)
        if float(result.metrics.get("empty_retrieval_count") or 0) > 0:
            warnings.append("web search returned no usable evidence")
        return {
            "request_id": request_id,
            "question": question,
            "answer": result.trace.final_answer,
            "evidence": evidence,
            "sources": sources,
            "search_queries": list(result.search_queries),
            "search_count": int(result.metrics.get("search_count") or 0),
            "finished": bool(result.finished),
            "answer_success": bool(result.trace.final_answer and result.trace.final_answer != "[no answer]"),
            "format_valid": bool(result.metrics.get("format_valid")),
            "warnings": warnings,
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

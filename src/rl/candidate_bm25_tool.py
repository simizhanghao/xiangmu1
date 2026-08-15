"""veRL BaseTool: Candidate-BM25 search bound to sample_id per trajectory.

Policy only emits query (via <search>…</search> / tool args).
sample_id is injected at create() from dataset tools_kwargs.create_kwargs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

import aiohttp

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class CandidateBM25Tool(BaseTool):
    """HTTP-backed Candidate-BM25 tool.

    Config keys:
      - retrieval_url: e.g. http://127.0.0.1:8001/retrieve
      - default_topk: int
      - timeout_s: float
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.retrieval_url = str(config.get("retrieval_url") or "http://127.0.0.1:8001/retrieve")
        self.default_topk = int(config.get("default_topk") or 5)
        self.timeout_s = float(config.get("timeout_s") or 30.0)
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        create_kwargs = create_kwargs or kwargs.get("create_kwargs") or {}
        if instance_id is None:
            instance_id = str(uuid4())
        sample_id = create_kwargs.get("sample_id")
        if not sample_id:
            raise ValueError("CandidateBM25Tool.create requires create_kwargs.sample_id")
        self._instances[instance_id] = {
            "sample_id": str(sample_id),
            "queries": [],
            "search_count": 0,
        }
        return instance_id, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances.get(instance_id)
        if state is None:
            return ToolResponse(text="Error: tool instance not created"), 0.0, {"error": "missing_instance"}

        sample_id = state["sample_id"]
        # Policy may only pass query; never trust model-supplied sample_id override.
        query = str(parameters.get("query") or "").strip()
        topk = int(parameters.get("topk") or parameters.get("top_k") or self.default_topk)

        if not query:
            return (
                ToolResponse(text="[no documents retrieved]"),
                0.0,
                {"error": "empty_query", "sample_id": sample_id},
            )

        payload = {"sample_id": sample_id, "query": query, "topk": topk}
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.retrieval_url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        return (
                            ToolResponse(text=f"[retrieval error {resp.status}] {text[:200]}"),
                            0.0,
                            {"error": f"http_{resp.status}", "sample_id": sample_id},
                        )
                    data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CandidateBM25Tool retrieve failed: %s", exc)
            return (
                ToolResponse(text=f"[retrieval error] {exc}"),
                0.0,
                {"error": str(exc), "sample_id": sample_id},
            )

        observation = str(data.get("observation") or "[no documents retrieved]")
        state["queries"].append(query)
        state["search_count"] += 1
        metrics = {
            "sample_id": sample_id,
            "query": query,
            "topk": topk,
            "num_docs": len(data.get("documents") or []),
            "documents": [
                {
                    "document_id": str(document.get("document_id") or ""),
                    "title": str(document.get("title") or ""),
                    "rank": int(document.get("rank") or index + 1),
                    "score": float(document.get("score") or 0.0),
                }
                for index, document in enumerate(data.get("documents") or [])
            ],
            "search_count": state["search_count"],
        }
        return ToolResponse(text=observation), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

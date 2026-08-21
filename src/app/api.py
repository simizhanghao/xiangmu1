"""FastAPI surface for the frozen Evidence-Aware Web Search Agent."""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from src.app.service import AppSettings, ResearchService, ROOT

app = FastAPI(
    title="Evidence-Aware Web Search Agent",
    version="1.0.0",
    description="Frozen GRPO@400 policy with Bocha Web search and grounded evidence output.",
)
_service: ResearchService | None = None
_lock = asyncio.Semaphore(int(os.environ.get("DEE_MAX_CONCURRENCY", "1")))


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    request_id: str | None = Field(default=None, max_length=128)


def get_service() -> ResearchService:
    global _service
    if _service is None:
        _service = ResearchService()
    return _service


def authorize(authorization: str | None) -> None:
    expected = os.environ.get("DEE_API_KEY", "")
    if not expected:
        return
    supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health() -> dict:
    settings = AppSettings()
    upstream = False
    try:
        with urlopen(settings.vllm_url.rstrip("/") + "/models", timeout=2) as response:
            upstream = response.status == 200
    except Exception:
        pass
    key_name = "BOCHA_API_KEY" if settings.web_provider == "bocha" else ""
    provider_ready = not key_name or bool(os.environ.get(key_name))
    return {
        "status": "ok" if upstream and provider_ready else "degraded",
        "vllm_ready": upstream,
        "web_provider": settings.web_provider,
        "web_provider_ready": provider_ready,
        "policy": "GRPO@400",
        "adaptive_controller": False,
        "memory_mode": "provenance_only",
        "policy_prompt_memory": False,
    }


@app.get("/v1/metrics")
def metrics() -> dict:
    path = ROOT / "config/final_metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=503, detail="final metrics artifact missing")
    return json.loads(path.read_text())


@app.post("/v1/research")
async def research(payload: ResearchRequest, authorization: str | None = Header(default=None)) -> dict:
    authorize(authorization)
    async with _lock:
        try:
            return await run_in_threadpool(
                lambda: get_service().ask(payload.question, payload.request_id)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"research backend failed: {exc}") from exc

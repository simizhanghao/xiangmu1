"""HTTP Candidate-BM25 retrieval server for veRL BaseTool / AgentLoop.

POST /retrieve
{
  "sample_id": "...",
  "query": "...",
  "topk": 5
}
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.rl.candidate_index import CandidateBM25Index

_INDEX: Optional[CandidateBM25Index] = None
_DEFAULT_TOPK = 5


def create_app_state(index_path: str, default_topk: int = 5) -> CandidateBM25Index:
    global _INDEX, _DEFAULT_TOPK
    _DEFAULT_TOPK = default_topk
    _INDEX = CandidateBM25Index.from_jsonl(index_path)
    return _INDEX


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # quieter
        if self.path in ("/health", "/retrieve"):
            return
        super().log_message(fmt, *args)

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(
                200,
                {
                    "ok": _INDEX is not None,
                    "num_samples": 0 if _INDEX is None else len(_INDEX),
                    "default_topk": _DEFAULT_TOPK,
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/retrieve":
            self._send(404, {"error": "not found"})
            return
        if _INDEX is None:
            self._send(503, {"error": "index not ready"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        sample_id = str(req.get("sample_id") or "").strip()
        query = str(req.get("query") or "").strip()
        topk = int(req.get("topk") or req.get("top_k") or _DEFAULT_TOPK)
        if not sample_id or not query:
            self._send(400, {"error": "sample_id and query required"})
            return
        if not _INDEX.has(sample_id):
            self._send(404, {"error": f"unknown sample_id: {sample_id}"})
            return
        packed = _INDEX.retrieve(sample_id, query, top_k=topk)
        self._send(
            200,
            {
                "sample_id": packed.get("sample_id") or sample_id,
                "query": packed.get("query") or query,
                "documents": packed.get("documents") or [],
                "observation": packed.get("observation") or "[no documents retrieved]",
                "retriever": packed.get("retriever") or {},
                "error": packed.get("error"),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-BM25 HTTP retrieval server")
    parser.add_argument("--index", required=True, help="contexts jsonl (sample_id + contexts)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    create_app_state(args.index, default_topk=args.topk)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[candidate-bm25] index={args.index} http://{args.host}:{args.port}/retrieve n={len(_INDEX or {})}")
    server.serve_forever()


if __name__ == "__main__":
    main()

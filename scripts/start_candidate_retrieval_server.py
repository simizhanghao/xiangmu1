#!/usr/bin/env python3
"""Start Candidate-BM25 HTTP server (host or container; --network=host)."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index",
        type=Path,
        default=REPO / "data/rl/train_smoke_128/contexts_index.jsonl",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    if not args.index.exists():
        raise SystemExit(
            f"Missing index {args.index}. Run: python scripts/build_grpo_smoke_dataset.py"
        )

    # Ensure repo importable.
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from src.rl.retrieval_server import create_app_state
    from src.rl.retrieval_server import Handler
    from http.server import ThreadingHTTPServer

    idx = create_app_state(str(args.index), default_topk=args.topk)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[candidate-bm25] index={args.index} http://{args.host}:{args.port}/retrieve n={len(idx)}")
    server.serve_forever()


if __name__ == "__main__":
    main()

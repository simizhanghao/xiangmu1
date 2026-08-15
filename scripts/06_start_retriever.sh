#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_file "$BM25_INDEX"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" scripts/start_candidate_retrieval_server.py \
  --index "$BM25_INDEX" \
  --port "$RETRIEVER_PORT"

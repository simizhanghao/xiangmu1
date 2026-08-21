#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PYTHON_BIN:-/data1/hcc/eca-verl-vexact/.venv/bin/python}
cd "$ROOT"
export PYTHONPATH="$ROOT"

run_py() { env -u LD_LIBRARY_PATH "$PY" "$@"; }

run_py -m py_compile cli.py src/app/api.py src/app/service.py
echo "API_IMPORT_PASS"

mock_json=$(run_py cli.py --mock --question "offline smoke" --json)
MOCK_JSON="$mock_json" run_py - <<'PY'
import json, os
x=json.loads(os.environ["MOCK_JSON"])
assert x["finished"] and x["answer"] and x["evidence"] and x["sources"]
assert x["adaptive_controller"] is False
assert x["memory"]["injected_into_policy_prompt"] is False
print("MOCK_WEB_PASS")
print("EVIDENCE_PARSE_PASS")
print("SOURCES_PARSE_PASS")
print("CLI_PASS")
PY

run_py - <<'PY'
from fastapi.testclient import TestClient
import src.app.api as api

class FakeService:
    def ask(self, question, request_id=None):
        return {
            "question": question, "answer": "offline", "evidence": [{"source_ids": ["S1"], "text": "cached"}],
            "sources": [{"id": "S1", "title": "cached", "url": "https://example.com"}],
            "trace": {"queries": ["cached query"], "search_count": 1},
            "adaptive_controller": False,
            "memory": {"mode": "provenance_only", "injected_into_policy_prompt": False},
        }

api._service = FakeService()
c=TestClient(api.app)
r=c.get('/v1/metrics')
assert r.status_code == 200 and r.json()['best_policy'] == 'GRPO@400'
r=c.post('/v1/research', json={'question': 'offline API smoke'})
assert r.status_code == 200 and r.json()['answer'] == 'offline'
print('FASTAPI_PASS')
PY

echo "FINAL_STACK_SMOKE_PASS"

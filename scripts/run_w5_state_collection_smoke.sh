#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=${PYTHON_BIN:-/data1/hcc/eca-verl-vexact/.venv/bin/python}
BASE_URL=${W5_VLLM_URL:-http://127.0.0.1:18120/v1}
MODEL=${W5_EXECUTOR_MODEL:-$ROOT/results/44_hf_formal_grpo_step400/model_view}
N=${W5_SMOKE_N:-8}

[[ -n "${BOCHA_API_KEY:-}" ]] || { echo "W5_BLOCKED missing BOCHA_API_KEY"; exit 2; }
curl -fsS --max-time 5 "$BASE_URL/models" >/dev/null
cd "$ROOT"
env -u LD_LIBRARY_PATH "$PY" scripts/prepare_w5_controller_split.py
env -u LD_LIBRARY_PATH "$PY" scripts/run_agent_rollout_smoke.py \
  --backend vllm_openai \
  --vllm-base-url "$BASE_URL" \
  --vllm-model-name sft8b \
  --model-path "$MODEL" \
  --eval-file data/w5_controller/controller_train4500.jsonl \
  --output-dir results/71_w5_controller/state_collection_smoke \
  --max-samples "$N" \
  --top-k 5 \
  --max-search-turns 1 \
  --max-new-tokens 512 \
  --retriever-scope web \
  --web-provider bocha \
  --web-timeout 30 \
  --web-retries 2 \
  --web-context-tokens 4096 \
  --memory-mode research_v2 \
  --collect-post-observation-only \
  --run-tag "w5_natural_state_smoke_n${N}"
echo W5_STATE_COLLECTION_SMOKE_PASS

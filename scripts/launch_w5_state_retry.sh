#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
BASE_URL=${W5_VLLM_URL:-http://127.0.0.1:18120/v1}
MODEL=${W5_EXECUTOR_MODEL:-$ROOT/results/44_hf_formal_grpo_step400/model_view}
ATTEMPT=${W5_RETRY_ATTEMPT:-1}

[[ -n "${BOCHA_API_KEY:-}" ]] || { echo "W5_BLOCKED missing BOCHA_API_KEY"; exit 2; }
cd "$ROOT"
env -u LD_LIBRARY_PATH "$PY" scripts/prepare_w5_retry_questions.py
COUNT=$(wc -l < data/w5_controller/controller_pending_retry.jsonl)
if (( COUNT == 0 )); then
  echo W5_RETRY_NOT_NEEDED
  exit 0
fi

env -u LD_LIBRARY_PATH "$PY" scripts/run_agent_rollout_smoke.py \
  --backend vllm_openai --vllm-base-url "$BASE_URL" --vllm-model-name sft8b \
  --model-path "$MODEL" \
  --eval-file data/w5_controller/controller_pending_retry.jsonl \
  --output-dir "results/71_w5_controller/retry_shards/attempt${ATTEMPT}" \
  --max-samples "$COUNT" --top-k 5 --max-search-turns 4 --max-new-tokens 256 \
  --retriever-scope web --web-provider bocha --web-timeout 30 --web-retries 4 \
  --web-context-tokens 4096 --memory-mode research_v2 \
  --collect-post-observation-only --run-tag "w5_retry_attempt${ATTEMPT}"

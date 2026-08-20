#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PYTHON_BIN=${PYTHON_BIN:-/data1/hcc/eca-verl-vexact/.venv/bin/python}
PROVIDER=${WEB_PROVIDER:-brave_llm_context}
N=${WEB_N:-8}
TIMEOUT=${WEB_TIMEOUT:-45}
RETRIES=${WEB_RETRIES:-3}
MODEL=${WEB_MODEL:-$ROOT/results/44_hf_formal_grpo_step400/model_view}
BASE_URL=${WEB_VLLM_URL:-http://127.0.0.1:18120/v1}
EVAL_FILE=${WEB_EVAL_FILE:-$ROOT/data/eval/hotpotqa_200.jsonl}
OUT=$ROOT/results/54_web_zero_shot

cd "$ROOT"
mkdir -p "$OUT"

if [[ "$PROVIDER" == brave* && -z "${BRAVE_SEARCH_API_KEY:-}" ]]; then
  echo "WEB_PROVIDER_BLOCKED missing BRAVE_SEARCH_API_KEY"
  exit 2
fi
if [[ "$PROVIDER" == searxng && -z "${SEARXNG_URL:-}" ]]; then
  echo "WEB_PROVIDER_BLOCKED missing SEARXNG_URL"
  exit 2
fi
curl -fsS --max-time 5 "$BASE_URL/models" >/dev/null

env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/smoke_web_adapter.py \
  --provider "$PROVIDER" --top-k 5 --timeout "$TIMEOUT" --retries "$RETRIES" \
  --output "$OUT/provider_smoke.json"

for memory in none research; do
  echo "WEB_PAIR_START memory=$memory n=$N provider=$PROVIDER"
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_agent_rollout_smoke.py \
    --backend vllm_openai \
    --vllm-base-url "$BASE_URL" \
    --vllm-model-name sft8b \
    --model-path "$MODEL" \
    --eval-file "$EVAL_FILE" \
    --output-dir "$OUT/$memory" \
    --max-samples "$N" \
    --top-k 5 \
    --max-search-turns 5 \
    --max-new-tokens 512 \
    --retriever-scope web \
    --web-provider "$PROVIDER" \
    --web-timeout "$TIMEOUT" \
    --web-retries "$RETRIES" \
    --web-context-tokens 4096 \
    --memory-mode "$memory" \
    --run-tag "web_${PROVIDER}_${memory}_n${N}"
done

echo "WEB_ZERO_SHOT_PAIR_PASS output=$OUT"

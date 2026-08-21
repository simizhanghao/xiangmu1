#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
QUESTIONS=$ROOT/data/w5_controller/controller_pending_resume.jsonl
SHARDS=${W5_RESUME_SHARDS:-2}
BASE_URL=${W5_VLLM_URL:-http://127.0.0.1:18120/v1}
MODEL=${W5_EXECUTOR_MODEL:-$ROOT/results/44_hf_formal_grpo_step400/model_view}

[[ -n "${BOCHA_API_KEY:-}" ]] || { echo "W5_BLOCKED missing BOCHA_API_KEY"; exit 2; }
curl -fsS --max-time 5 "$BASE_URL/models" >/dev/null
cd "$ROOT"
env -u LD_LIBRARY_PATH /data1/hcc/eca-verl-vexact/.venv/bin/python scripts/prepare_w5_resume_questions.py
TOTAL=$(wc -l < "$QUESTIONS")
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))

for (( shard=0; shard<SHARDS; shard++ )); do
  offset=$(( shard * PER ))
  remaining=$(( TOTAL - offset ))
  (( remaining > 0 )) || break
  count=$PER
  (( remaining < count )) && count=$remaining
  session="q8_w5_resume_${shard}"
  tmux kill-session -t "$session" 2>/dev/null || true
  log="$ROOT/logs/w5_resume_shard${shard}.log"
  tmux new-session -d -s "$session" "cd '$ROOT'; set -o pipefail; env -u LD_LIBRARY_PATH /data1/hcc/eca-verl-vexact/.venv/bin/python scripts/run_agent_rollout_smoke.py --backend vllm_openai --vllm-base-url '$BASE_URL' --vllm-model-name sft8b --model-path '$MODEL' --eval-file '$QUESTIONS' --output-dir results/71_w5_controller/resume_shards/shard${shard} --sample-offset $offset --max-samples $count --top-k 5 --max-search-turns 4 --max-new-tokens 256 --retriever-scope web --web-provider bocha --web-timeout 30 --web-retries 4 --web-context-tokens 4096 --memory-mode research_v2 --collect-post-observation-only --run-tag w5_resume_shard${shard} 2>&1 | tee '$log'; rc=\${PIPESTATUS[0]}; echo W5_RESUME_SHARD_EXIT=\$rc; exec bash"
  echo "W5_RESUME_STARTED shard=$shard offset=$offset count=$count session=$session"
done

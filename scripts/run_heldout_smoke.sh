#!/usr/bin/env bash
# Sealed-test pipeline smoke: GRPO@400, n=8, vLLM det, Harness v1.
# No training. Does not pick a checkpoint. Leaves vLLM up on success.
set -euo pipefail
PROJECT=/data1/hcc/deepresearch/Dee
CONFIG="${CONFIG:-$PROJECT/config/harness_v1.json}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/results/51_heldout_test/smoke_n8_grpo400}"
DEBUG="${DEBUG:-0}"
SKIP_VLLM="${SKIP_VLLM:-0}"
RUN_TAG="${RUN_TAG:-heldout_smoke_n8_grpo400}"
GPU="${GPU:-0}"
PORT="${PORT:-18120}"
NAME=dee-vllm-heldout
MODEL="$PROJECT/results/44_hf_formal_grpo_step400/model_view"
EVAL="${EVAL:-$PROJECT/data/sealed/hotpotqa_test500.jsonl}"
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python

mkdir -p "$OUTPUT_DIR/logs"
cd "$PROJECT"

[[ -f "$MODEL/config.json" ]] || { echo "MISSING_MODEL $MODEL"; exit 2; }
[[ -f "$EVAL" ]] || { echo "MISSING_EVAL $EVAL"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "MISSING_CONFIG $CONFIG"; exit 2; }

curl -sf http://127.0.0.1:8001/health >/dev/null && echo RETRIEVER_OK || {
  echo RETRIEVER_DOWN
  exit 2
}

if [[ "$SKIP_VLLM" == 1 ]]; then
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && echo VLLM_REUSED || {
    echo VLLM_NOT_UP
    exit 2
  }
else
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "${used:-0}" -gt 1024 ]]; then
    echo "GPU${GPU}_BUSY used=${used}MiB"
    nvidia-smi -i "$GPU"
    exit 2
  fi

  if ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
    echo "PORT_${PORT}_BUSY"
    exit 2
  fi

  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --name "$NAME" \
    --gpus "device=${GPU}" \
    --ipc=host --network host --ulimit memlock=-1 \
    -v /data1/hcc/deepresearch:/data1/hcc/deepresearch \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    vllm/vllm-openai:latest \
    "$MODEL" \
    --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --port "$PORT" \
    --served-model-name sft8b --trust-remote-code \
    >/dev/null

  ok=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
      echo VLLM_READY
      ok=1
      break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
      echo VLLM_CONTAINER_DEAD
      docker logs --tail 80 "$NAME" || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ok" != 1 ]]; then
    echo VLLM_SERVE_FAIL
    docker logs --tail 80 "$NAME" || true
    docker rm -f "$NAME" || true
    exit 1
  fi
fi

export PYTHONPATH=/data1/hcc/deepresearch/Dee
debug_flag=()
[[ "$DEBUG" == 1 ]] && debug_flag+=(--debug)

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= \
  "$PY" "$PROJECT/scripts/run_agent_rollout_smoke.py" \
  --config "$CONFIG" \
  --seed "$SEED" \
  --model-path "$MODEL" \
  --eval-file "$EVAL" \
  --max-samples "$MAX_SAMPLES" \
  --top-k 5 --max-search-turns 2 --temperature 0.0 --max-new-tokens 512 \
  --backend vllm_openai \
  --vllm-base-url "http://127.0.0.1:${PORT}/v1" \
  --vllm-model-name sft8b \
  --output-dir "$OUTPUT_DIR" \
  --run-tag "$RUN_TAG" \
  "${debug_flag[@]}"

SUM=$(find "$OUTPUT_DIR" -name summary.json | sort | tail -n 1)
python3 - <<PY
import json
from pathlib import Path
p = Path("$SUM")
s = json.loads(p.read_text())
print("SUMMARY", p)
for k in ["num_samples","backend","finish_rate","parse_ok_rate","mean_token_f1","mean_em","mean_evidence_f1","search_rate","mean_generated_tokens"]:
    print(f"{k}={s.get(k)}")
n = int("$MAX_SAMPLES")
ok = s.get("num_samples")==n and s.get("backend")=="vllm_openai" and float(s.get("finish_rate") or 0)>=0.75
gate = "HELDOUT_SMOKE_PASS" if n <= 8 else "HELDOUT_GRPO400_DONE"
print(gate if ok else ("HELDOUT_SMOKE_FAIL" if n <= 8 else "HELDOUT_GRPO400_FAIL"))
PY
echo "VLLM_LEFT_UP name=$NAME port=$PORT"

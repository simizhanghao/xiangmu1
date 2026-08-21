#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
MODEL=${DEE_MODEL_PATH:-$ROOT/results/44_hf_formal_grpo_step400/model_view}
VLLM_PORT=${VLLM_PORT:-18120}
API_PORT=${API_PORT:-8010}
API_HOST=${API_HOST:-127.0.0.1}
GPU=${GPU:-1}
CONTAINER=${VLLM_CONTAINER:-dee-final-vllm}
PY=${PYTHON_BIN:-/data1/hcc/eca-verl-vexact/.venv/bin/python}

test -s "$MODEL/config.json" || { echo "MISSING_MODEL $MODEL"; exit 2; }
[[ -n "${BOCHA_API_KEY:-}" ]] || {
  echo "ERROR: BOCHA_API_KEY is not set; live Web calls are disabled."
  echo "Use: export BOCHA_API_KEY=..."
  echo "For an offline demo: python3 cli.py --mock"
  exit 2
}
if [[ -z "${DEE_ASSISTANT_API_KEY:-${TEACHER_API_KEY:-${DEEPSEEK_API_KEY:-}}}" ]]; then
  echo "WARNING: no DeepSeek key; hybrid mode unavailable (frozen mode still works)."
fi
if [[ "$API_HOST" != "127.0.0.1" && "$API_HOST" != "localhost" && -z "${DEE_API_KEY:-}" ]]; then
  echo "REFUSING_PUBLIC_API_WITHOUT_DEE_API_KEY"
  exit 2
fi

if curl -sf --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
  echo "VLLM_REUSED port=$VLLM_PORT"
else
  used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "${used:-0}" -lt 2048 ]] || { echo "GPU_BUSY gpu=$GPU used_mib=$used"; exit 2; }
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -d --name "$CONTAINER" \
    --gpus "device=${GPU}" --ipc=host --network host --ulimit memlock=-1 \
    -v /data1/hcc/deepresearch:/data1/hcc/deepresearch \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    vllm/vllm-openai:latest "$MODEL" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --port "$VLLM_PORT" --served-model-name sft8b --trust-remote-code >/dev/null
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null && break
    docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || {
      docker logs --tail 100 "$CONTAINER"; exit 1;
    }
    sleep 2
  done
  curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null || {
    echo "VLLM_START_TIMEOUT"; exit 1;
  }
  echo "VLLM_READY port=$VLLM_PORT gpu=$GPU"
fi

cd "$ROOT"
export PYTHONPATH="$ROOT"
export DEE_VLLM_URL="http://127.0.0.1:${VLLM_PORT}/v1"
exec env -u LD_LIBRARY_PATH "$PY" -m uvicorn src.app.api:app \
  --host "$API_HOST" --port "$API_PORT" --workers 1

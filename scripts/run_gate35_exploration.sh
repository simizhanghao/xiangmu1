#!/usr/bin/env bash
# Detached vLLM + Gate 3.5 / 3.5B audit. Only GPU 0 and dee-vllm-n8 / $EXP_NAME.
set -u
REPO=/data1/hcc/deepresearch
OUT="${OUT:-$REPO/Dee/results/29_gate35b_trial_a_16x8}"
MODEL="$REPO/Dee/outputs/22_sft_qwen3_8b_merged"
GPU=0
MEM=0.9
PORT=18000
TEMP="${TEMP:-0.9}"
TOP_P="${TOP_P:-0.95}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
EXP_NAME="${EXP_NAME:-dee-g35a-exp}"

mkdir -p "$OUT/logs"

used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "${used:-0}" -gt 1024 ]; then
  echo "GPU${GPU}_BUSY used=${used}MiB — refuse to start. Do not kill other GPUs."
  nvidia-smi -i "$GPU"
  exit 2
fi

docker rm -f dee-vllm-n8 "$EXP_NAME" 2>/dev/null || true

echo "[gate35-launch] GPU=${GPU} mem=${MEM} T=${TEMP} top_p=${TOP_P} n=${MAX_SAMPLES}x${N_ROLLOUTS}"
docker run -d --name dee-vllm-n8 \
  --gpus "device=${GPU}" \
  --ipc=host --network host --ulimit memlock=-1 \
  -v "$REPO:$REPO" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  vllm/vllm-openai:latest \
  "$MODEL" \
  --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "$MEM" --port "$PORT" \
  --served-model-name sft8b --trust-remote-code \
  >/dev/null

ok=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo VLLM_READY
    ok=1
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx dee-vllm-n8; then
    echo VLLM_CONTAINER_DEAD
    docker logs --tail 80 dee-vllm-n8 || true
    exit 1
  fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo VLLM_SERVE_FAIL
  docker logs --tail 80 dee-vllm-n8 || true
  docker rm -f dee-vllm-n8 || true
  exit 1
fi

docker logs dee-vllm-n8 >"$OUT/logs/vllm_serve.log" 2>&1 || true

exp_ec=0
docker run --rm --network host --name "$EXP_NAME" \
  -v "$REPO:$REPO" \
  -w "$REPO" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH="$REPO" \
  hiyouga/llamafactory:latest \
  bash -lc "pip install -q bm25s && python scripts/audit_gate35_exploration.py \
    --config $REPO/Dee/config/harness_v1.json \
    --seed 42 --debug --max-samples $MAX_SAMPLES --n-rollouts $N_ROLLOUTS \
    --train-parquet $REPO/data/rl/train_smoke_128/train.parquet \
    --contexts-index $REPO/data/rl/train_smoke_128/contexts_index.jsonl \
    --model-path $MODEL \
    --vllm-base-url http://127.0.0.1:${PORT}/v1 --vllm-model-name sft8b \
    --temperature $TEMP --top-p $TOP_P --top-k 5 --max-search-turns 2 \
    --output-dir $OUT" \
  >"$OUT/logs/gate35.log" 2>&1 || exp_ec=$?

echo "EXP_EXIT:${exp_ec}"
docker logs dee-vllm-n8 >"$OUT/logs/vllm_serve.log" 2>&1 || true
docker run --rm -v "$REPO:$REPO" hiyouga/llamafactory:latest \
  bash -lc "chown -R $(id -u):$(id -g) $OUT" || true
if [ -f "$OUT/gate35_summary.json" ]; then
  cat "$OUT/gate35_summary.json"
else
  echo SUMMARY_MISSING
fi
docker rm -f dee-vllm-n8 "$EXP_NAME" 2>/dev/null || true
echo GATE35_TMUX_DONE
exit "$exp_ec"

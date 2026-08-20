#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
EVAL="$PROJECT/data/sealed/hotpotqa_test500.jsonl"
CONFIG="$PROJECT/config/harness_v1.json"
BASE="$PROJECT/model"
SFT="$PROJECT/outputs/22_sft_qwen3_8b_merged"
ROOT="$PROJECT/results/53_heldout_four_arm"
arm=${1:?usage: $0 <direct|rag|sft> [gpu]}
gpu=${2:-1}

mkdir -p "$ROOT" "$PROJECT/logs"
cd "$PROJECT"
curl -fsS http://127.0.0.1:8001/health >/dev/null

case "$arm" in
  direct|rag)
    out="$ROOT/base_${arm}"
    [[ ! -e "$out/summary.json" ]] || { echo "ALREADY_DONE $out/summary.json"; exit 0; }
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" scripts/run_controlled_baseline.py \
        --mode "$arm" --model-path "$BASE" --eval-file "$EVAL" \
        --output-dir "$out" --max-samples 500 --top-k 5 \
        --max-new-tokens 512 --seed 42
    ;;
  sft)
    out="$ROOT/sft_agent"
    [[ ! -e "$out/summary.json" ]] || { echo "ALREADY_DONE $out/summary.json"; exit 0; }
    port=${SFT_VLLM_PORT:-18121}
    name=dee-vllm-heldout-sft
    if ! curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      docker rm -f "$name" >/dev/null 2>&1 || true
      docker run -d --name "$name" \
        --gpus "device=${gpu}" --ipc=host --network host --ulimit memlock=-1 \
        -v /data1/hcc/deepresearch:/data1/hcc/deepresearch \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        vllm/vllm-openai:latest \
        "$SFT" --dtype bfloat16 --max-model-len 8192 \
        --gpu-memory-utilization 0.9 --port "$port" \
        --served-model-name sft8b --trust-remote-code >/dev/null
      ready=0
      for _ in $(seq 1 120); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then ready=1; break; fi
        docker ps --format '{{.Names}}' | grep -qx "$name" || {
          docker logs --tail 100 "$name"; exit 1;
        }
        sleep 2
      done
      [[ "$ready" == 1 ]] || { docker logs --tail 100 "$name"; exit 1; }
    fi
    run_dir="$out/runs"
    mkdir -p "$run_dir"
    env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= PYTHONPATH="$PROJECT" \
      "$PY" scripts/run_agent_rollout_smoke.py \
        --config "$CONFIG" --seed 42 --model-path "$SFT" --eval-file "$EVAL" \
        --max-samples 500 --top-k 5 --max-search-turns 2 \
        --temperature 0.0 --max-new-tokens 512 \
        --backend vllm_openai --vllm-base-url "http://127.0.0.1:${port}/v1" \
        --vllm-model-name sft8b --output-dir "$run_dir" \
        --run-tag heldout_n500_sft_corrected
    summary=$(find "$run_dir" -type f -name summary.json -printf '%T@ %p\n' \
      | sort -n | tail -n 1 | cut -d' ' -f2-)
    [[ -f "$summary" ]] || { echo "MISSING_SFT_SUMMARY"; exit 1; }
    cp "$summary" "$out/summary.json"
    ;;
  *) echo "ERROR unknown arm=$arm" >&2; exit 2 ;;
esac

echo "HELDOUT_ARM_PASS arm=$arm summary=$out/summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
DATA="$ROOT/results/60_provider_decoupled_replay/behavior_dev40"

cd "$ROOT"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${BEHAVIOR_GPU:-1} "$PY" scripts/eval_webmt_behavior_dev.py \
  --model-path results/44_hf_formal_grpo_step400/model_view \
  --data-dir "$DATA" --output-dir results/66_webmt_behavior_v2/base_grpo400 --batch-size 4
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${BEHAVIOR_GPU:-1} "$PY" scripts/eval_webmt_behavior_dev.py \
  --model-path outputs/65_webmt_lora_v2_token_balanced_merged \
  --data-dir "$DATA" --output-dir results/66_webmt_behavior_v2/webmt_lora_v2 --batch-size 4
echo WEBMT_BEHAVIOR_COMPARE_V2_PASS

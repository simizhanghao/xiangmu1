#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
cd "$ROOT"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W4_GPU:-1} "$PY" scripts/eval_atomic_controller_behavior.py \
  --model-path "${W4_MODEL:-outputs/65_webmt_lora_v2_token_balanced_merged}" \
  --data-dir results/60_provider_decoupled_replay/behavior_dev40 \
  --score-dir "${W4_SCORE_DIR:-results/67_w4_atomic_decision/score_audit_v2_sanitized}" \
  --output-dir "${W4_BEHAVIOR_OUT:-results/67_w4_atomic_decision/behavior_gate_v2_sanitized}" \
  --batch-size "${W4_BATCH_SIZE:-4}"
echo W4_ATOMIC_BEHAVIOR_EXIT=0

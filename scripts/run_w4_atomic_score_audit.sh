#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
MODEL=${W4_MODEL:-$ROOT/outputs/65_webmt_lora_v2_token_balanced_merged}
OUT=${W4_OUT:-$ROOT/results/67_w4_atomic_decision/score_audit_v2_sanitized}

cd "$ROOT"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W4_GPU:-1} "$PY" scripts/audit_atomic_decision_scores.py \
  --model-path "$MODEL" \
  --data-dir results/60_provider_decoupled_replay/behavior_dev40 \
  --output-dir "$OUT" --batch-size "${W4_BATCH_SIZE:-4}" \
  --fixed-remaining-budget "${W4_FIXED_REMAINING_BUDGET:-4}"
echo W4_0_SCORE_AUDIT_EXIT=0

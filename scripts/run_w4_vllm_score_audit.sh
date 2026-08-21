#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/hcc/deepresearch/Dee
cd "$ROOT"
env -u LD_LIBRARY_PATH /data1/hcc/eca-verl-vexact/.venv/bin/python scripts/audit_atomic_decision_scores_openai.py \
  --model-path outputs/65_webmt_lora_v2_token_balanced_merged \
  --data-dir results/60_provider_decoupled_replay/behavior_dev40 \
  --output-dir results/67_w4_atomic_decision/score_audit_v2_vllm_sanitized \
  --base-url http://127.0.0.1:18121/v1 --served-model w4v2 \
  --fixed-remaining-budget "${W4_FIXED_REMAINING_BUDGET:-4}"
echo W4_0_VLLM_SCORE_AUDIT_EXIT=0

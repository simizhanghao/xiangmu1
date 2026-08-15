#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check() { [[ -e "$2" ]] && printf 'PASS  %-18s %s\n' "$1" "$2" || printf 'WAIT  %-18s %s\n' "$1" "$2"; }
check base_model "$BASE_MODEL/config.json"
check sft_adapter "$SFT_ADAPTER/adapter_config.json"
check sft_merged "$SFT_MERGED/config.json"
check grpo_smoke "$PROJECT_ROOT/artifacts/evidence_grpo_smoke_ckpt/global_step_1/actor/huggingface/config.json"
for step in 200 400 600 800 1000; do
  check "dev_step${step}" "$PROJECT_ROOT/results/frozen_dev/step${step}/summary.json"
done
check rl_tracker "$RL_CKPT_ROOT/latest_checkpointed_iteration.txt"
check selection "$PROJECT_ROOT/results/checkpoint_selection.json"
check best_hf "$PROJECT_ROOT/artifacts/best_hf/config.json"

echo
echo "tmux:"
tmux ls 2>/dev/null || true
echo
echo "gpu:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true

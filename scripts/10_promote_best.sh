#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

[[ "${ALLOW_BEST_REPLACE:-0}" == 1 ]] || {
  echo "Refusing to replace best_hf without ALLOW_BEST_REPLACE=1" >&2
  exit 2
}
selection="$PROJECT_ROOT/results/checkpoint_selection.json"
require_file "$selection"
best_step=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_step"])' "$selection")
best_healthy=$("$PYTHON_BIN" -c 'import json,sys; x=json.load(open(sys.argv[1])); print(next(c["health_gate"] for c in x["candidates"] if c["step"]==x["best_step"]))' "$selection")
[[ "$best_healthy" == True ]] || {
  echo "ERROR selected checkpoint step=$best_step failed the frozen health gate" >&2
  exit 1
}
dst="$PROJECT_ROOT/artifacts/best_hf"
if [[ -f "$dst/FROZEN_GRPO_STEP" && "$(tr -d '[:space:]' <"$dst/FROZEN_GRPO_STEP")" == "$best_step" ]]; then
  echo "BEST_MODEL_ALREADY_CURRENT step=$best_step path=$dst"
  exit 0
fi
src="$RL_CKPT_ROOT/global_step_${best_step}/actor/huggingface"
require_file "$src/config.json"

tmp="$PROJECT_ROOT/artifacts/.best_hf_step${best_step}.tmp"
rm -rf -- "$tmp"
mkdir -p "$tmp"
# Immutable checkpoint files: hard links preserve the model when veRL retires old state.
cp -al "$src/." "$tmp/"
printf '%s\n' "$best_step" >"$tmp/FROZEN_GRPO_STEP"
if [[ -e "$dst" ]]; then
  rm -rf -- "$dst"
fi
mv "$tmp" "$dst"
sha256sum "$dst/config.json" "$dst/tokenizer.json" >"$PROJECT_ROOT/results/best_model_identity.sha256"
echo "BEST_MODEL_PROMOTED step=$best_step path=$dst"

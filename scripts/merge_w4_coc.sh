#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ADAPTER="$PROJECT_ROOT/outputs/68_w4_coc_dpo"
MERGED="$PROJECT_ROOT/outputs/69_w4_coc_merged"
require_file "$ADAPTER/adapter_model.safetensors"
if [[ -e "$MERGED" ]]; then
  echo "W4_COC_MERGE_TARGET_EXISTS=$MERGED"
  exit 1
fi
cd "$LLAMAFACTORY_ROOT"
run_llamafactory export "$PROJECT_ROOT/config/w4_coc_merge.yaml"
require_file "$MERGED/config.json"
sha256sum "$MERGED/config.json" "$MERGED/tokenizer.json" \
  > "$PROJECT_ROOT/results/69_w4_coc_merged_identity.sha256"
echo "W4_COC_MERGE_PASS=$MERGED"

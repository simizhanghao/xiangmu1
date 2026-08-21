#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ADAPTER="$PROJECT_ROOT/outputs/64_webmt_lora_v2_token_balanced"
MERGED="$PROJECT_ROOT/outputs/65_webmt_lora_v2_token_balanced_merged"
require_file "$ADAPTER/adapter_model.safetensors"
if [[ -e "$MERGED" ]]; then
  echo "WEBMT_MERGE_TARGET_EXISTS=$MERGED"
  exit 1
fi
cd "$LLAMAFACTORY_ROOT"
run_llamafactory export "$PROJECT_ROOT/config/webmt_merge_v2_token_balanced.yaml"
require_file "$MERGED/config.json"
sha256sum "$MERGED/config.json" "$MERGED/tokenizer.json" \
  > "$PROJECT_ROOT/results/65_webmt_lora_v2_merged_identity.sha256"
echo "WEBMT_V2_MERGE_PASS=$MERGED"

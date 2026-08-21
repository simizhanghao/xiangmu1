#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ADAPTER="$PROJECT_ROOT/outputs/61_webmt_lora_v1"
MERGED="$PROJECT_ROOT/outputs/62_webmt_lora_v1_merged"
require_file "$ADAPTER/adapter_model.safetensors"
if [[ -e "$MERGED" ]]; then
  echo "WEBMT_MERGE_TARGET_EXISTS=$MERGED"
  exit 1
fi
cd "$LLAMAFACTORY_ROOT"
run_llamafactory export "$PROJECT_ROOT/config/webmt_merge_v1.yaml"
require_file "$MERGED/config.json"
sha256sum "$MERGED/config.json" "$MERGED/tokenizer.json" \
  > "$PROJECT_ROOT/results/62_webmt_lora_v1_merged_identity.sha256"
echo "WEBMT_MERGE_PASS=$MERGED"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_dir "$SFT_ADAPTER"
[[ ! -e "$SFT_MERGED" ]] || {
  echo "ERROR merge target already exists: $SFT_MERGED" >&2
  echo "Move it aside explicitly before rerunning; this script will not overwrite a model." >&2
  exit 1
}

log="$PROJECT_ROOT/logs/sft_merge_$(date +%Y%m%d_%H%M%S).log"
cd "$LLAMAFACTORY_ROOT"
run_llamafactory export "$PROJECT_ROOT/config/sft_merge.yaml" 2>&1 | tee "$log"
require_file "$SFT_MERGED/config.json"
sha256sum "$SFT_MERGED/config.json" "$SFT_MERGED/tokenizer.json" \
  | tee "$PROJECT_ROOT/results/sft_merged_identity.sha256"
echo "SFT_MERGE_PASS=$SFT_MERGED"


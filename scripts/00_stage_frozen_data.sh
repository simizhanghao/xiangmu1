#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

source_root=${SOURCE_REPO:-/data1/hcc/deepresearch}

copy_frozen() {
  local rel="$1" dst="$2"
  local src="$source_root/$rel"
  require_file "$src"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$dst" ]]; then
    src_hash=$(sha256sum "$src" | awk '{print $1}')
    dst_hash=$(sha256sum "$dst" | awk '{print $1}')
    [[ "$src_hash" == "$dst_hash" ]] || {
      echo "ERROR local frozen input differs: $dst" >&2
      exit 1
    }
    echo "ALREADY_STAGED $dst"
  else
    cp --reflink=auto "$src" "$dst"
    echo "STAGED $src -> $dst"
  fi
}

copy_frozen data/sft/llamafactory/eca_coldstart_v1_train.jsonl "$SFT_DATA_DIR/eca_coldstart_v1_train.jsonl"
copy_frozen data/sft/llamafactory/eca_coldstart_v1_dev.jsonl "$SFT_DATA_DIR/eca_coldstart_v1_dev.jsonl"
copy_frozen data/sft/llamafactory/eca_coldstart_v1_smoke.jsonl "$SFT_DATA_DIR/eca_coldstart_v1_smoke.jsonl"
copy_frozen data/rl/train_smoke_128/train.parquet "$RL_TRAIN"
copy_frozen data/rl/train_smoke_128/val.parquet "$RL_VAL"
copy_frozen data/rl/train_smoke_128/contexts_index.jsonl "$BM25_INDEX"
copy_frozen data/eval/hotpotqa_200.jsonl "$FROZEN_DEV"

sha256sum \
  "$SFT_DATA_DIR/eca_coldstart_v1_train.jsonl" \
  "$SFT_DATA_DIR/eca_coldstart_v1_dev.jsonl" \
  "$SFT_DATA_DIR/eca_coldstart_v1_smoke.jsonl" \
  "$RL_TRAIN" "$RL_VAL" "$BM25_INDEX" "$FROZEN_DEV" \
  >"$PROJECT_ROOT/results/input_sha256.txt"
echo "FROZEN_DATA_STAGE_PASS"


#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "$(dirname "$BASE_MODEL")"
if [[ -s "$BASE_MODEL/config.json" ]]; then
  echo "MODEL_ALREADY_PRESENT=$BASE_MODEL"
  exit 0
fi

DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE:-hf-mirror}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}

case "$DOWNLOAD_SOURCE" in
  modelscope)
    command -v modelscope >/dev/null 2>&1 || {
      echo "ERROR: modelscope CLI not found" >&2; exit 1;
    }
    modelscope --help >/dev/null 2>&1 || {
      echo "ERROR: modelscope CLI entry exists but its Python package is unavailable" >&2
      echo "Use DOWNLOAD_SOURCE=hf-mirror instead" >&2
      exit 1
    }
    echo "DOWNLOAD_SOURCE=modelscope MODEL_ID=$MODEL_ID WORKERS=$DOWNLOAD_WORKERS"
    modelscope download "$MODEL_ID" \
      --local-dir "$BASE_MODEL" \
      --max-workers "$DOWNLOAD_WORKERS"
    ;;
  hf-mirror)
    command -v hf >/dev/null 2>&1 || {
      echo "ERROR: hf CLI not found" >&2; exit 1;
    }
    export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
    echo "DOWNLOAD_SOURCE=hf-mirror ENDPOINT=$HF_ENDPOINT MODEL_ID=$MODEL_ID WORKERS=$DOWNLOAD_WORKERS"
    hf download "$MODEL_ID" \
      --local-dir "$BASE_MODEL" \
      --max-workers "$DOWNLOAD_WORKERS"
    ;;
  huggingface)
    command -v hf >/dev/null 2>&1 || {
      echo "ERROR: hf CLI not found" >&2; exit 1;
    }
    echo "DOWNLOAD_SOURCE=huggingface MODEL_ID=$MODEL_ID WORKERS=$DOWNLOAD_WORKERS"
    hf download "$MODEL_ID" \
      --local-dir "$BASE_MODEL" \
      --max-workers "$DOWNLOAD_WORKERS"
    ;;
  *)
    echo "ERROR DOWNLOAD_SOURCE must be: modelscope | hf-mirror | huggingface" >&2
    exit 2
    ;;
esac

require_file "$BASE_MODEL/config.json"
"$PYTHON_BIN" - "$BASE_MODEL" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
c = json.loads((p / "config.json").read_text())
assert c.get("model_type") == "qwen3_moe", c.get("model_type")
print("MODEL_DOWNLOAD_PASS", p, "model_type=", c["model_type"])
PY

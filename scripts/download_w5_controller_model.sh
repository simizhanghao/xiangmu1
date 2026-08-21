#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/hcc/deepresearch/Dee
MODEL_DIR=$ROOT/model/Qwen3-1.7B
MODELSCOPE=${MODELSCOPE_BIN:-$ROOT/.venv-modelscope/bin/modelscope}
export MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS=${MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS:-8}
export MODELSCOPE_DOWNLOAD_PART_SIZE_MB=${MODELSCOPE_DOWNLOAD_PART_SIZE_MB:-160}
mkdir -p "$MODEL_DIR"
"$MODELSCOPE" download --model Qwen/Qwen3-1.7B --local_dir "$MODEL_DIR" --max-workers 8
test -s "$MODEL_DIR/config.json"
echo "W5_CONTROLLER_MODEL_READY path=$MODEL_DIR"

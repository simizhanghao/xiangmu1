#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

GPUS=${W7_GPUS:-1,2,3,6}
IFS=, read -r -a ids <<< "$GPUS"
[[ ${#ids[@]} -eq 4 ]] || { echo "W7_GPU_CONFIG_FAIL"; exit 1; }
test -s "$PROJECT_ROOT/model/Qwen3-1.7B/config.json"
test ! -e "$PROJECT_ROOT/outputs/84_w7_s2g_lora/adapter_config.json" || {
  echo "W7_OUTPUT_EXISTS"; exit 1;
}
env -u LD_LIBRARY_PATH "$LLAMAFACTORY_PYTHON" "$PROJECT_ROOT/scripts/build_w7_s2g_dataset.py"
for gpu in "${ids[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  [[ "$used" -lt 2048 ]] || { echo "W7_GPU_BUSY gpu=$gpu used_mib=$used"; exit 1; }
done
cd "$LLAMAFACTORY_ROOT"
export FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4
export MASTER_PORT=${MASTER_PORT:-29641}
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$GPUS"
run_llamafactory train "$PROJECT_ROOT/config/w7_s2g_lora.yaml"
echo W7_S2G_TRAIN_PASS

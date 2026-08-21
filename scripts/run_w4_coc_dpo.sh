#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

W4_GPUS=${W4_GPUS:-1,2,3,6}
IFS=, read -r -a gpu_ids <<< "$W4_GPUS"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "W4_GPU_CONFIG_FAIL expected=4 got=${#gpu_ids[@]}"; exit 1; }

env -u LD_LIBRARY_PATH "$LLAMAFACTORY_PYTHON" "$PROJECT_ROOT/scripts/prepare_w4_coc_dpo.py"
for gpu in "${gpu_ids[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
  [[ "$used" -lt 2048 ]] || { echo "W4_GPU_BUSY gpu=$gpu used_mib=$used"; exit 1; }
done

cd "$LLAMAFACTORY_ROOT"
export FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4
export MASTER_PORT=${MASTER_PORT:-29614} TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$W4_GPUS"
run_llamafactory train "$PROJECT_ROOT/config/w4_coc_dpo.yaml"

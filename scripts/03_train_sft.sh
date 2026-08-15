#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_file "$BASE_MODEL/config.json"
require_file "$LLAMAFACTORY_ROOT/data/eca_qwen3_30b_coldstart_train.jsonl"
[[ -x "$LLAMAFACTORY_PYTHON" ]] || {
  echo "ERROR SFT environment is missing: $LLAMAFACTORY_PYTHON" >&2
  echo "Run scripts/00_setup_sft_env.sh first." >&2
  exit 1
}
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$LLAMAFACTORY_PYTHON" - <<'PY'
import inspect
import deepspeed, flash_attn, peft, torch, transformers, trl
from transformers.integrations.flash_attention import flash_attention_forward
assert torch.cuda.is_available()
assert "s_aux is not None" in inspect.getsource(flash_attention_forward), (
    "Transformers FA2 optional-s_aux patch missing; run scripts/00_patch_sft_transformers.py"
)
print("SFT_DEPENDENCY_GATE_PASS", torch.__version__, transformers.__version__)
PY

log=${RUN_LOG:-"$PROJECT_ROOT/logs/sft_$(date +%Y%m%d_%H%M%S).log"}
cd "$LLAMAFACTORY_ROOT"
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="$N_GPUS"
export MASTER_PORT=${MASTER_PORT:-29531}
export TOKENIZERS_PARALLELISM=false
run_llamafactory train "$PROJECT_ROOT/config/sft_lora.yaml" 2>&1 | tee "$log"
echo "SFT_LOG=$log"

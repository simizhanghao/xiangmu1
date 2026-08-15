#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "[paths]"
for path in "$PROJECT_ROOT" "$LLAMAFACTORY_ROOT" "$VEXACT_ROOT"; do
  require_dir "$path"
  echo "OK $path"
done
for path in "$RL_TRAIN" "$RL_VAL" "$BM25_INDEX" "$FROZEN_DEV"; do
  require_file "$path"
  echo "OK $path"
done
require_file "$PROJECT_ROOT/results/experiment_contract.json"
require_file "$PROJECT_ROOT/results/sealed_test_manifest.json"
require_file "$PROJECT_ROOT/results/rl_parquet_compat_manifest.json"
grep -q 'EXPERIMENT_CONTRACT_PASS' "$PROJECT_ROOT/results/experiment_contract.json"
grep -q 'SEALED_TEST_FREEZE_PASS' "$PROJECT_ROOT/results/sealed_test_manifest.json"
grep -q 'RL_PARQUET_COMPAT_PASS' "$PROJECT_ROOT/results/rl_parquet_compat_manifest.json"
echo "OK experiment contract, sealed-test manifest, and RL Parquet compatibility gate"

echo "[data hashes]"
sha256sum \
  "$SFT_DATA_DIR/eca_coldstart_v1_train.jsonl" \
  "$SFT_DATA_DIR/eca_coldstart_v1_dev.jsonl" \
  "$RL_TRAIN_SOURCE" "$RL_VAL_SOURCE" "$RL_TRAIN" "$RL_VAL" \
  "$BM25_INDEX" "$FROZEN_DEV" \
  | tee "$PROJECT_ROOT/results/input_sha256.txt"

echo "[gpu]"
command -v nvidia-smi >/dev/null
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$gpu_count" -ge "$N_GPUS" ]] || {
  echo "ERROR need at least $N_GPUS visible GPUs, found $gpu_count" >&2
  exit 1
}
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "[disk]"
df -h "$PROJECT_ROOT"
free_gb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
[[ "$free_gb" -ge 80 ]] || {
  echo "ERROR recommend >=80 GiB free for 8B base + merged SFT + one resumable RL state; found ${free_gb} GiB" >&2
  exit 1
}

echo "[framework support]"
grep -q 'name="qwen3_nothink"' "$LLAMAFACTORY_ROOT/src/llamafactory/data/template.py"
require_file "$BASE_MODEL/config.json"
require_file "$VEXACT_ROOT/.venv/bin/python"
require_file "$LLAMAFACTORY_PYTHON"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$VEXACT_ROOT/.venv/bin/python" - <<'PY'
import bm25s, torch, transformers, verl, veomni, vexact, vllm
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("bm25s", getattr(bm25s, "__version__", "installed"))
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("verl", verl.__file__)
print("veomni", veomni.__file__)
print("vexact", vexact.__file__)
assert torch.cuda.is_available()
print("P0_PREFLIGHT_PASS")
PY

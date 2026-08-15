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

echo "[gate0 8b contract]"
[[ "$PROJECT_ROOT" == /data1/hcc/deepresearch/Dee ]]
[[ "$MODEL_ID" == Qwen/Qwen3-8B ]]
[[ "$BASE_MODEL" == /data1/hcc/deepresearch/Dee/model ]]
[[ "$SFT_MERGED" == /data1/hcc/deepresearch/Dee/artifacts/models/qwen3_8b_sft_merged ]]
[[ "${GRPO_FSDP_OFFLOAD}" == 0 ]]
[[ "${GRPO_MAX_MODEL_LEN}" == 8192 ]]
[[ "${GRPO_PP_SIZE}" == 4 ]]
grep -q 'template: qwen3_nothink' "$PROJECT_ROOT/config/sft_lora.yaml"
grep -q 'enable_thinking: false' "$PROJECT_ROOT/config/sft_lora.yaml"
grep -q 'segment {200|400|600|800}' "$PROJECT_ROOT/scripts/07_run_evidence_grpo.sh"
stale=$(grep -R --include='*.sh' --include='*.py' --include='*.yaml' --include='*.env' \
  --exclude='00_preflight.sh' -n '/data1/hcc/deepresearch/Qwen3_30B' \
  "$PROJECT_ROOT/config" "$PROJECT_ROOT/scripts" "$PROJECT_ROOT/src" || true)
if [[ -n "$stale" ]]; then
  printf '%s\n' "$stale"
  echo "ERROR live Qwen3_30B path in runtime config/scripts/src" >&2
  exit 1
fi
python3 - "$PROJECT_ROOT" "$BASE_MODEL" "$MODEL_ID" <<'PY'
import hashlib, json, sys
from pathlib import Path

root = Path(sys.argv[1])
base = Path(sys.argv[2])
model_id = sys.argv[3]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

cfg = json.loads((base / "config.json").read_text())
assert cfg.get("model_type") == "qwen3", cfg.get("model_type")
assert cfg.get("architectures") == ["Qwen3ForCausalLM"], cfg.get("architectures")
assert "moe" not in cfg.get("model_type", "")

replacements = (
    ("/data1/hcc/deepresearch/Qwen3_30B/xiangmu", str(root)),
    ("/data1/hcc/deepresearch/Qwen3_30B/model", str(base)),
)
for rel in (
    "results/experiment_contract.json",
    "results/sealed_test_manifest.json",
    "results/rl_parquet_compat_manifest.json",
):
    path = root / rel
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    if "Qwen3_30B" in text:
        raise SystemExit(f"ERROR stale Qwen3_30B path remains in {path}")
    path.write_text(text)

contract_path = root / "results/experiment_contract.json"
contract = json.loads(contract_path.read_text())
contract["model"] = {
    "path": str(base),
    "model_id": model_id,
    "model_type": cfg["model_type"],
    "architectures": cfg.get("architectures"),
    "config_sha256": sha256_file(base / "config.json"),
    "tokenizer_sha256": sha256_file(base / "tokenizer.json"),
}
contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")

gate = {
    "gate": "GATE0_CONTRACT_PASS",
    "project_root": str(root),
    "model_id": model_id,
    "base_model": str(base),
    "model_type": cfg["model_type"],
    "thinking": "disabled",
    "grpo_fsdp_offload": False,
    "grpo_max_model_len": 8192,
    "grpo_pp_size": 4,
    "milestones": [200, 400, 600, 800],
    "sealed_test_untouched": True,
}
(root / "results/gate0_contract.json").write_text(json.dumps(gate, indent=2) + "\n")
print("GATE0_CONTRACT_PASS")
print(json.dumps(gate, indent=2))
PY
grep -q 'EXPERIMENT_CONTRACT_PASS' "$PROJECT_ROOT/results/experiment_contract.json"
grep -q 'SEALED_TEST_FREEZE_PASS' "$PROJECT_ROOT/results/sealed_test_manifest.json"
grep -q 'RL_PARQUET_COMPAT_PASS' "$PROJECT_ROOT/results/rl_parquet_compat_manifest.json"
grep -q 'GATE0_CONTRACT_PASS' "$PROJECT_ROOT/results/gate0_contract.json"
echo "OK experiment contract, sealed-test manifest, RL Parquet compatibility, and Gate 0"

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

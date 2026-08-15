#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

model=${1:-$BASE_MODEL}
tag=${2:-base}
require_file "$model/config.json"
# The verifier changes cwd to the VeXact repository. Canonicalize user-supplied
# relative paths first so Transformers never mistakes a local model for a Hub ID.
model=$(realpath -e "$model")
out="$PROJECT_ROOT/results/vexact_compat/$tag"
mkdir -p "$out/capture"
log="$PROJECT_ROOT/logs/vexact_compat_${tag}_$(date +%Y%m%d_%H%M%S).log"

# Seeded VeXact sampling imports vLLM's Gumbel kernel. Fail before loading a
# 58GB model when the optional vllm extra was omitted from `uv sync`.
env -u LD_LIBRARY_PATH "$VEXACT_ROOT/.venv/bin/python" - <<'PY'
try:
    import vllm
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
except Exception as exc:
    raise SystemExit(
        "ERROR_VLLM_SEEDED_SAMPLER_MISSING: install the locked vllm extra with:\n"
        "  cd /data1/hcc/eca-verl-vexact && "
        "uv sync --frozen --extra gpu --extra vllm --extra verl --extra veomni\n"
        f"original_error={type(exc).__name__}: {exc}"
    ) from exc
print("VLLM_SEEDED_SAMPLER_IMPORT_PASS", vllm.__version__)
PY

export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export INFER_FA_IMPL=triton-invariant
export TOKENIZERS_PARALLELISM=false
# Rollout-side Qwen3-MoE always calls VeOmni's fused MoE kernel. The verifier
# must use the same arithmetic; its upstream test default is intentionally
# eager and is not an Exact parity configuration for MoE models.
export VEXACT_TESTS_MOE_IMPL=fused_triton

echo "VEXACT_COMPAT_MODEL_ABS=$model"

cd "$VEXACT_ROOT"
timeout --signal=TERM --kill-after=30s 30m \
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${COMPAT_GPU:-0} \
  "$VEXACT_ROOT/.venv/bin/python" tests/scripts/hf_inference.py \
    --model_path "$model" \
    --simulate_requests 4 \
    --max_length 256 \
    --max_new_tokens 1 \
    --max_num_batched_tokens 1024 \
    --max_cache_blocks 64 \
    --request_interval 0.01 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k -1 \
    --do_sample \
    --seed 42 \
    --enable_batch_invariant \
    --use_fp32_logits \
    --enforce_eager \
    --attn_impl triton-invariant \
    --output_dir "$out/capture" \
    2>&1 | tee "$log"

timeout --signal=TERM --kill-after=30s 30m \
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${COMPAT_GPU:-0} \
  "$VEXACT_ROOT/.venv/bin/python" tests/scripts/verify_logits_vs_native_hf.py \
    --model_path "$model" \
    --data_dir "$out/capture" \
    --model_backend veomni \
    --attn_impl triton-invariant \
    --enable_batch_invariant \
    --use_remove_padding \
    --logprobs_from_logits \
    --skip_backward \
    --rtol 0 \
    --atol 0 \
    --log_file "$out/verify.log" \
    2>&1 | tee -a "$log"

sha256sum "$model/config.json" "$model/tokenizer.json" >"$out/model_identity.sha256"
echo "VEXACT_MODEL_COMPAT_PASS tag=$tag model=$model"

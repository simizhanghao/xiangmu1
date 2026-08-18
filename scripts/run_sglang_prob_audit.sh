#!/usr/bin/env bash
# Step A: SGLang metrics-only probability audit. Does not edit 07_run_evidence_grpo.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MAX_SAMPLES=${MAX_SAMPLES:-8}
N=${N:-4}
OUT=${OUT:-"$PROJECT_ROOT/results/34_sglang_prob_audit"}
CONTAINER=${CONTAINER:-eca-verl}
# eca-verl bind-mounts host /data1/hcc/deepresearch → /workspace/deepresearch
C_ROOT=/workspace/deepresearch
C_DEE="$C_ROOT/Dee"
DUMP="$OUT/sglang_traj_dump.jsonl"
C_DUMP="$C_DEE/results/34_sglang_prob_audit/sglang_traj_dump.jsonl"
SLICE="$OUT/train_audit8.parquet"
C_SLICE="$C_DEE/results/34_sglang_prob_audit/train_audit8.parquet"
C_SFT="$C_DEE/artifacts/models/qwen3_8b_sft_merged"
C_SFT_OFFICIAL="$C_DEE/outputs/22_sft_qwen3_8b_merged"
LOG="$OUT/run.log"
mkdir -p "$OUT"
rm -f "$DUMP"

require_file "$RL_TRAIN"
require_file "$SFT_MERGED/config.json"
curl -sf "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || {
  echo "ERROR retriever :$RETRIEVER_PORT down" >&2
  exit 1
}
docker inspect "$CONTAINER" >/dev/null 2>&1 || {
  echo "SGLANG_ENV_MISSING container=$CONTAINER" >&2
  exit 3
}
docker exec "$CONTAINER" python3 -c "import sglang, verl; print('SGLANG_OK', sglang.__version__)" || {
  echo "SGLANG_IMPORT_FAIL inside $CONTAINER" >&2
  exit 3
}

c_has() { docker exec "$CONTAINER" test -f "$1"; }
if c_has "$C_SFT/config.json"; then
  echo "C_SFT $C_SFT"
elif c_has "$C_SFT_OFFICIAL/config.json"; then
  C_SFT="$C_SFT_OFFICIAL"
  echo "C_SFT_FALLBACK $C_SFT"
else
  echo "SFT_NOT_VISIBLE_IN_CONTAINER tried $C_SFT and $C_SFT_OFFICIAL" >&2
  exit 3
fi

# Same eca-verl as 3B. 8B LlamaFactory merge metadata is TF5; 4.57.1 rejects it.
# model_view = 22_ weight shards only + Dee/model tokenizer/config. Do not mutate 22_.
if [[ "$C_SFT" == "$C_SFT_OFFICIAL" ]]; then
  HOST_SFT="$PROJECT_ROOT/outputs/22_sft_qwen3_8b_merged"
else
  HOST_SFT="$SFT_MERGED"
fi
VIEW="$OUT/model_view"
C_VIEW="$C_DEE/results/34_sglang_prob_audit/model_view"
HOST_SFT="$HOST_SFT" HOST_BASE="$BASE_MODEL" VIEW="$VIEW" "$PYTHON_BIN" - <<'PY'
import os, shutil
from pathlib import Path
src = Path(os.environ["HOST_SFT"]).resolve()
base = Path(os.environ["HOST_BASE"]).resolve()
dst = Path(os.environ["VIEW"]).resolve()
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

def relink(target: Path, dest: Path) -> None:
    dest.symlink_to(os.path.relpath(target, dest.parent))

weight_ext = {".safetensors", ".bin"}
weight_names = {"model.safetensors.index.json"}
nw = 0
for p in src.iterdir():
    if p.suffix in weight_ext or p.name in weight_names:
        relink(p, dst / p.name)
        nw += 1
nm = 0
for p in base.iterdir():
    if p.suffix in weight_ext or p.name in weight_names:
        continue
    relink(p, dst / p.name)
    nm += 1
if nw < 1 or not (dst / "tokenizer.json").is_file() or not (dst / "config.json").is_file():
    raise SystemExit(f"OVERLAY_INCOMPLETE weights={nw} src={src} base={base}")
print(f"OVERLAY_OK weights={nw} meta={nm} cfg={os.readlink(dst / 'config.json')}")
PY
C_SFT="$C_VIEW"
docker exec "$CONTAINER" python3 -c "
import json, transformers
from pathlib import Path
from transformers import AutoTokenizer
p = Path('$C_VIEW')
cfg = p / 'config.json'
print('TF', transformers.__version__)
print('CFG_OK', cfg.is_file(), 'model_type', json.loads(cfg.read_text()).get('model_type'))
AutoTokenizer.from_pretrained('$C_DEE/model', local_files_only=True)
print('BASE_TOK_OK')
AutoTokenizer.from_pretrained(str(p), local_files_only=True)
print('TOKENIZER_LOAD_OK')
" || { echo "TOKENIZER_LOAD_FAIL $C_VIEW" >&2; exit 3; }
echo "DEE_VISIBLE_OK $C_DEE model=$C_SFT"

"$VEXACT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/audit_sglang_prob.py" \
  --config "$PROJECT_ROOT/config/harness_v1.json" \
  --seed 42 --debug --max-samples "$MAX_SAMPLES" \
  --output-dir "$OUT" \
  --slice-src "$RL_TRAIN" --slice-dst "$SLICE"
echo "SLICE_OK $SLICE"

# FSDP + SGLang inside historical eca-verl. Use Dee launch_grpo.py (no 3B metrics patch).
docker exec "$CONTAINER" python3 "$C_ROOT/scripts/patch_verl_sgl055_compat.py" || {
  echo "SGLANG_COMPAT_PATCH_FAIL" >&2
  exit 3
}
set +e
docker exec \
  -e PYTHONPATH="$C_DEE:/workspace/verl" \
  -e ECA_ROLLOUT_BACKEND=sglang \
  -e ECA_PARITY_DUMP="$C_DUMP" \
  -e ECA_EVIDENCE_WEIGHT=0.5 \
  -e ECA_AUDIT_STOP_MODE=sequence \
  -e ECA_MAX_ASSISTANT_TURN_TOKENS=256 \
  -e ECA_FINAL_ANSWER_RESERVE=256 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONUNBUFFERED=1 \
  -w /workspace/verl \
  "$CONTAINER" \
  python3 "$C_DEE/scripts/launch_grpo.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$C_SLICE" \
  data.val_files="$C_SLICE" \
  data.train_batch_size="$MAX_SAMPLES" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed=42 \
  actor_rollout_ref.model.path="$C_SFT" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-8 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MAX_SAMPLES" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature="${GRPO_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.top_p="${GRPO_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.val_kwargs.n="$N" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${GRPO_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${GRPO_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$C_DEE/config/rl/candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$C_DEE/config/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$C_DEE/src/rl/rewards_evidence.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=eca_qwen3_8b_sglang_audit \
  trainer.experiment_name=sglang_prob_audit \
  trainer.default_local_dir="$C_DEE/results/34_sglang_prob_audit/ckpt_scratch" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
docker exec "$CONTAINER" bash -lc 'ray stop --force 2>/dev/null; pkill -f sglang.launch_server 2>/dev/null; exit 0' || true
echo "SGLANG_GEN_EXIT=$rc"
test -s "$DUMP" || {
  echo "DUMP_EMPTY $DUMP" >&2
  exit 4
}

# Dump-only first (stdlib). HF recompute must unset host LD_LIBRARY_PATH
# (same as 07_run_evidence_grpo.sh) or vexact torch hits cusparse/nvJitLink.
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_sglang_prob.py" \
  --config "$PROJECT_ROOT/config/harness_v1.json" \
  --seed 42 --debug --max-samples "$MAX_SAMPLES" \
  --output-dir "$OUT" --dump "$DUMP"
echo "DUMP_ANALYZE_OK $OUT/sglang_prob_summary.json"
env -u LD_LIBRARY_PATH "$PYTHON_BIN" "$PROJECT_ROOT/scripts/audit_sglang_prob.py" \
  --config "$PROJECT_ROOT/config/harness_v1.json" \
  --seed 42 --debug --max-samples "$MAX_SAMPLES" \
  --output-dir "$OUT" --dump "$DUMP" \
  --recompute-hf --model-path "$PROJECT_ROOT/outputs/22_sft_qwen3_8b_merged"
echo "SGLANG_PROB_AUDIT_DONE out=$OUT"

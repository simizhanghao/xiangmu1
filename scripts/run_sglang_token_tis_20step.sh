#!/usr/bin/env bash
# Gate 5: 20-step SGLang + official Decoupled Token-TIS. Does not edit 07_run_evidence_grpo.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"

BATCH=${GRPO_BATCH:-32}
N=${GRPO_N:-4}
MINI=${GRPO_MINI_BATCH:-8}
MICRO=${GRPO_MICRO_BATCH:-1}
LR=${GRPO_LR:-1e-6}
STEPS=${GRPO_STEPS:-20}
OUT=${OUT:-"$PROJECT_ROOT/results/36_sglang_token_tis_20step"}
CONTAINER=${CONTAINER:-eca-verl}
C_ROOT=/workspace/deepresearch
C_DEE="$C_ROOT/Dee"
C_TRAIN="$C_DEE/data/rl/train_verl.parquet"
C_VAL="$C_DEE/data/rl/val_verl.parquet"
LOG="$OUT/run.log"
mkdir -p "$OUT"

require_file "$RL_TRAIN"
require_file "$RL_VAL"
require_file "$SFT_MERGED/config.json"
curl -sf "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || {
  echo "ERROR retriever :$RETRIEVER_PORT down" >&2
  exit 1
}
docker exec "$CONTAINER" python3 -c \
  "from verl.trainer.config.algorithm import RolloutCorrectionConfig as C; print('TIS_OK', C.decoupled_token_is())" \
  | tee -a "$LOG" \
  || {
    echo "ROLLOUT_CORRECTION_IMPORT_FAIL" >&2
    exit 3
  }

# 22_ shards + Dee/model tokenizer/config. Relative links. Do not mutate 22_.
VIEW="$OUT/model_view"
C_VIEW="$C_DEE/results/36_sglang_token_tis_20step/model_view"
HOST_SFT="$PROJECT_ROOT/outputs/22_sft_qwen3_8b_merged"
HOST_SFT="$HOST_SFT" HOST_BASE="$BASE_MODEL" VIEW="$VIEW" "$PYTHON_BIN" - <<'PY' | tee -a "$LOG"
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
for p in base.iterdir():
    if p.suffix in weight_ext or p.name in weight_names:
        continue
    relink(p, dst / p.name)
if nw < 1 or not (dst / "config.json").is_file():
    raise SystemExit("OVERLAY_INCOMPLETE")
print(f"OVERLAY_OK weights={nw} cfg={os.readlink(dst / 'config.json')}")
PY

echo "GATE5 batch=$BATCH n=$N mini=$MINI lr=$LR steps=$STEPS tis=token/2.0" | tee -a "$LOG"
set +e
docker exec \
  -e PYTHONPATH="$C_DEE:/workspace/verl" \
  -e ECA_ROLLOUT_BACKEND=sglang \
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
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.bypass_mode=false \
  data.train_files="$C_TRAIN" \
  data.val_files="$C_VAL" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed=42 \
  actor_rollout_ref.model.path="$C_VIEW" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature="${GRPO_TEMPERATURE:-0.7}" \
  actor_rollout_ref.rollout.top_p="${GRPO_TOP_P:-0.95}" \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.enforce_eager=True \
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
  trainer.total_epochs=20 \
  trainer.total_training_steps="$STEPS" \
  trainer.val_before_train=False \
  trainer.save_freq=10 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.logger='["console"]' \
  trainer.project_name=eca_qwen3_8b_sglang_tis \
  trainer.experiment_name=sglang_token_tis_20step \
  trainer.default_local_dir="$C_DEE/results/36_sglang_token_tis_20step/ckpt" \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
docker exec "$CONTAINER" bash -c 'ray stop --force 2>/dev/null; pkill -f sglang.launch_server 2>/dev/null; exit 0' || true

OUT="$OUT" LOG="$LOG" "$PYTHON_BIN" - <<'PY' || true
import json, os, re
from pathlib import Path
log = Path(os.environ["LOG"])
out = Path(os.environ["OUT"]) / "gate5_step_metrics.jsonl"
pat = re.compile(r"\bstep:(\d+)\s+-\s+(.*)")
rows = []
if log.is_file():
    for line in log.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        rec = {"step": int(m.group(1))}
        for tok in m.group(2).split(" - "):
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                rec[k] = float(v)
            except ValueError:
                rec[k] = v
        rows.append(rec)
# keep last record per step
uniq = {}
for rec in rows:
    uniq[rec["step"]] = rec
with out.open("w") as f:
    for k in sorted(uniq):
        f.write(json.dumps(uniq[k], ensure_ascii=False) + "\n")
print(f"GATE5_METRICS n={len(uniq)} path={out}")
PY

echo "GATE5_EXIT=$rc log=$LOG"
exit "$rc"

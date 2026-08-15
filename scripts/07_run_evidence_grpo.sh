#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mode=${1:-}
target_step=${2:-}
case "$mode" in
  smoke)
    target_step=1
    ckpt_root="$PROJECT_ROOT/artifacts/evidence_grpo_smoke_ckpt"
    # Same batch/n as formal so 1-step smoke is a real memory/contract probe.
    batch="$GRPO_BATCH"
    n="$GRPO_N"
    micro="$GRPO_MICRO_BATCH"
    # VeOmni cosine scheduler does min_lr/init_lr; lr=0 crashes at init_model.
    # 1e-8 keeps the 1-step smoke update negligible while unblocking Exact GRPO.
    lr=1e-8
    save_freq=1
    resume_mode=disable
    experiment=qwen3_8b_evidence_smoke
    ;;
  segment)
    case "$target_step" in 200|400|600|800) ;; *)
      echo "usage: $0 segment {200|400|600|800}" >&2; exit 2;; esac
    ckpt_root="$RL_CKPT_ROOT"
    batch="$GRPO_BATCH"
    n="$GRPO_N"
    micro="$GRPO_MICRO_BATCH"
    lr=1e-6
    save_freq=200
    resume_mode=auto
    experiment=qwen3_8b_evidence_800
    ;;
  *)
    echo "usage: $0 smoke | segment {200|400|600|800}" >&2
    exit 2
    ;;
esac

require_file "$SFT_MERGED/config.json"
require_file "$RL_TRAIN"
require_file "$RL_VAL"
curl -sf "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || {
  echo "ERROR Candidate-BM25 is not healthy on :$RETRIEVER_PORT" >&2
  echo "Start it in another tmux window with scripts/06_start_retriever.sh" >&2
  exit 1
}

mkdir -p "$ckpt_root" "$PROJECT_ROOT/results/training"
tracker="$ckpt_root/latest_checkpointed_iteration.txt"
if [[ -s "$tracker" ]]; then
  current=$(tr -d '[:space:]' <"$tracker")
  [[ "$target_step" -gt "$current" ]] || {
    echo "ERROR target_step=$target_step is not greater than current=$current" >&2
    exit 1
  }
else
  current=0
fi

stamp=$(date +%Y%m%d_%H%M%S)
log=${RUN_LOG:-"$PROJECT_ROOT/logs/grpo_${mode}_to${target_step}_${stamp}.log"}
tensorboard_dir=${TENSORBOARD_DIR:-"$PROJECT_ROOT/tensorboard/$experiment"}
mkdir -p "$tensorboard_dir"
export TENSORBOARD_DIR="$tensorboard_dir"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export ECA_ROLLOUT_BACKEND=vexact
export ECA_EVIDENCE_WEIGHT
export ECA_AUDIT_STOP_MODE=sequence
export ECA_MAX_ASSISTANT_TURN_TOKENS=256
export ECA_FINAL_ANSWER_RESERVE=256
export INFER_FA_IMPL=triton-invariant
export VEOMNI_ATTN_IMPLEMENTATION=triton-invariant
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

case "${GRPO_FSDP_OFFLOAD:-0}" in
  1|true|True|TRUE) fsdp_offload=True ;;
  *) fsdp_offload=False ;;
esac

echo "GRPO mode=$mode current=$current target=$target_step batch=$batch n=$n micro=$micro fsdp_offload=$fsdp_offload"
echo "log=$log"
echo "tensorboard_dir=$tensorboard_dir"

set +e
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "$VEXACT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/launch_grpo.py" \
  model_engine=veomni \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$RL_TRAIN" \
  data.val_files="$RL_VAL" \
  data.train_batch_size="$batch" \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.seed=42 \
  actor_rollout_ref.model.path="$SFT_MERGED" \
  actor_rollout_ref.model.external_lib=vexact.integrations.verl.fsdp_enable_invariant \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.optim.lr="$lr" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="${GRPO_MINI_BATCH:-$batch}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$micro" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.actor.veomni.fsdp_size="$N_GPUS" \
  actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
  actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
  actor_rollout_ref.actor.veomni.attn_implementation=triton-invariant \
  actor_rollout_ref.actor.veomni.moe_implementation=fused_triton \
  actor_rollout_ref.actor.veomni.enable_fsdp_offload="$fsdp_offload" \
  actor_rollout_ref.actor.veomni.param_offload=False \
  actor_rollout_ref.actor.veomni.optimizer_offload=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$micro" \
  actor_rollout_ref.ref.veomni.enable_fsdp_offload="$fsdp_offload" \
  actor_rollout_ref.ref.veomni.param_offload=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vexact \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.n="$n" \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size="${GRPO_PP_SIZE:-4}" \
  actor_rollout_ref.rollout.max_model_len="$GRPO_MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.max_num_seqs="$GRPO_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$GRPO_MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$micro" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_cache_blocks="$GRPO_MAX_CACHE_BLOCKS" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.max_model_len="$GRPO_MAX_MODEL_LEN" \
  ++actor_rollout_ref.rollout.engine_kwargs.vexact.attn_impl=triton-invariant \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=384 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_ROOT/config/rl/candidate_bm25_tool.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=eca_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$PROJECT_ROOT/config/rl/eca_agent_loop.yaml" \
  reward.custom_reward_function.path="$PROJECT_ROOT/src/rl/rewards_evidence.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_epochs=800 \
  trainer.total_training_steps="$target_step" \
  trainer.val_before_train=False \
  trainer.save_freq="$save_freq" \
  trainer.test_freq=-1 \
  trainer.resume_mode="$resume_mode" \
  trainer.max_actor_ckpt_to_keep=1 \
  'trainer.logger=[console,tensorboard]' \
  trainer.project_name=eca_qwen3_8b_final \
  trainer.experiment_name="$experiment" \
  trainer.default_local_dir="$ckpt_root" \
  2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
echo "GRPO_EXIT=$rc"
[[ "$rc" -eq 0 ]]

[[ -s "$tracker" && "$(tr -d '[:space:]' <"$tracker")" == "$target_step" ]]
require_file "$ckpt_root/global_step_${target_step}/actor/huggingface/config.json"
echo "GRPO_SEGMENT_PASS step=$target_step hf=$ckpt_root/global_step_${target_step}/actor/huggingface"

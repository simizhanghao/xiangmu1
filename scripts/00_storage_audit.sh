#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch

echo "[active processes: inspect before deleting anything]"
pgrep -af 'raylet|gcs_server|verl|vexact|tensorboard.main|start_candidate_retrieval_server' || true

echo "[formal checkpoints: KEEP]"
du -sh \
  "$ROOT/outputs/00_sft_v1_merged" \
  "$ROOT/outputs/rl/02_hf_answer_only_step100" \
  "$ROOT/outputs/rl/03_hf_evidence_step400" \
  2>/dev/null || true

echo "[old diagnostic checkpoints: cleanup candidates]"
du -sh \
  "$ROOT/outputs/rl/07_ckpt_boundary_exact" \
  "$ROOT/outputs/rl/08_ckpt_rfpp_baseline" \
  "$ROOT/outputs/rl/09_ckpt_grpo_no_std" \
  "$ROOT/results/18_boundary_exact_rollout/checkpoints/step10_hf" \
  "$ROOT/results/20_rfpp_baseline/checkpoints/step10_hf" \
  "$ROOT/results/20_grpo_no_std/checkpoints/step10_hf" \
  2>/dev/null || true

echo "[partial downloads and generic scratch files]"
find "$ROOT/Qwen3_30B" "$ROOT/outputs" "$ROOT/results" \
  -xdev -type f \
  \( -name '*.incomplete' -o -name '*.part' -o -name '*.tmp' -o -name 'core.*' \) \
  -print 2>/dev/null || true

echo "[disk]"
df -h "$ROOT"
echo "STORAGE_AUDIT_COMPLETE_READ_ONLY"

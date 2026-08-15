#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

require_file "$PROJECT_ROOT/results/rl_parquet_compat_manifest.json"
grep -q RL_PARQUET_COMPAT_PASS "$PROJECT_ROOT/results/rl_parquet_compat_manifest.json"
curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null

tb_dir="$PROJECT_ROOT/tensorboard/qwen3_30b_evidence_1000"
mkdir -p "$tb_dir" "$PROJECT_ROOT/logs"

smoke_hf="$PROJECT_ROOT/artifacts/evidence_grpo_smoke_ckpt/global_step_1/actor/huggingface"
if [[ ! -f "$smoke_hf/config.json" ]]; then
  echo "[pipeline] Exact GRPO 1-step smoke"
  RUN_LOG="$PROJECT_ROOT/logs/grpo_smoke_to1_$(date +%Y%m%d_%H%M%S).log" \
    TENSORBOARD_DIR="$tb_dir" \
    bash "$PROJECT_ROOT/scripts/07_run_evidence_grpo.sh" smoke
else
  echo "[pipeline] smoke already passed: $smoke_hf"
fi
require_file "$smoke_hf/config.json"
echo "[pipeline] SMOKE_HARD_GATE_PASS"

for step in 200 400 600 800 1000; do
  tracker="$RL_CKPT_ROOT/latest_checkpointed_iteration.txt"
  current=0
  [[ -s "$tracker" ]] && current=$(tr -d '[:space:]' <"$tracker")

  if (( current < step )); then
    echo "[pipeline] train current=$current target=$step"
    RUN_LOG="$PROJECT_ROOT/logs/grpo_segment_to${step}_$(date +%Y%m%d_%H%M%S).log" \
      TENSORBOARD_DIR="$tb_dir" \
      bash "$PROJECT_ROOT/scripts/07_run_evidence_grpo.sh" segment "$step"
  else
    echo "[pipeline] training target=$step already reached (tracker=$current)"
  fi

  hf="$RL_CKPT_ROOT/global_step_${step}/actor/huggingface"
  summary="$PROJECT_ROOT/results/frozen_dev/step${step}/summary.json"
  if [[ ! -f "$summary" ]]; then
    require_file "$hf/config.json"
    echo "[pipeline] frozen-dev step=$step"
    EVAL_GPU=0 EVAL_MAX_SAMPLES=200 \
      bash "$PROJECT_ROOT/scripts/08_eval_frozen_dev.sh" \
        "step${step}" agent "$hf"
  else
    echo "[pipeline] frozen-dev step=$step already exists"
  fi

  "$PYTHON_BIN" - "$summary" <<'PY'
import json
import sys

path = sys.argv[1]
s = json.load(open(path, encoding="utf-8"))
gate = (
    float(s["finish_rate"]) >= 0.95
    and float(s["parse_ok_rate"]) >= 0.95
    and float(s["observation_mask_ok_rate"]) == 1.0
)
print(
    "[pipeline] HEALTH_GATE",
    "PASS" if gate else "FAIL",
    f"finish={s['finish_rate']}",
    f"parse={s['parse_ok_rate']}",
    f"observation_mask={s['observation_mask_ok_rate']}",
)
if not gate:
    raise SystemExit(3)
PY

  if [[ "$step" == 1000 ]]; then
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/09_select_best.py"
  else
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/09_select_best.py" --allow-partial
  fi
  ALLOW_BEST_REPLACE=1 bash "$PROJECT_ROOT/scripts/10_promote_best.sh"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/12_build_result_table.py"
  echo "[pipeline] STEP_${step}_TRAIN_EVAL_SELECT_PASS"
done

echo "FULL_GRPO_PIPELINE_PASS"
echo "Sealed Test remains unopened."

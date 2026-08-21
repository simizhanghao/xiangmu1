#!/usr/bin/env bash
set -uo pipefail
ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
cd "$ROOT"

bash scripts/run_w6_stage1_train.sh
train_rc=$?
echo "W6_STAGE1_TRAIN_EXIT=$train_rc"
if [[ $train_rc -ne 0 ]]; then
  exit "$train_rc"
fi

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W6_STAGE1_EVAL_GPU:-1} \
  "$PY" scripts/eval_w6_stage1_offline.py
eval_rc=$?
echo "W6_STAGE1_EVAL_EXIT=$eval_rc"
exit "$eval_rc"

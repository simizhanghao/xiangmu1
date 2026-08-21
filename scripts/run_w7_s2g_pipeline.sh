#!/usr/bin/env bash
set -uo pipefail
ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
cd "$ROOT"
bash scripts/run_w7_s2g_train.sh
train_rc=$?
echo "W7_S2G_TRAIN_EXIT=$train_rc"
[[ $train_rc -eq 0 ]] || exit "$train_rc"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W7_EVAL_GPU:-1} \
  "$PY" scripts/eval_w7_s2g_offline.py
eval_rc=$?
echo "W7_S2G_EVAL_EXIT=$eval_rc"
exit "$eval_rc"

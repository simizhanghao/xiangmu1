#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/LlamaFactory/.venv/bin/python
cd "$ROOT"
env -u LD_LIBRARY_PATH "$PY" scripts/prepare_w55_probe_split.py
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W55_GPU:-1} "$PY" scripts/extract_w55_hidden.py
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${W55_GPU:-1} "$PY" scripts/train_eval_w55_linear_probe.py
echo W55_LINEAR_PROBE_PIPELINE_DONE

#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
OUT="$ROOT/results/58_w3_recovery/q2_beam3_deepseek"
MANIFEST="$ROOT/results/57_bocha_migration/depth_mining_n400/candidate_manifest.jsonl"

: "${BOCHA_API_KEY:?BOCHA_API_KEY is required}"
: "${TEACHER_API_KEY:?TEACHER_API_KEY is required}"
: "${TEACHER_BASE_URL:=https://api.deepseek.com}"
: "${TEACHER_MODEL:=deepseek-v4-flash}"

cd "$ROOT"
mkdir -p "$OUT"

env -u LD_LIBRARY_PATH \
  "$PY" scripts/build_web_multiturn_v2.py \
  --candidate-manifest "$MANIFEST" \
  --output-dir "$OUT" \
  --target 112 \
  --max-candidates 400 \
  --quota-depth1 0 \
  --quota-depth2 112 \
  --quota-depth3 0 \
  --query-beam-size 3 \
  --web-provider bocha \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY" \
  --teacher-temperature 0 \
  --teacher-seed 42 \
  --diagnostic-log "$OUT/teacher_gold_blind_diagnostics.jsonl" \
  --seed 42

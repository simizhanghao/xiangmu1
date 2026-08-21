#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
BASE="$ROOT/results/59_source_first_d2/smoke_v2_bridge_frozen"
POOL="$BASE/synthesis/builder_pool.jsonl"
MANIFEST="$BASE/synthesis/candidate_manifest.jsonl"
BUILT="$BASE/causal_builder_subset5"

: "${BOCHA_API_KEY:?BOCHA_API_KEY is required}"
: "${TEACHER_API_KEY:?TEACHER_API_KEY is required}"
: "${TEACHER_BASE_URL:=https://api.deepseek.com}"
: "${TEACHER_MODEL:=deepseek-v4-flash}"

cd "$ROOT"
accepted=$(wc -l < "$POOL")
if [ "$accepted" -eq 0 ]; then
  echo SOURCE_FIRST_V2_NO_CANDIDATES
  exit 1
fi

set +e
env -u LD_LIBRARY_PATH "$PY" scripts/build_web_multiturn_v2.py \
  --pool "$POOL" \
  --pool-scope external_isolated \
  --candidate-manifest "$MANIFEST" \
  --output-dir "$BUILT" \
  --target "$accepted" \
  --max-candidates "$accepted" \
  --quota-depth1 0 \
  --quota-depth2 "$accepted" \
  --quota-depth3 0 \
  --query-beam-size 1 \
  --web-provider bocha \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY" \
  --teacher-temperature 0 \
  --teacher-seed 42 \
  --diagnostic-log "$BUILT/teacher_gold_blind_diagnostics.jsonl" \
  --seed 42
build_rc=$?
set -e

env -u LD_LIBRARY_PATH "$PY" scripts/audit_web_multiturn_v2.py \
  "$BUILT" --allow-incomplete
audit_rc=$?
echo "SOURCE_FIRST_V2_SUBSET_BUILD_EXIT=$build_rc AUDIT_EXIT=$audit_rc"

#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

echo "[stage] aggregating full batch k-study outputs"

for K in "${K_VALUES[@]}"; do
  "$PY" scripts/aggregate_corpus.py \
    --root "$KSTUDY_BASE/batch/study_runs/k${K}/capa" \
    --bootstrap 10000
done

SUMMARY_CONDITIONS=()
for K in "${K_VALUES[@]}"; do
  SUMMARY_CONDITIONS+=(--condition "${K}=$KSTUDY_BASE/batch/study_runs/k${K}/capa")
done

"$PY" scripts/k_window_study.py summarize \
  "${SUMMARY_CONDITIONS[@]}" \
  --out-dir "$KSTUDY_BASE/batch/summary"

echo "[done] batch summary: $KSTUDY_BASE/batch/summary"

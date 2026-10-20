#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

CANONICAL_BATCH_PREP_ROOT="$KSTUDY_BASE/batch/canonical_seed_k${K_REF}/_prep"
SHARED_BATCH_EIM_CACHE_ROOT="$KSTUDY_BASE/batch/shared_eim_cache"
RUN_SCRIPT="$KSTUDY_BASE/batch/run_k_study.sh"
VALIDATION_REPORT="$KSTUDY_BASE/batch/canonical_prep_validation.json"

if [[ ! -e "$CANONICAL_BATCH_PREP_ROOT" ]]; then
  echo "[ERROR] Missing canonical batch prep root: $CANONICAL_BATCH_PREP_ROOT" >&2
  echo "[ERROR] Run scripts/k_window_study/04_batch_canonical_freeze.sh first." >&2
  exit 1
fi

echo "[stage] validating canonical batch prep"

"$PY" scripts/k_window_study.py validate-canonical-prep \
  --input-dir "$INPUT_DIR" \
  --canonical-prep-root "$CANONICAL_BATCH_PREP_ROOT" \
  --limit "$BATCH_LIMIT" \
  --output-json "$VALIDATION_REPORT"

echo "[stage] emitting shard-wise batch commands"

"$PY" scripts/k_window_study.py emit-batch-commands \
  --input-dir "$INPUT_DIR" \
  --output-root "$KSTUDY_BASE/batch/study_runs" \
  --canonical-prep-root "$CANONICAL_BATCH_PREP_ROOT" \
  --shared-eim-cache-root "$SHARED_BATCH_EIM_CACHE_ROOT" \
  --model "$MODEL" \
  --limit "$BATCH_LIMIT" \
  --k-values "$K_VALUES_CSV" \
  --participants all \
  --capa-N "$N" \
  --cap-meeting "$CAP_MEETING" \
  --cap-per-participant "$CAP_PER_PARTICIPANT" \
  --min-gap "$MIN_GAP" \
  --ctx-strict "$CTX_STRICT" \
  --rel-strict "$REL_STRICT" \
  --ctx-loose "$CTX_LOOSE" \
  --rel-loose "$REL_LOOSE" \
  --python-bin "$PY" \
  --output "$RUN_SCRIPT"

echo "[stage] executing generated batch run script: $RUN_SCRIPT"
bash -euo pipefail "$RUN_SCRIPT"

echo "[done] batch study runs written under: $KSTUDY_BASE/batch/study_runs"

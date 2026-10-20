#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

CANONICAL_REDUCED_PREP="$KSTUDY_BASE/reduced/${REDUCED_MEETING}/canonical_seed_k${K_REF}/_prep/${REDUCED_MEETING}"
REDUCED_SHARED_EIM_CACHE="$KSTUDY_BASE/reduced/${REDUCED_MEETING}/shared_eim_cache/${REDUCED_MEETING}"

ensure_path_exists "$CANONICAL_REDUCED_PREP/capa_anchor_inventory" "canonical reduced anchor inventory"

echo "[stage] reduced k sweep for ${REDUCED_MEETING}"

for K in "${K_VALUES[@]}"; do
  "$PY" scripts/run_meeting_pipeline.py \
    --input "$INPUT_DIR/${REDUCED_MEETING}_parsed.json" \
    --output-dir "$KSTUDY_BASE/reduced/${REDUCED_MEETING}/k${K}/${REDUCED_MEETING}" \
    --shared-prep-dir "$KSTUDY_BASE/reduced/${REDUCED_MEETING}/k${K}/_prep/${REDUCED_MEETING}" \
    --participants all \
    --delegate-variant capa \
    --model "$MODEL" \
    --capa-N "$N" \
    --capa-k "$K" \
    --anchor-inventory "$CANONICAL_REDUCED_PREP/capa_anchor_inventory" \
    --anchor-seed-k "$K_REF" \
    --static-prep-source-dir "$CANONICAL_REDUCED_PREP" \
    --eim-cache-dir "$REDUCED_SHARED_EIM_CACHE" \
    --cap-meeting "$CAP_MEETING" \
    --cap-per-participant "$CAP_PER_PARTICIPANT" \
    --min-gap "$MIN_GAP" \
    --ctx-strict "$CTX_STRICT" \
    --rel-strict "$REL_STRICT" \
    --ctx-loose "$CTX_LOOSE" \
    --rel-loose "$REL_LOOSE" \
    --debug-max-utterances "$REDUCED_DEBUG_UTTS"

  "$PY" scripts/aggregate_corpus.py \
    --root "$KSTUDY_BASE/reduced/${REDUCED_MEETING}/k${K}/${REDUCED_MEETING}" \
    --bootstrap 10000
done

SUMMARY_CONDITIONS=()
for K in "${K_VALUES[@]}"; do
  SUMMARY_CONDITIONS+=(--condition "${K}=$KSTUDY_BASE/reduced/${REDUCED_MEETING}/k${K}/${REDUCED_MEETING}")
done

"$PY" scripts/k_window_study.py summarize \
  "${SUMMARY_CONDITIONS[@]}" \
  --out-dir "$KSTUDY_BASE/reduced/${REDUCED_MEETING}/summary"

echo "[done] reduced sweep summary: $KSTUDY_BASE/reduced/${REDUCED_MEETING}/summary"

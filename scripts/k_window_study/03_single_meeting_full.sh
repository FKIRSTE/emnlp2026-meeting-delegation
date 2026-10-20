#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

CANONICAL_FULL_PREP="$KSTUDY_BASE/full_one/${FULL_MEETING}/canonical_seed_k${K_REF}/_prep/${FULL_MEETING}"
FULL_SHARED_EIM_CACHE="$KSTUDY_BASE/full_one/${FULL_MEETING}/shared_eim_cache/${FULL_MEETING}"

echo "[stage] full single-meeting canonical freeze for ${FULL_MEETING}"

"$PY" src/data_pipeline.py \
  --input "$INPUT_DIR/${FULL_MEETING}_parsed.json" \
  --output "$CANONICAL_FULL_PREP" \
  --meeting-id "$FULL_MEETING" \
  --mode capa \
  --selector llm-idea \
  --capa-N "$N" \
  --capa-k "$K_REF" \
  --freeze-anchor-inventory \
  --anchor-seed-k "$K_REF" \
  --cap-meeting "$CAP_MEETING" \
  --cap-per-participant "$CAP_PER_PARTICIPANT" \
  --min-gap "$MIN_GAP" \
  --debug-capa-episodes \
  --jit-enable-sliding-windows \
  --jit-enable-llm-cleanup \
  --model "$MODEL" \
  --resume

echo "[stage] full single-meeting k sweep for ${FULL_MEETING}"

for K in "${K_VALUES[@]}"; do
  "$PY" scripts/run_meeting_pipeline.py \
    --input "$INPUT_DIR/${FULL_MEETING}_parsed.json" \
    --output-dir "$KSTUDY_BASE/full_one/${FULL_MEETING}/k${K}/${FULL_MEETING}" \
    --shared-prep-dir "$KSTUDY_BASE/full_one/${FULL_MEETING}/k${K}/_prep/${FULL_MEETING}" \
    --participants all \
    --delegate-variant capa \
    --model "$MODEL" \
    --capa-N "$N" \
    --capa-k "$K" \
    --anchor-inventory "$CANONICAL_FULL_PREP/capa_anchor_inventory" \
    --anchor-seed-k "$K_REF" \
    --static-prep-source-dir "$CANONICAL_FULL_PREP" \
    --eim-cache-dir "$FULL_SHARED_EIM_CACHE" \
    --cap-meeting "$CAP_MEETING" \
    --cap-per-participant "$CAP_PER_PARTICIPANT" \
    --min-gap "$MIN_GAP" \
    --ctx-strict "$CTX_STRICT" \
    --rel-strict "$REL_STRICT" \
    --ctx-loose "$CTX_LOOSE" \
    --rel-loose "$REL_LOOSE"

  "$PY" scripts/aggregate_corpus.py \
    --root "$KSTUDY_BASE/full_one/${FULL_MEETING}/k${K}/${FULL_MEETING}" \
    --bootstrap 10000
done

SUMMARY_CONDITIONS=()
for K in "${K_VALUES[@]}"; do
  SUMMARY_CONDITIONS+=(--condition "${K}=$KSTUDY_BASE/full_one/${FULL_MEETING}/k${K}/${FULL_MEETING}")
done

"$PY" scripts/k_window_study.py summarize \
  "${SUMMARY_CONDITIONS[@]}" \
  --out-dir "$KSTUDY_BASE/full_one/${FULL_MEETING}/summary"

echo "[done] full single-meeting summary: $KSTUDY_BASE/full_one/${FULL_MEETING}/summary"

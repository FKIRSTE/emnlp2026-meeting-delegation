#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

CANONICAL_REDUCED_PREP="$KSTUDY_BASE/reduced/${REDUCED_MEETING}/canonical_seed_k${K_REF}/_prep/${REDUCED_MEETING}"

echo "[stage] reduced canonical freeze for ${REDUCED_MEETING} (debug_max_utterances=${REDUCED_DEBUG_UTTS})"

"$PY" scripts/run_meeting_pipeline.py \
  --input "$INPUT_DIR/${REDUCED_MEETING}_parsed.json" \
  --output-dir "$KSTUDY_BASE/reduced/${REDUCED_MEETING}/canonical_seed_k${K_REF}/${REDUCED_MEETING}" \
  --shared-prep-dir "$CANONICAL_REDUCED_PREP" \
  --participants all \
  --delegate-variant capa \
  --model "$MODEL" \
  --capa-N "$N" \
  --capa-k "$K_REF" \
  --freeze-anchor-inventory \
  --anchor-seed-k "$K_REF" \
  --cap-meeting "$CAP_MEETING" \
  --cap-per-participant "$CAP_PER_PARTICIPANT" \
  --min-gap "$MIN_GAP" \
  --ctx-strict "$CTX_STRICT" \
  --rel-strict "$REL_STRICT" \
  --ctx-loose "$CTX_LOOSE" \
  --rel-loose "$REL_LOOSE" \
  --debug-max-utterances "$REDUCED_DEBUG_UTTS" \
  --skip-simulation \
  --skip-evaluation

echo "[done] reduced canonical prep root: $CANONICAL_REDUCED_PREP"

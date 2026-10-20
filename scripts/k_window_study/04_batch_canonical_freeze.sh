#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
print_config

echo "[stage] canonical batch freeze for first ${BATCH_LIMIT} meetings"

MEETING_FILES=()
while IFS= read -r meeting_path; do
  MEETING_FILES+=("$(basename "$meeting_path")")
done < <(find "$INPUT_DIR" -maxdepth 1 -type f -name '*_parsed.json' | sort)
if [[ "$BATCH_LIMIT" =~ ^[0-9]+$ ]] && [[ "$BATCH_LIMIT" -gt 0 ]] && [[ "${#MEETING_FILES[@]}" -gt "$BATCH_LIMIT" ]]; then
  MEETING_FILES=("${MEETING_FILES[@]:0:$BATCH_LIMIT}")
fi

for MEETING_FILE in "${MEETING_FILES[@]}"; do
  MEETING="${MEETING_FILE%_parsed.json}"
  "$PY" src/data_pipeline.py \
    --input "$INPUT_DIR/${MEETING}_parsed.json" \
    --output "$KSTUDY_BASE/batch/canonical_seed_k${K_REF}/_prep/${MEETING}" \
    --meeting-id "$MEETING" \
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
done

echo "[done] canonical batch prep root: $KSTUDY_BASE/batch/canonical_seed_k${K_REF}/_prep"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VARIANTS="capa"
DEFAULT_MODEL="gpt-4o-openai"
DEFAULT_PARTICIPANTS="all"
DEFAULT_CAPA_N="20"
DEFAULT_CAPA_K="5"
DEFAULT_CAP_MEETING="20"
DEFAULT_CAP_PER_PARTICIPANT="5"
DEFAULT_MIN_GAP="0"
DEFAULT_CTX_STRICT="0.75"
DEFAULT_REL_STRICT="0.75"
DEFAULT_CTX_LOOSE="0.65"
DEFAULT_REL_LOOSE="0.65"
DEFAULT_RAW_CONTEXT_N="50"
DEFAULT_ANCHOR_SEED_K="5"

MODE=""
INPUT_PATH=""
INPUT_DIRS=""
INPUT_DIR_ARRAY=()
OUTPUT_ROOT=""
SHARED_PREP_ROOT=""
STATIC_PREP_ROOT=""
EIM_CACHE_ROOT=""
ANCHOR_INVENTORY_ROOT=""
MEETINGS=""
SKIP=""
LIMIT=""
PARTICIPANTS="${DEFAULT_PARTICIPANTS}"
VARIANTS="${DEFAULT_VARIANTS}"
MODEL="${DEFAULT_MODEL}"
CAPA_N="${DEFAULT_CAPA_N}"
CAPA_K="${DEFAULT_CAPA_K}"
CAP_MEETING="${DEFAULT_CAP_MEETING}"
CAP_PER_PARTICIPANT="${DEFAULT_CAP_PER_PARTICIPANT}"
MIN_GAP="${DEFAULT_MIN_GAP}"
CTX_STRICT="${DEFAULT_CTX_STRICT}"
REL_STRICT="${DEFAULT_REL_STRICT}"
CTX_LOOSE="${DEFAULT_CTX_LOOSE}"
REL_LOOSE="${DEFAULT_REL_LOOSE}"
RAW_CONTEXT_N="${DEFAULT_RAW_CONTEXT_N}"
DEBUG_MAX_UTTERANCES=""
ANCHOR_SEED_K="${DEFAULT_ANCHOR_SEED_K}"
PYTHON_BIN="${PYTHON:-python}"
RUN_RESUME="1"
DRY_RUN="0"
FORCE="0"
NO_RECALIBRATION="0"
FREEZE_ANCHOR_INVENTORY="0"
SKIP_GENERATION="0"
SKIP_SIMULATION="0"
SKIP_EVALUATION="0"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_experiment_variants.sh single --input <meeting_json> --output-root <root> [options]
  scripts/run_experiment_variants.sh batch  --input-dirs <dir1,dir2> --output-root <root> [options]

Description:
  Wrapper around scripts/run_meeting_pipeline.py and scripts/run_batch_meetings.py
  for running one or more delegate variants with consistent defaults.

Modes:
  single   Run one meeting across one or more variants.
  batch    Run one or more input directories across one or more variants.

Default variants:
  capa

Examples:
  scripts/run_experiment_variants.sh single \
    --input data/ami/test/Meeting32test_parsed.json \
    --output-root results \
    --variants capa

  scripts/run_experiment_variants.sh single \
    --input data/ami/test/Meeting33test_parsed.json \
    --output-root results/smoke \
    --variants capa \
    --debug-max-utterances 150

  scripts/run_experiment_variants.sh batch \
    --input-dirs data/ami/test,data/ami/val,data/ami/train \
    --output-root results/ami \
    --variants capa,mono-raw,mono-raw-reflection \
    --resume

Core options:
  --input PATH                     Single-meeting transcript JSON
  --input-dir DIR                  Batch input directory with *_parsed.json files; can be repeated
  --input-dirs CSV                 Comma-separated batch input directories
  --output-root DIR                Root output directory
  --variants CSV                   Variants to run: capa,raw-modular,raw-modular-fair,mono-raw,mono-raw-reflection,mono-raw-reflection-v2,all
  --participants CSV               Participant list or 'all' (default: all)
  --model NAME                     Model to use (default: gpt-4o-openai)

Common experiment knobs:
  --capa-N INT                     CAPA window size N (default: 20)
  --capa-k INT                     CAPA window size k (default: 5)
  --cap-meeting INT                Episode cap per meeting (default: 20)
  --cap-per-participant INT        Episode cap per participant (default: 5)
  --min-gap INT                    Min cue gap (default: 0)
  --ctx-strict FLOAT               Strict context threshold (default: 0.75)
  --rel-strict FLOAT               Strict relevance threshold (default: 0.75)
  --ctx-loose FLOAT                Loose context threshold (default: 0.65)
  --rel-loose FLOAT                Loose relevance threshold (default: 0.65)
  --raw-context-N INT              Raw baseline transcript window (default: 50)

Single-mode extras:
  --shared-prep-root DIR           Prep root; defaults to <output-root>/_prep
  --debug-max-utterances INT       Debug-only transcript truncation cap
  --force                          Force rerun selected meetings
  --dry-run                        Print the single or batch commands without executing
  --skip-generation                Single mode: skip Stage 1 and reuse existing prep
  --skip-simulation                Single mode: skip Stage 2 and reuse existing sims
  --skip-evaluation                Single mode: skip Stage 3

Batch-mode extras:
  --meetings CSV                   Include only these meetings
  --skip CSV                       Exclude exact meeting IDs, inclusive ranges (<start>..<end>), or glob patterns
  --limit INT                      Limit to first N meetings
  --resume / --no-resume           Resume batch runs (default: resume)
  --dry-run                        Print batch commands without executing

Advanced prep controls:
  --freeze-anchor-inventory        Freeze anchor inventory before k rebuild
  --anchor-inventory-root DIR      Root with frozen anchor inventories
  --anchor-seed-k INT              Seed k used for frozen anchor inventory
  --static-prep-root DIR           Reusable prep root with profiles/JIT
  --eim-cache-root DIR             Shared EIM cache root

CAPA-only:
  --no-recalibration               Disable recalibration for capa

Interpreter:
  --python BIN                     Python interpreter to invoke (default: $PYTHON, else python)

Notes:
  - single mode is resume-by-default because run_meeting_pipeline.py is resume-by-default
  - batch mode defaults to --resume
  - use --variants all to run capa,raw-modular,raw-modular-fair,mono-raw,mono-raw-reflection,mono-raw-reflection-v2
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

log() {
  echo "[INFO] $*"
}

abs_path() {
  local path="$1"
  "${PYTHON_BIN}" - "$path" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

# Portable to bash 3.2 (macOS default): no namerefs or associative arrays, and
# empty arrays are expanded with ${arr[@]+"${arr[@]}"} so `set -u` does not trip.
trim_csv_to_array() {
  local raw="$1"
  local out_name="$2"
  local items=()
  local item
  IFS=',' read -r -a items <<< "$raw"
  for item in ${items[@]+"${items[@]}"}; do
    item="$(echo "${item}" | xargs)"
    if [[ -n "${item}" ]]; then
      eval "${out_name}+=(\"\${item}\")"
    fi
  done
}

extract_meeting_id() {
  local input_path="$1"
  local base
  base="$(basename "${input_path}")"
  base="${base%.json}"
  base="${base%_parsed}"
  echo "${base}"
}

normalize_variants() {
  local raw="$1"
  if [[ "${raw}" == "all" ]]; then
    echo "capa,raw-modular,raw-modular-fair,mono-raw,mono-raw-reflection,mono-raw-reflection-v2"
    return
  fi
  echo "${raw}"
}

validate_variant() {
  local variant="$1"
  case "${variant}" in
    capa|raw-modular|raw-modular-fair|mono-raw|mono-raw-reflection|mono-raw-reflection-v2) ;;
    *) die "Unsupported variant: ${variant}" ;;
  esac
}

ensure_repo_root() {
  cd "${REPO_ROOT}"
}

resolve_meeting_anchor_inventory_dir() {
  local root="$1"
  local meeting_id="$2"
  if [[ -z "${root}" ]]; then
    echo ""
    return
  fi
  if [[ -f "${root}" ]]; then
    echo "${root}"
    return
  fi
  if [[ -d "${root}/${meeting_id}" ]]; then
    echo "${root}/${meeting_id}"
    return
  fi
  echo "${root}"
}

resolve_meeting_static_prep_dir() {
  local root="$1"
  local meeting_id="$2"
  if [[ -z "${root}" ]]; then
    echo ""
    return
  fi
  if [[ -d "${root}/user_profiles" || -d "${root}/jit_briefings" || "$(basename "${root}")" == "${meeting_id}" ]]; then
    echo "${root}"
    return
  fi
  echo "${root}/${meeting_id}"
}

resolve_meeting_eim_cache_dir() {
  local root="$1"
  local meeting_id="$2"
  if [[ -z "${root}" ]]; then
    echo ""
    return
  fi
  if [[ -d "${root}/window_vectors" || -d "${root}/history_checkpoints" || "$(basename "${root}")" == "${meeting_id}" ]]; then
    echo "${root}"
    return
  fi
  echo "${root}/${meeting_id}"
}

normalize_batch_input_dirs() {
  BATCH_INPUT_DIRS=()
  local raw_inputs=()
  local resolved existing duplicate

  if [[ -n "${INPUT_DIRS}" ]]; then
    trim_csv_to_array "${INPUT_DIRS}" raw_inputs
  fi
  for input_dir in ${INPUT_DIR_ARRAY[@]+"${INPUT_DIR_ARRAY[@]}"}; do
    input_dir="$(echo "${input_dir}" | xargs)"
    if [[ -n "${input_dir}" ]]; then
      raw_inputs+=("${input_dir}")
    fi
  done

  for input_dir in ${raw_inputs[@]+"${raw_inputs[@]}"}; do
    resolved="$(abs_path "${input_dir}")"
    [[ -d "${resolved}" ]] || die "Input directory not found: ${resolved}"
    duplicate=0
    for existing in ${BATCH_INPUT_DIRS[@]+"${BATCH_INPUT_DIRS[@]}"}; do
      if [[ "${existing}" == "${resolved}" ]]; then
        duplicate=1
        break
      fi
    done
    if [[ "${duplicate}" -eq 0 ]]; then
      BATCH_INPUT_DIRS+=("${resolved}")
    fi
  done
}

run_single_variant() {
  local variant="$1"
  local input_abs="$2"
  local output_root_abs="$3"
  local prep_root_abs="$4"
  local meeting_id="$5"

  local output_dir="${output_root_abs}/${variant}/${meeting_id}"
  local prep_dir="${prep_root_abs}/${meeting_id}"
  local meeting_anchor_inventory_dir
  local meeting_static_prep_dir
  local meeting_eim_cache_dir

  meeting_anchor_inventory_dir="$(resolve_meeting_anchor_inventory_dir "${ANCHOR_INVENTORY_ROOT}" "${meeting_id}")"
  meeting_static_prep_dir="$(resolve_meeting_static_prep_dir "${STATIC_PREP_ROOT}" "${meeting_id}")"
  meeting_eim_cache_dir="$(resolve_meeting_eim_cache_dir "${EIM_CACHE_ROOT}" "${meeting_id}")"

  mkdir -p "${output_dir}" "${prep_dir}"

  local cmd=(
    "${PYTHON_BIN}" "scripts/run_meeting_pipeline.py"
    "--input" "${input_abs}"
    "--output-dir" "${output_dir}"
    "--shared-prep-dir" "${prep_dir}"
    "--participants" "${PARTICIPANTS}"
    "--delegate-variant" "${variant}"
    "--model" "${MODEL}"
    "--capa-N" "${CAPA_N}"
    "--capa-k" "${CAPA_K}"
    "--cap-meeting" "${CAP_MEETING}"
    "--cap-per-participant" "${CAP_PER_PARTICIPANT}"
    "--min-gap" "${MIN_GAP}"
    "--ctx-strict" "${CTX_STRICT}"
    "--rel-strict" "${REL_STRICT}"
    "--ctx-loose" "${CTX_LOOSE}"
    "--rel-loose" "${REL_LOOSE}"
    "--raw-context-N" "${RAW_CONTEXT_N}"
  )

  if [[ "${FREEZE_ANCHOR_INVENTORY}" == "1" ]]; then
    cmd+=("--freeze-anchor-inventory")
  fi
  if [[ -n "${meeting_anchor_inventory_dir}" ]]; then
    cmd+=("--anchor-inventory" "${meeting_anchor_inventory_dir}")
  fi
  if [[ -n "${ANCHOR_SEED_K}" ]]; then
    cmd+=("--anchor-seed-k" "${ANCHOR_SEED_K}")
  fi
  if [[ -n "${meeting_static_prep_dir}" ]]; then
    cmd+=("--static-prep-source-dir" "${meeting_static_prep_dir}")
  fi
  if [[ -n "${meeting_eim_cache_dir}" ]]; then
    cmd+=("--eim-cache-dir" "${meeting_eim_cache_dir}")
  fi
  if [[ -n "${DEBUG_MAX_UTTERANCES}" ]]; then
    cmd+=("--debug-max-utterances" "${DEBUG_MAX_UTTERANCES}")
  fi
  if [[ "${NO_RECALIBRATION}" == "1" && "${variant}" == "capa" ]]; then
    cmd+=("--no-recalibration")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    cmd+=("--force")
  fi
  if [[ "${SKIP_GENERATION}" == "1" ]]; then
    cmd+=("--skip-generation")
  fi
  if [[ "${SKIP_SIMULATION}" == "1" ]]; then
    cmd+=("--skip-simulation")
  fi
  if [[ "${SKIP_EVALUATION}" == "1" ]]; then
    cmd+=("--skip-evaluation")
  fi

  log "Running single meeting '${meeting_id}' for variant '${variant}'"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "${cmd[@]}"
}

run_batch_variant() {
  local variant="$1"
  local output_root_abs="$2"
  local prep_root_abs="$3"

  local cmd=(
    "${PYTHON_BIN}" "scripts/run_batch_meetings.py"
    "--output-dir" "${output_root_abs}"
    "--shared-prep-root" "${prep_root_abs}"
    "--participants" "${PARTICIPANTS}"
    "--delegate-variant" "${variant}"
    "--model" "${MODEL}"
    "--capa-N" "${CAPA_N}"
    "--capa-k" "${CAPA_K}"
    "--cap-meeting" "${CAP_MEETING}"
    "--cap-per-participant" "${CAP_PER_PARTICIPANT}"
    "--min-gap" "${MIN_GAP}"
    "--ctx-strict" "${CTX_STRICT}"
    "--rel-strict" "${REL_STRICT}"
    "--ctx-loose" "${CTX_LOOSE}"
    "--rel-loose" "${REL_LOOSE}"
    "--raw-context-N" "${RAW_CONTEXT_N}"
  )

  [[ ${#BATCH_INPUT_DIRS[@]} -gt 0 ]] || die "No batch input directories were normalized"
  cmd+=("--input-dirs" "$(join_by_comma "${BATCH_INPUT_DIRS[@]}")")

  if [[ -n "${MEETINGS}" ]]; then
    cmd+=("--meetings" "${MEETINGS}")
  fi
  if [[ -n "${SKIP}" ]]; then
    cmd+=("--skip" "${SKIP}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    cmd+=("--limit" "${LIMIT}")
  fi
  if [[ "${FREEZE_ANCHOR_INVENTORY}" == "1" ]]; then
    cmd+=("--freeze-anchor-inventory")
  fi
  if [[ -n "${ANCHOR_INVENTORY_ROOT}" ]]; then
    cmd+=("--anchor-inventory-root" "${ANCHOR_INVENTORY_ROOT}")
  fi
  if [[ -n "${ANCHOR_SEED_K}" ]]; then
    cmd+=("--anchor-seed-k" "${ANCHOR_SEED_K}")
  fi
  if [[ -n "${STATIC_PREP_ROOT}" ]]; then
    cmd+=("--static-prep-root" "${STATIC_PREP_ROOT}")
  fi
  if [[ -n "${EIM_CACHE_ROOT}" ]]; then
    cmd+=("--eim-cache-root" "${EIM_CACHE_ROOT}")
  fi
  if [[ "${RUN_RESUME}" == "1" ]]; then
    cmd+=("--resume")
  fi
  if [[ "${FORCE}" == "1" ]]; then
    cmd+=("--force")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=("--dry-run")
  fi
  if [[ "${NO_RECALIBRATION}" == "1" && "${variant}" == "capa" ]]; then
    cmd+=("--no-recalibration")
  fi
  if [[ "${SKIP_GENERATION}" == "1" ]]; then
    cmd+=("--skip-generation")
  fi
  if [[ "${SKIP_SIMULATION}" == "1" ]]; then
    cmd+=("--skip-simulation")
  fi
  if [[ "${SKIP_EVALUATION}" == "1" ]]; then
    cmd+=("--skip-evaluation")
  fi

  log "Running batch for variant '${variant}'"
  printf '  %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

case "${MODE}" in
  single|batch) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    die "First argument must be 'single' or 'batch'"
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_PATH="$2"; shift 2 ;;
    --input-dir) INPUT_DIR_ARRAY+=("$2"); shift 2 ;;
    --input-dirs) INPUT_DIRS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --shared-prep-root) SHARED_PREP_ROOT="$2"; shift 2 ;;
    --static-prep-root) STATIC_PREP_ROOT="$2"; shift 2 ;;
    --eim-cache-root) EIM_CACHE_ROOT="$2"; shift 2 ;;
    --anchor-inventory-root) ANCHOR_INVENTORY_ROOT="$2"; shift 2 ;;
    --meetings) MEETINGS="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --participants) PARTICIPANTS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --capa-N) CAPA_N="$2"; shift 2 ;;
    --capa-k) CAPA_K="$2"; shift 2 ;;
    --cap-meeting) CAP_MEETING="$2"; shift 2 ;;
    --cap-per-participant) CAP_PER_PARTICIPANT="$2"; shift 2 ;;
    --min-gap) MIN_GAP="$2"; shift 2 ;;
    --ctx-strict) CTX_STRICT="$2"; shift 2 ;;
    --rel-strict) REL_STRICT="$2"; shift 2 ;;
    --ctx-loose) CTX_LOOSE="$2"; shift 2 ;;
    --rel-loose) REL_LOOSE="$2"; shift 2 ;;
    --raw-context-N) RAW_CONTEXT_N="$2"; shift 2 ;;
    --debug-max-utterances) DEBUG_MAX_UTTERANCES="$2"; shift 2 ;;
    --anchor-seed-k) ANCHOR_SEED_K="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --resume) RUN_RESUME="1"; shift ;;
    --no-resume) RUN_RESUME="0"; shift ;;
    --dry-run) DRY_RUN="1"; shift ;;
    --force) FORCE="1"; shift ;;
    --skip-generation) SKIP_GENERATION="1"; shift ;;
    --skip-simulation) SKIP_SIMULATION="1"; shift ;;
    --skip-evaluation) SKIP_EVALUATION="1"; shift ;;
    --no-recalibration) NO_RECALIBRATION="1"; shift ;;
    --freeze-anchor-inventory) FREEZE_ANCHOR_INVENTORY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

ensure_repo_root

[[ -n "${OUTPUT_ROOT}" ]] || die "--output-root is required"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  die "Python interpreter not found: ${PYTHON_BIN}"
fi

VARIANTS="$(normalize_variants "${VARIANTS}")"
IFS=',' read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
[[ ${#VARIANT_ARRAY[@]} -gt 0 ]] || die "No variants requested"

for i in "${!VARIANT_ARRAY[@]}"; do
  VARIANT_ARRAY[$i]="$(echo "${VARIANT_ARRAY[$i]}" | xargs)"
  validate_variant "${VARIANT_ARRAY[$i]}"
done

OUTPUT_ROOT="$(abs_path "${OUTPUT_ROOT}")"
SHARED_PREP_ROOT="${SHARED_PREP_ROOT:-${OUTPUT_ROOT}/_prep}"
SHARED_PREP_ROOT="$(abs_path "${SHARED_PREP_ROOT}")"

if [[ -n "${STATIC_PREP_ROOT}" ]]; then
  STATIC_PREP_ROOT="$(abs_path "${STATIC_PREP_ROOT}")"
fi
if [[ -n "${EIM_CACHE_ROOT}" ]]; then
  EIM_CACHE_ROOT="$(abs_path "${EIM_CACHE_ROOT}")"
fi
if [[ -n "${ANCHOR_INVENTORY_ROOT}" ]]; then
  ANCHOR_INVENTORY_ROOT="$(abs_path "${ANCHOR_INVENTORY_ROOT}")"
fi

mkdir -p "${OUTPUT_ROOT}" "${SHARED_PREP_ROOT}"

case "${MODE}" in
  single)
    [[ -n "${INPUT_PATH}" ]] || die "--input is required in single mode"
    [[ -z "${INPUT_DIRS}" && ${#INPUT_DIR_ARRAY[@]} -eq 0 ]] || die "--input-dir/--input-dirs are not valid in single mode"
    INPUT_PATH="$(abs_path "${INPUT_PATH}")"
    [[ -f "${INPUT_PATH}" ]] || die "Input file not found: ${INPUT_PATH}"
    MEETING_ID="$(extract_meeting_id "${INPUT_PATH}")"

    log "Single-meeting run"
    log "Meeting: ${MEETING_ID}"
    log "Variants: ${VARIANTS}"
    log "Output root: ${OUTPUT_ROOT}"
    log "Shared prep root: ${SHARED_PREP_ROOT}"
    log "Model: ${MODEL}"

    for variant in "${VARIANT_ARRAY[@]}"; do
      run_single_variant "${variant}" "${INPUT_PATH}" "${OUTPUT_ROOT}" "${SHARED_PREP_ROOT}" "${MEETING_ID}"
    done
    ;;
  batch)
    [[ -n "${INPUT_DIRS}" || ${#INPUT_DIR_ARRAY[@]} -gt 0 ]] || die "--input-dir or --input-dirs is required in batch mode"
    [[ -z "${INPUT_PATH}" ]] || die "--input is not valid in batch mode"
    [[ -z "${DEBUG_MAX_UTTERANCES}" ]] || die "--debug-max-utterances is only supported in single mode"
    normalize_batch_input_dirs
    [[ ${#BATCH_INPUT_DIRS[@]} -gt 0 ]] || die "No valid input directories were provided"

    log "Batch run"
    log "Input dirs: $(join_by_comma "${BATCH_INPUT_DIRS[@]}")"
    log "Variants: ${VARIANTS}"
    log "Output root: ${OUTPUT_ROOT}"
    log "Shared prep root: ${SHARED_PREP_ROOT}"
    log "Model: ${MODEL}"

    for variant in "${VARIANT_ARRAY[@]}"; do
      run_batch_variant "${variant}" "${OUTPUT_ROOT}" "${SHARED_PREP_ROOT}"
    done
    ;;
esac

log "Done"

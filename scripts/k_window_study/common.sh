#!/usr/bin/env bash
set -euo pipefail

KSTUDY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$KSTUDY_SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PY:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PY="$REPO_ROOT/.venv/bin/python"
  else
    PY="$(command -v python3 || echo python3)"
  fi
fi
INPUT_DIR="${INPUT_DIR:-data/ami/test}"
MODEL="${MODEL:-gpt-4o-openai}"

N="${N:-20}"
K_REF="${K_REF:-9}"
K_VALUES_CSV="${K_VALUES_CSV:-3,5,7,9}"

CAP_MEETING="${CAP_MEETING:-20}"
CAP_PER_PARTICIPANT="${CAP_PER_PARTICIPANT:-5}"
MIN_GAP="${MIN_GAP:-0}"

CTX_STRICT="${CTX_STRICT:-0.75}"
REL_STRICT="${REL_STRICT:-0.75}"
CTX_LOOSE="${CTX_LOOSE:-0.60}"
REL_LOOSE="${REL_LOOSE:-0.60}"

KSTUDY_BASE="${KSTUDY_BASE:-results/k_window_study}"

REDUCED_MEETING="${REDUCED_MEETING:-Meeting33test}"
REDUCED_DEBUG_UTTS="${REDUCED_DEBUG_UTTS:-120}"

FULL_MEETING="${FULL_MEETING:-Meeting33test}"

BATCH_LIMIT="${BATCH_LIMIT:-20}"

IFS=',' read -r -a K_VALUES <<< "$K_VALUES_CSV"

function ensure_python() {
  if [[ ! -x "$PY" ]]; then
    echo "[ERROR] Python executable not found or not executable: $PY" >&2
    exit 1
  fi
}

function ensure_path_exists() {
  local target="$1"
  local label="${2:-path}"
  if [[ ! -e "$target" ]]; then
    echo "[ERROR] Missing ${label}: ${target}" >&2
    exit 1
  fi
}

function print_config() {
  cat <<EOF
[k-study]
repo_root=$REPO_ROOT
python=$PY
input_dir=$INPUT_DIR
model=$MODEL
N=$N
k_ref=$K_REF
k_values=$K_VALUES_CSV
results_base=$KSTUDY_BASE
EOF
}

ensure_python

#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: engineer first 100 rows from merged_transactions.csv
# and send to Triton for all three models.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_CSV="$REPO_DIR/datasets/merged_transactions.csv"
ARTIFACTS="$SCRIPT_DIR/artifacts/engineering_artifacts.json"
TRITON_URL="${TRITON_URL:-localhost:8000}"
ROWS="${ROWS:-10000}"

# Resolve Python interpreter (prefer python3)
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "No Python interpreter found. Install python3 or set PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

if [[ ! -f "$ARTIFACTS" ]]; then
  echo "Engineering artifacts not found: $ARTIFACTS" >&2
  echo "Fitting artifacts from $INPUT_CSV ..." >&2
  "$PYTHON_BIN" "$SCRIPT_DIR/fit_engineering_artifacts.py" "$INPUT_CSV" --out "$ARTIFACTS"
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/engineer_and_infer.py" \
  --input "$INPUT_CSV" \
  -n "$ROWS" \
  --model all \
  --artifacts "$ARTIFACTS" \
  --triton-url "$TRITON_URL"

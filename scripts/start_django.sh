#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"

if [[ -x "$BACKEND_DIR/venv_new/bin/python" ]]; then
  PYTHON_BIN="$BACKEND_DIR/venv_new/bin/python"
elif [[ -x "$BACKEND_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$BACKEND_DIR/venv/bin/python"
elif [[ -x "$BACKEND_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "$BACKEND_DIR"
exec "$PYTHON_BIN" manage.py runserver 0.0.0.0:8000 --noreload

#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_DIR/backend"

if [[ -x "$BACKEND_DIR/venv_new/bin/celery" ]]; then
  CELERY_BIN="$BACKEND_DIR/venv_new/bin/celery"
elif [[ -x "$BACKEND_DIR/venv/bin/celery" ]]; then
  CELERY_BIN="$BACKEND_DIR/venv/bin/celery"
elif [[ -x "$BACKEND_DIR/.venv/bin/celery" ]]; then
  CELERY_BIN="$BACKEND_DIR/.venv/bin/celery"
else
  CELERY_BIN="${CELERY_BIN:-celery}"
fi

cd "$BACKEND_DIR"
exec "$CELERY_BIN" -A quantvision worker --loglevel=info --pool=solo

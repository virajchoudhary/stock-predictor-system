#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

ensure_redis() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker not found. Skipping Redis startup."
    return 0
  fi

  if docker ps -a --format '{{.Names}}' | grep -qx "redis"; then
    docker start redis >/dev/null 2>&1 || true
  else
    docker run -d -p 6379:6379 --name redis redis >/dev/null 2>&1 || true
  fi
}

launch_background() {
  local name="$1"
  local script_path="$2"
  local log_file="$LOG_DIR/${name}.log"
  bash "$script_path" >"$log_file" 2>&1 &
  echo "$name started in background. Log: $log_file"
}

launch_mac_terminal() {
  local title="$1"
  local script_path="$2"
  osascript -e "tell application \"Terminal\" to do script \"cd \\\"$REPO_DIR\\\"; printf '\\\\e]1;$title\\\\a'; bash \\\"$script_path\\\"\"" >/dev/null
}

echo "Starting QuantVis services..."
ensure_redis

if [[ "$(uname -s)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
  launch_mac_terminal "Django" "$REPO_DIR/scripts/start_django.sh"
  launch_mac_terminal "Celery Worker" "$REPO_DIR/scripts/start_worker.sh"
  launch_mac_terminal "Celery Beat" "$REPO_DIR/scripts/start_beat.sh"
  launch_mac_terminal "Streamlit" "$REPO_DIR/scripts/start_streamlit.sh"
  (sleep 8; open http://localhost:8501 >/dev/null 2>&1 || true) &
else
  launch_background "django" "$REPO_DIR/scripts/start_django.sh"
  launch_background "worker" "$REPO_DIR/scripts/start_worker.sh"
  launch_background "beat" "$REPO_DIR/scripts/start_beat.sh"
  launch_background "streamlit" "$REPO_DIR/scripts/start_streamlit.sh"
  echo "Open http://localhost:8501 once Streamlit is ready."
fi

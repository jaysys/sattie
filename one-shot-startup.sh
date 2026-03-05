#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="$ROOT_DIR/run/satti.pid"
LOG_FILE="$ROOT_DIR/logs/satti.log"
PORT="${PORT:-6001}"
HOST="${HOST:-0.0.0.0}"

if [[ ! -d "$ROOT_DIR/venv" ]]; then
  echo "[startup] venv not found. run ./one-shot-setup.sh first."
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[startup] server already running (pid=$OLD_PID)"
    echo "[startup] log: $LOG_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

source "$ROOT_DIR/venv/bin/activate"
mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/run"

echo "[startup] starting uvicorn app.satti_api:app on ${HOST}:${PORT}"
nohup "$ROOT_DIR/venv/bin/uvicorn" app.satti_api:app \
  --host "$HOST" \
  --port "$PORT" \
  >"$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

sleep 1
if kill -0 "$NEW_PID" 2>/dev/null; then
  echo "[startup] started (pid=$NEW_PID)"
  echo "[startup] url: http://127.0.0.1:${PORT}/ui"
  echo "[startup] log: $LOG_FILE"
else
  echo "[startup] failed to start. check log:"
  tail -n 80 "$LOG_FILE" || true
  rm -f "$PID_FILE"
  exit 1
fi

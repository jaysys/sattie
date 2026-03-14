#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="$ROOT_DIR/run/sattie.pid"
LOG_FILE="$ROOT_DIR/logs/sattie.log"
PORT="${PORT:-6001}"
HOST="${HOST:-0.0.0.0}"
RELOAD="${RELOAD:-0}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

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

if command -v lsof >/dev/null 2>&1; then
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[startup] port ${PORT} is already in use"
    exit 1
  fi
fi

echo "[startup] starting uvicorn app.sattie_api:app on ${HOST}:${PORT}"
UVICORN_ARGS=(
  app.sattie_api:app
  --host "$HOST"
  --port "$PORT"
)

if [[ "$RELOAD" == "1" ]]; then
  UVICORN_ARGS+=(
    --reload
    --reload-dir app
  )
fi

nohup "$ROOT_DIR/venv/bin/uvicorn" "${UVICORN_ARGS[@]}" >"$LOG_FILE" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

for _ in {1..20}; do
  if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "[startup] failed to start. check log:"
    tail -n 80 "$LOG_FILE" || true
    rm -f "$PID_FILE"
    exit 1
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      echo "[startup] started (pid=$NEW_PID)"
      echo "[startup] url: http://127.0.0.1:${PORT}/ui"
      echo "[startup] log: $LOG_FILE"
      exit 0
    fi
  fi
  sleep 0.5
done

echo "[startup] process is running but health check did not succeed. check log:"
tail -n 80 "$LOG_FILE" || true
rm -f "$PID_FILE"
exit 1

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PID_FILE="$ROOT_DIR/run/satti.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "[stop] pid file not found. server may already be stopped."
  exit 0
fi

PID="$(cat "$PID_FILE" || true)"
if [[ -z "${PID:-}" ]]; then
  echo "[stop] invalid pid file. removing."
  rm -f "$PID_FILE"
  exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "[stop] process not running (pid=$PID). cleaning pid file."
  rm -f "$PID_FILE"
  exit 0
fi

echo "[stop] stopping server (pid=$PID)"
kill "$PID" 2>/dev/null || true

for _ in {1..10}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "[stop] stopped"
    exit 0
  fi
  sleep 0.3
done

echo "[stop] force kill (pid=$PID)"
kill -9 "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "[stop] stopped"

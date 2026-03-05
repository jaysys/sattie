#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] project root: $ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "[setup] creating virtual environment: venv"
  "$PYTHON_BIN" -m venv venv
else
  echo "[setup] venv already exists"
fi

source "$ROOT_DIR/venv/bin/activate"

echo "[setup] upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
  echo "[setup] installing requirements"
  pip install -r "$ROOT_DIR/requirements.txt"
fi

mkdir -p "$ROOT_DIR/logs" "$ROOT_DIR/run" "$ROOT_DIR/samples" "$ROOT_DIR/mock_store"

echo "[setup] done"
echo "[setup] next: ./one-shot-startup.sh"

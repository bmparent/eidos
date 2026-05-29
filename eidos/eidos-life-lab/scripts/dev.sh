#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
PYTHON_BIN="$BACKEND_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Eidos Life Lab backend on http://127.0.0.1:8787"
(cd "$BACKEND_DIR" && "$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 8787 --reload) &
BACKEND_PID=$!

echo "Starting Eidos Life Lab frontend on http://127.0.0.1:5173"
(cd "$FRONTEND_DIR" && npm run dev) &
FRONTEND_PID=$!

wait

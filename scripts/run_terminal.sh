#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d venv ]]; then
  echo "Missing venv. Create the QuantTerm Python environment first." >&2
  exit 1
fi

source venv/bin/activate

if ! python -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "Installing terminal API dependencies…"
  python -m pip install 'fastapi>=0.115.0' 'uvicorn>=0.30.0'
fi

if [[ ! -d frontend/node_modules ]]; then
  echo "Installing terminal frontend dependencies…"
  (cd frontend && npm install)
fi

cleanup() {
  [[ -n "${API_PID:-}" ]] && kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

python -m uvicorn terminal_api:app --host 127.0.0.1 --port 8765 &
API_PID=$!

cd frontend
npm run dev

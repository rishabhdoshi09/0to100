#!/usr/bin/env bash
# Legacy entry name — same product API as run_quantterm.sh (terminal_product_api on :8765).
# terminal_api:app omits scanner-workspace, stock-intelligence, data/ratios, and market routes.
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

echo "[TERMINAL] Starting product API at http://127.0.0.1:8765 (terminal_product_api)…"
python -m uvicorn terminal_product_api:app --host 127.0.0.1 --port 8765 &
API_PID=$!

echo "[TERMINAL] Waiting for /api/health…"
API_READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[TERMINAL] API exited during startup." >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:8765/api/health" >/dev/null 2>&1; then
    API_READY=1
    break
  fi
  sleep 0.5
done
if [[ "$API_READY" != "1" ]]; then
  echo "[TERMINAL] API did not become ready within 120s." >&2
  exit 1
fi
echo "[TERMINAL] API ready. For autonomy + report API use: bash scripts/run_quantterm_complete.sh"

cd frontend
npm run dev

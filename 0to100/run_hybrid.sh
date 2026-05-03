#!/usr/bin/env bash
# ── sq_ai hybrid launcher ────────────────────────────────────────────────
# 1. activate venv (create if missing)
# 2. install deps
# 3. start FastAPI in background (APScheduler runs in-process inside lifespan)
# 4. launch Textual TUI in foreground
# 5. on Ctrl-C, kill FastAPI
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.installed" ]; then
  pip install --upgrade pip
  pip install -r requirements.txt
  touch .venv/.installed
fi

mkdir -p data logs models

if [ ! -f ".env" ]; then
  echo "✗ .env missing – copy .env.template and fill keys" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "▶ starting FastAPI on ${SQ_API_HOST:-127.0.0.1}:${SQ_API_PORT:-8000} …"
uvicorn sq_ai.api.app:app \
  --host "${SQ_API_HOST:-127.0.0.1}" \
  --port "${SQ_API_PORT:-8000}" \
  --log-level warning \
  > logs/api.log 2>&1 &
API_PID=$!
trap 'echo; echo "▶ shutting down (api pid=$API_PID) …"; kill $API_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 30); do
  if curl -fs "http://${SQ_API_HOST:-127.0.0.1}:${SQ_API_PORT:-8000}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "▶ launching Textual TUI – press q to quit"
python -m sq_ai.ui.terminal

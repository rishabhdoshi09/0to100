#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 0to100 / sq_ai – one-command startup
#   1. activates venv (creates if missing)
#   2. installs deps if needed
#   3. starts FastAPI in background (uvicorn, port 8000)
#   4. launches Textual TUI in foreground
#   on Ctrl-C, the API is killed too.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1. venv
if [ ! -d ".venv" ]; then
  echo "▶ creating venv …"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. deps
if [ ! -f ".venv/.installed" ]; then
  echo "▶ installing requirements …"
  pip install --upgrade pip
  pip install -r requirements.txt
  touch .venv/.installed
fi

# 3. dirs
mkdir -p data logs models

# 4. env
if [ ! -f ".env" ]; then
  echo "✗ .env missing – copy .env template and fill keys" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

# 5. start FastAPI in background
echo "▶ starting FastAPI on ${SQ_API_HOST:-127.0.0.1}:${SQ_API_PORT:-8000} …"
uvicorn sq_ai.api.app:app \
  --host "${SQ_API_HOST:-127.0.0.1}" \
  --port "${SQ_API_PORT:-8000}" \
  --log-level warning \
  > logs/api.log 2>&1 &
API_PID=$!
trap 'echo; echo "▶ shutting down (api pid=$API_PID) …"; kill $API_PID 2>/dev/null || true' EXIT INT TERM

# 6. wait for /health
for _ in $(seq 1 30); do
  if curl -fs "http://${SQ_API_HOST:-127.0.0.1}:${SQ_API_PORT:-8000}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

# 7. launch TUI (foreground)
echo "▶ launching Textual TUI – press q to quit"
python -m sq_ai.ui.terminal

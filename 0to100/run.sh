#!/usr/bin/env bash
# ─── sq_ai unified launcher (FastAPI + Streamlit + Textual TUI) ─────────
# 1. activate venv (create if missing)
# 2. install deps
# 3. start FastAPI in background  (APScheduler runs in-process)
# 4. start Streamlit in background (port 8501)
# 5. launch Textual TUI in foreground
# 6. on Ctrl-C, kill both background services
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

mkdir -p data logs models reports

if [ ! -f ".env" ]; then
  if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✓ Created .env from .env.example — please fill in your API keys before proceeding." >&2
    echo "  Required: ANTHROPIC_API_KEY, DEEPSEEK_API_KEY" >&2
    echo "  Optional: KITE_API_KEY + KITE_ACCESS_TOKEN (needed for live trading)" >&2
    echo "  Optional: NEWSAPI_KEY, ALPHA_VANTAGE_KEY" >&2
    echo "" >&2
    echo "  Re-run ./run.sh once you have filled in your keys." >&2
    exit 1
  fi
  echo "✗ .env missing – cp .env.example .env and fill keys" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

API_HOST="${SQ_API_HOST:-127.0.0.1}"
API_PORT="${SQ_API_PORT:-8000}"
ST_PORT="${SQ_STREAMLIT_PORT:-8501}"

echo "▶ FastAPI  → http://${API_HOST}:${API_PORT}"
uvicorn sq_ai.api.app:app --host "$API_HOST" --port "$API_PORT" \
  --log-level warning > logs/api.log 2>&1 &
API_PID=$!

echo "▶ Streamlit → http://${API_HOST}:${ST_PORT}"
streamlit run sq_ai/ui/streamlit_app.py --server.port "$ST_PORT" \
  --server.headless true > logs/streamlit.log 2>&1 &
ST_PID=$!

trap 'echo; echo "▶ shutting down (api=$API_PID streamlit=$ST_PID) …"; \
      kill $API_PID $ST_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 30); do
  if curl -fs "http://${API_HOST}:${API_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "▶ Textual TUI – press q to quit (web UI keeps running until Ctrl-C)"
python -m sq_ai.ui.terminal

# QuantTerm Terminal

This is the dedicated product frontend for QuantTerm. It is intentionally separate from Streamlit.

## Architecture

- `terminal_api.py` projects existing QuantTerm state and forwards a small whitelist of owner controls.
- `frontend/` is a React + TypeScript + Vite application.
- Scanner, paper book, market regime, long-term shortlist and autonomy stores remain authoritative.
- The frontend never submits broker orders and does not run scanners or trading loops.

## Run

One terminal, from the repo root:

```bash
bash scripts/run_desk.sh
```

Then open:

```text
http://127.0.0.1:5173
```

That starts `terminal_product_api` on :8765, the Vite desk on :5173, and autonomy. Zerodha login runs in the same terminal when the daily token is missing.

## Manual development

```bash
source venv/bin/activate
python -m uvicorn terminal_api:app --host 127.0.0.1 --port 8765
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

## Truthfulness rules

- Missing data renders as unavailable, never as a fabricated metric.
- Current fundamentals are not treated as point-in-time research evidence.
- Controls are queued to the autonomy supervisor; the API does not execute jobs directly.
- LIVE broker order submission is outside this frontend.

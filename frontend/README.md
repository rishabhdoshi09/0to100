# QuantTerm Terminal

This is the dedicated product frontend for QuantTerm. Streamlit is not the
product UI and is not started.

## Architecture

- `terminal_api.py` / `terminal_product_api.py` project existing QuantTerm state and enqueue durable market-operations jobs.
- `frontend/` is a React + TypeScript + Vite application — the only product UI.
- Scanner, paper book, market regime, long-term shortlist and autonomy stores remain authoritative.
- The frontend never submits broker orders. Primary clicks enqueue real backend jobs and poll them.

## Run

One terminal, from the repo root:

```bash
bash scripts/run_quantterm_complete.sh
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

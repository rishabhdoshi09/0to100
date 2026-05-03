# Handover prompt — paste this into a fresh Claude chat

Here's a complete, copy-paste-ready briefing. It tells Claude **what was built**,
**where things are**, **what's broken**, and **what's next** — so it can pick up
the work cleanly without re-asking everything.

---

## How to use this

1. Open a new Claude conversation (claude.ai, Claude Code, or any tool that wraps the API).
2. **Attach your repo** if the tool supports it (Claude Code, Projects, file upload).
   If not, push to GitHub and paste the URL.
3. Copy everything between the `=== START HANDOVER ===` and `=== END HANDOVER ===`
   markers below.
4. Paste as your **first message**. Claude will read it, ask 1-2 clarifying questions,
   and continue.

---

```
=== START HANDOVER ===

You are taking over a project called sq_ai — a personal Bloomberg-grade
quant trading + research cockpit built for an 8 GB MacBook Air 2015.
The repo lives at: https://github.com/rishabhdoshi09/0to100
Local checkout:   ~/0to100

Please read the repo before answering. Key files to scan first:
  README.md
  memory/PRD.md            (the running product log – READ THIS FIRST)
  0to100/config.yaml
  0to100/sq_ai/backend/scheduler.py   (the orchestrator)
  0to100/sq_ai/backend/decision.py    (5-min Claude loop)
  0to100/sq_ai/backend/screener.py    (30-min DeepSeek loop)
  0to100/sq_ai/backend/ensemble.py    (>10 % size veto)
  0to100/sq_ai/signals/composite_signal.py   (THE regime fix)
  0to100/sq_ai/api/app.py             (FastAPI routes)
  0to100/sq_ai/ui/streamlit_app.py    (multi-page web UI)

================================================================
WHAT'S ALREADY DONE  (versions v0.1 → v0.3, all green)
================================================================

v0.1 – core cockpit
  • FastAPI :8000 (paper Kite executor, journal.md, signals table)
  • Textual TUI (terminal cockpit, polls API every 2 s)
  • SQLite single-file cache (data/sq_ai.db)
  • APScheduler in-process inside FastAPI lifespan
  • LightGBM inference wrapper (training stays on Colab)
  • Backtester reusing the SAME composite signal as live
  • Risk manager: Kelly + vol-target + 4 % daily-loss kill-switch
                 + 50 % gross-exposure cap

v0.2 – dual-LLM upgrade
  • backend/llm_clients.py
      ClaudeClient   (anthropic SDK)
      DeepSeekClient (openai SDK, base_url=https://api.deepseek.com/v1)
      Both expose generate(prompt, max_tokens=300, temperature=0.2,
                            system=None) and degrade gracefully when
                            the key is missing or has 'REPLACE' in it.
  • backend/screener.py     – 30-min DeepSeek pre-filter (top 10)
  • backend/decision.py     – 5-min Claude loop on those top 10
  • backend/ensemble.py     – when Claude size_pct > 10 %, re-prompt
                              DeepSeek with the SAME prompt; HOLD on
                              disagreement; persist to llm_disagreements
  • backend/universe.py     – kite.instruments('NSE') cached in SQLite,
                              refreshed daily 08:00 IST; YAML fallback
  • config.yaml             – universe + cadences + ensemble threshold

v0.3 – unified Bloomberg + Screener.in + Moneycontrol platform
  • backend/cache.py            – SQLite-backed @cached(prefix, ttl_seconds)
  • backend/financials.py       – Alpha Vantage → yfinance fallback
                                  (ratios, annual P&L/BS/CF, quarterly)
  • backend/analyst_estimates.py – AV EARNINGS_ESTIMATES + yfinance
  • backend/shareholding.py     – yfinance major_holders + 8-q synth
  • backend/corporate_actions.py – dividends + splits
  • backend/earnings_analyzer.py – pdfplumber → Claude → strict-JSON
                                  highlights + guidance (cached forever)
  • backend/screener_engine.py  – declarative JSON filter:
        technical (RSI, SMAs, MACD, volume, ATR, 52W),
        fundamental (P/E, P/B, ROE, D/E, mcap, divyld),
        momentum (1w/1m/3m). Returns ranked & scored matches.
  • backend/stock_research.py   – aggregator for /api/stock/profile/{sym}
  • backend/watchlist.py        – CRUD with .NS normalisation
  • backend/report_scheduler.py – APScheduler 17:30 IST → reportlab PDF
                                  (index snapshot + sector heatmap +
                                  movers + Claude narrative). Cached
                                  narrative TTL 12 h.

  • api/app.py routes (all under /api/*):
        health, portfolio, positions, trades, signals/latest,
        cycle/status, cycle/run, screener (latest/run/presets CRUD),
        disagreements, universe (list + refresh),
        stock/profile|header|technicals|financials|earnings|estimates|
              shareholding|actions|news|peers,
        watchlist (list/add/delete),
        reports (list/generate/download),
        trade (manual BUY/SELL)

  • Streamlit multi-page (port 8501, ui/streamlit_app.py)
        Dashboard / Screener / Stock Research / Portfolio /
        Reports / Settings  – all pages call FastAPI via ui/_api.py

  • run.sh    – boots FastAPI + Streamlit + Textual TUI together
  • run_hybrid.sh – API + TUI only (no web UI)

  • SQLite tables now include:
        prices, trades, signals, daily_equity, screener_results,
        screener_presets, user_watchlist, earnings_calls,
        llm_disagreements, instruments_cache, reports, kv_cache

  • 71 / 71 pytest cases pass. ruff lint-clean. FastAPI smoke green.

================================================================
THE CRITICAL REGIME FIX (preserved through every version)
================================================================
The original repo silently defaulted features['regime'] to 0, so the
LightGBM model never saw the trend regime → the 2022 backtest lost
–7 %. The fix is in TWO places and is reused EVERYWHERE:

  • signals/composite_signal.py::compute_indicators
      now emits 'regime' ∈ {0=down, 1=side, 2=up} from SMA20 vs SMA50
  • signals/composite_signal.py::compute  enforces a hard regime gate:
      'BUY in regime=0' is clamped to HOLD

The same engineering is reused by:
  - backtest/backtester.py
  - backend/decision.py
  - backend/screener_engine.py
  - train/walk_forward.py
  - train/colab_train.ipynb
so backtest = live (no more train/serve skew).

================================================================
KNOWN BUG → fix this FIRST
================================================================
The .env shipped with model name `claude-3-sonnet-20240229`, which
Anthropic deprecated on 2024-07-21 and now returns 404. Our
ClaudeClient swallows the exception and falls back to ML+regime →
it LOOKS LIKE CLAUDE IS NEVER RUNNING.

Fix in .env:
  CLAUDE_MODEL=claude-sonnet-4-5-20250929   # (or haiku-4-5 / opus-4-5)

Verification one-liner from inside ~/0to100:
  source .venv/bin/activate
  python -c "
  import os; from dotenv import load_dotenv; load_dotenv()
  from sq_ai.backend.llm_clients import ClaudeClient
  c = ClaudeClient()
  print('available:', c.available, 'model:', c.model)
  print(c.generate('reply OK', max_tokens=10))
  "
Expect: "OK". If it prints None, the key is wrong / model still bad.

================================================================
ENVIRONMENT (MacBook Air 2015, 8 GB)
================================================================
  • Total resident target: < 500 MB
  • All training (LightGBM / LSTM) runs on Google Colab T4 ONLY.
    The notebook is at sq_ai/train/colab_train.ipynb.
    Output: lgb_trading_model.pkl + feature_names.txt → drop in
            ~/0to100/models/
  • Until that .pkl is dropped, ML score = 0.5 (no info) and the
    composite signal degrades to factor + regime + LLM weights.
  • Paper-trading mode is the default (SQ_PAPER_TRADING=true).
    Flipping to false will route Order objects to real Kite.

================================================================
WHAT'S ON THE BACKLOG (priority order)
================================================================
P0 (do today)
  1. Fix CLAUDE_MODEL in .env (see "KNOWN BUG" above) and confirm
     /api/cycle/status shows used_claude=true after one cycle.
  2. Wire NewsAPI per-symbol into decision.py prompt (key already
     read by data_fetcher.fetch_news; just needs NEWSAPI_KEY in .env).
  3. Drop the trained model into models/ from a fresh Colab run.

P1 (this week)
  4. Hook the screener directly into the daily PDF report — load the
     user's first preset from preset_list(), run screener_engine.run_screener
     on it, embed the top 5 names as the PDF's "Stock of the day" /
     "Stock losing momentum" sections, and have Claude narrate around
     them. ~5 lines in report_scheduler.ReportGenerator.generate.
  5. Liquidity sort in universe.get_active_universe – currently first 500;
     change to turnover-weighted (20-d avg ₹ volume).
  6. Inline stop / target editing for open positions in
     ui/portfolio_page.py (current UI is read-only).
  7. Add /api/equity (date, equity, cash) and a Plotly equity-curve
     chart to ui/dashboard_page.py.

P2 (next 2-4 weeks)
  8. LSTM head on Colab (input: same 8 features + 60-bar sequence),
     blend with LightGBM probability via convex combination.
  9. Auto-retraining trigger: if rolling AUC < 0.55 for 30 trading
     days, fire a Colab webhook (or just notify Telegram).
 10. Multi-broker abstraction (Upstox / Dhan) behind the existing
     Executor interface.
 11. Inline regime label + ML probability inside stock_research_page
     header card.
 12. Telegram bot that posts the daily PDF + every Claude BUY signal.

P3 (research / stretch)
 13. Earnings-call audio → Whisper → transcript → Claude pipeline
     (currently we only handle PDF transcripts).
 14. Sentiment from Twitter/X via free RapidAPI.
 15. SHAP explanations for each LightGBM trade (saved with journal row).
 16. Walk-forward back-tester running nightly on Colab and pushing
     metrics to /api/metrics/walkforward.

================================================================
HOUSE RULES (please follow)
================================================================
  • Single source of truth for indicators is composite_signal.py.
    Do NOT recompute SMA/RSI/ATR/regime anywhere else — import from
    there. The regime gate must stay enforced.
  • Every external network call MUST go through @cached(prefix, ttl)
    in backend/cache.py. Never hammer Alpha Vantage / NewsAPI.
  • Every new endpoint goes under /api/*  and is async unless it
    needs CPU-bound work.
  • Every new feature gets a pytest in tests/  before merge.
    Goal: keep tests green, lint clean (ruff), 8 GB RAM ceiling.
  • ALL training on Colab. The laptop only does inference.
  • Paper-trading is the default. Don't change SQ_PAPER_TRADING in
    code; only the user flips it in .env.
  • Don't introduce React, Postgres, or Docker. Plain Python +
    SQLite + Streamlit + Textual.

================================================================
TASK
================================================================
Start by:
  1. Reading memory/PRD.md and the file list above.
  2. Confirming the regime fix is intact.
  3. Running:  cd ~/0to100 && pytest -q tests/    (must show 71+ passing)
  4. Then propose your plan for P0 items 1–3 in order, ask me any
     clarifying questions, and start coding.

When you finish a task, append a row to memory/PRD.md under
"Tasks done" with today's date, run pytest + ruff, and only then
ask me to review.

=== END HANDOVER ===
```

---

## Tips for the handover to actually stick

- **If the new Claude session can't see your repo**, push first, then in the new
  chat say *"Repo is at github.com/rishabhdoshi09/0to100, please read it before responding."*
- **If you're using Claude Code (the CLI agent)**, run it from inside `~/0to100` so
  it has filesystem access — then paste the handover.
- **If you're using the web app (claude.ai)**, attach `memory/PRD.md` and the README
  as files — that gives Claude the same grounding without needing to fetch GitHub.
- **Keep the file `memory/PRD.md` updated** — that's the running log every future
  Claude session will read first. Treat it like a captain's logbook.

That's it. Paste it, hit send, and your next session continues from exactly where
we stopped — no re-explaining.

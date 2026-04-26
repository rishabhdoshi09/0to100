# SimpleQuant AI — Developer Reference

## Project Layout

```
simplequant/
├── main.py              # CLI entry point (live | backtest | login | kill | status)
├── config.py            # Pydantic settings (reads from .env)
├── logger.py            # Structlog + Rich structured logging
├── requirements.txt
├── .env.example         # Copy to .env and fill credentials
│
├── data/                # Zerodha Kite Connect data layer
│   ├── kite_client.py   # KiteConnect wrapper (auth, quotes, orders, ticks)
│   ├── instruments.py   # Symbol → instrument_token mapping (daily cache)
│   ├── historical.py    # Historical OHLCV downloader (batched, rate-limited)
│   └── tick_processor.py# Live tick → 1-min OHLCV bar builder
│
├── news/                # External intelligence layer
│   ├── fetcher.py       # RSS feed fetcher (feedparser)
│   ├── normalizer.py    # Filter + clean articles by universe relevance
│   └── summarizer.py    # Bullet-point context builder for LLM prompt
│
├── features/            # Technical indicator engine
│   └── indicators.py    # SMA/EMA, RSI, ATR, Z-score, momentum, volatility
│
├── llm/                 # DeepSeek decision engine
│   ├── deepseek_client.py   # OpenAI-compatible API client
│   ├── context_builder.py   # Assembles the structured LLM prompt
│   └── signal_validator.py  # Validates JSON signal (strict type + range checks)
│
├── risk/                # Risk management (gatekeeper)
│   └── risk_manager.py  # Max exposure, daily loss limit, kill switch
│
├── portfolio/           # Portfolio state manager
│   └── state.py         # Positions, PnL, trade journal, equity curve
│
├── execution/           # Zerodha live execution
│   └── zerodha_broker.py# place/cancel/poll orders via Kite API
│
├── engine/              # Event-driven orchestrator
│   └── trade_engine.py  # Main live trading loop (fetch → signal → risk → execute)
│
├── backtest/            # Historical simulation
│   ├── backtester.py    # Event-driven backtester (no lookahead)
│   └── simulator.py     # Simulated order book with slippage + TC
│
└── analytics/           # Performance reporting
    └── reporter.py      # Sharpe, CAGR, drawdown, win rate, plots, CSV export
```

## Running the system

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill: KITE_API_KEY, KITE_API_SECRET, DEEPSEEK_API_KEY, UNIVERSE

# 3. Generate Kite access token (do this each morning before market open)
python main.py login

# 4. Run live trading
python main.py live

# 5. Run backtest (technical strategy, no LLM cost)
python main.py backtest --from 2023-01-01 --to 2024-01-01 --no-llm

# 6. Run backtest with LLM signals
python main.py backtest --from 2023-01-01 --to 2024-01-01

# 7. Emergency stop
python main.py kill

# 8. Check live positions
python main.py status
```

## Key Design Invariants

1. **LLM never executes trades** — it outputs JSON signals only.
2. **Risk manager is the last gate** — every trade must pass all checks.
3. **No lookahead in backtester** — indicators use `df[:bar_t]` only; orders fill at `open[t+1]`.
4. **Every trade is logged** — `logs/` contains full audit trail.
5. **Kill switch is immediate** — `RiskManager.activate_kill_switch()` blocks all new trades.
6. **Credentials never in code** — all secrets via `.env`.

## Daily Operations (Live Trading)

1. Before 9:00 AM: `python main.py login` → paste new access token into `.env`
2. Start system: `python main.py live`
3. System runs 5-minute decision cycles during market hours
4. At 3:25 PM: manually stop (Ctrl+C) — engine cancels open orders on shutdown
5. Review `logs/simplequant.log` for trade audit

## Adding a New Symbol

Add the NSE symbol to `UNIVERSE` in `.env`:
```
UNIVERSE=RELIANCE,INFY,TCS,HDFCBANK,ICICIBANK,SBIN,AXISBANK,WIPRO,LT,BAJFINANCE,TATAMOTORS
```
No code changes required.

## Modifying Risk Parameters

All risk limits are in `.env`:
```
MAX_CAPITAL_EXPOSURE=0.20
MAX_POSITION_SIZE_PCT=0.10
MAX_OPEN_POSITIONS=5
MAX_DAILY_LOSS_PCT=0.02
MIN_SIGNAL_CONFIDENCE=0.60
```

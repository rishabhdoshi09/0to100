"""CLI entrypoint – ``python -m sq_ai.main {backtest|live|screener}``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def cmd_backtest(args: argparse.Namespace) -> int:
    from sq_ai.backend.data_fetcher import fetch_yf_history
    from sq_ai.backtest.backtester import Backtester

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"], index_col="date")
        df.columns = [c.lower() for c in df.columns]
    else:
        df = fetch_yf_history(args.symbol, period=args.period, interval="1d")
        if df.empty:
            print(f"✗ no data for {args.symbol}", file=sys.stderr)
            return 1

    bt = Backtester()
    res = bt.run_single(df, symbol=args.symbol)
    print(json.dumps(res.stats, indent=2, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.equity.to_csv(args.out)
        print(f"▶ equity curve → {args.out}")
    return 0


def cmd_live(_args: argparse.Namespace) -> int:
    import uvicorn
    host = os.environ.get("SQ_API_HOST", "127.0.0.1")
    port = int(os.environ.get("SQ_API_PORT", 8000))
    uvicorn.run("sq_ai.api.app:app", host=host, port=port, log_level="info")
    return 0


def cmd_screener(_args: argparse.Namespace) -> int:
    from sq_ai.backend.scheduler import TradingScheduler
    s = TradingScheduler()
    out = s.run_screener()
    print(json.dumps(out, indent=2, default=float))
    return 0


def cmd_cycle(_args: argparse.Namespace) -> int:
    from sq_ai.backend.scheduler import TradingScheduler
    s = TradingScheduler()
    out = s.run_cycle()
    print(json.dumps(out, indent=2, default=float))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="sq_ai")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="run backtester on a single symbol")
    bt.add_argument("--symbol", default="RELIANCE.NS")
    bt.add_argument("--period", default="2y")
    bt.add_argument("--csv", help="CSV path (overrides yfinance)")
    bt.add_argument("--out", help="equity curve CSV path")
    bt.set_defaults(func=cmd_backtest)

    live = sub.add_parser("live", help="start FastAPI + APScheduler")
    live.set_defaults(func=cmd_live)

    sc = sub.add_parser("screener", help="run overnight screener once")
    sc.set_defaults(func=cmd_screener)

    cy = sub.add_parser("cycle", help="run one 5-min decision cycle")
    cy.set_defaults(func=cmd_cycle)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

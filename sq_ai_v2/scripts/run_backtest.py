#!/usr/bin/env python3
"""
Backtest runner.

Usage:
  python scripts/run_backtest.py [--start 2020-01-01] [--end 2023-12-31] [--capital 1000000]

Saves results to:
  logs/equity_curve.parquet
  logs/trades.parquet
  logs/backtest_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from config.settings import settings
from backtest.engine import BacktestEngine
from data.ingestion.historical import HistoricalIngestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--start", default=settings.backtest_start_date)
    parser.add_argument("--end", default=settings.backtest_end_date)
    parser.add_argument("--capital", type=float, default=settings.backtest_initial_capital)
    parser.add_argument("--symbols", default="", help="Comma-separated (default: all)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or settings.symbol_list
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"Loading data: {symbols} ({start} → {end})")
    ingest = HistoricalIngestion()
    data = ingest.load_from_cache(symbols=symbols, from_date=start, to_date=end)

    if not data:
        logger.error("No data. Run 'make ingest' first.")
        sys.exit(1)

    logger.info(f"Running backtest on {len(data)} symbols, capital={args.capital:,.0f}")
    engine = BacktestEngine(data, initial_capital=args.capital)
    results = engine.run()

    # Save outputs
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    equity: pd.Series = results.get("equity_curve", pd.Series())
    trades: pd.DataFrame = results.get("trades", pd.DataFrame())
    stats: dict = results.get("stats", {})

    if not equity.empty:
        equity.to_frame("equity").to_parquet(log_dir / "equity_curve.parquet")
        logger.info(f"Equity curve saved → {log_dir}/equity_curve.parquet")

    if not trades.empty:
        trades.to_parquet(log_dir / "trades.parquet")
        logger.info(f"Trades saved → {log_dir}/trades.parquet ({len(trades)} trades)")

    with open(log_dir / "backtest_stats.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, float) else v) for k, v in stats.items()}, f, indent=2)

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key, val in stats.items():
        if isinstance(val, float):
            if "rate" in key or "return" in key or "drawdown" in key:
                logger.info(f"  {key:30s}: {val:.2%}")
            else:
                logger.info(f"  {key:30s}: {val:.4f}")
        else:
            logger.info(f"  {key:30s}: {val}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

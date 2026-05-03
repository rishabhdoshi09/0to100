#!/usr/bin/env python3
"""
Data ingestion script.

Usage:
  python scripts/ingest_data.py [--symbols SYM1,SYM2] [--start 2020-01-01] [--end 2023-12-31]

Downloads OHLCV data for all symbols in the universe, stores in QuestDB,
and writes Parquet cache.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import settings
from data.ingestion.historical import HistoricalIngestion
from data.storage.postgres_client import PostgresClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical OHLCV data")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: all in UNIVERSE)")
    parser.add_argument("--start", default=settings.backtest_start_date)
    parser.add_argument("--end", default=settings.backtest_end_date)
    parser.add_argument("--interval", default="day", choices=["minute", "5minute", "15minute", "60minute", "day"])
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or settings.symbol_list
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    logger.info(f"Ingesting {len(symbols)} symbols from {start} to {end}")

    # Initialise PostgreSQL schema
    pg = PostgresClient()
    pg.init_schema()

    ingestion = HistoricalIngestion()
    data = ingestion.ingest_all(symbols=symbols, from_date=start, to_date=end, interval=args.interval)

    logger.info(f"Ingestion complete: {len(data)} symbols ready")
    for sym, df in data.items():
        logger.info(f"  {sym}: {len(df)} bars ({df.index.min().date()} → {df.index.max().date()})")


if __name__ == "__main__":
    main()

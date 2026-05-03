#!/usr/bin/env python3
"""
Model training script.

Loads cached historical data and runs the full training pipeline including:
  • Feature engineering
  • LightGBM, CNN, LSTM training
  • Meta-learner training
  • Calibration
  • HMM regime detection
  • Walk-forward validation (optional)

Usage:
  python scripts/train_models.py [--walk-forward] [--symbols SYM1,SYM2]
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
from models.train_pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: all)")
    parser.add_argument("--start", default=settings.backtest_start_date)
    parser.add_argument("--end", default=settings.backtest_end_date)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or settings.symbol_list
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Load data
    logger.info(f"Loading data for {len(symbols)} symbols ({start} → {end})")
    ingest = HistoricalIngestion()
    data = ingest.load_from_cache(symbols=symbols, from_date=start, to_date=end)

    if not data:
        logger.error("No data loaded. Run 'make ingest' first.")
        sys.exit(1)

    if args.walk_forward:
        from backtest.walk_forward import WalkForwardValidator
        logger.info("Starting walk-forward validation...")
        wf = WalkForwardValidator()
        results = wf.run(data)
        logger.info(f"Walk-forward results: {results}")
    else:
        logger.info("Starting single-fold training...")
        pipeline = TrainingPipeline()
        metrics = pipeline.run(data)
        logger.info(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()

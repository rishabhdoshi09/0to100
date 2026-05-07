"""
SimpleQuant AI — Main Entry Point

Usage:
  python main.py live                          # start live trading loop
  python main.py backtest --from 2023-01-01 --to 2024-01-01
  python main.py backtest --from 2023-01-01 --to 2024-01-01 --no-llm
  python main.py login                         # generate Kite access token
  python main.py kill                          # activate kill switch
  python main.py status                        # print portfolio status

Environment:
  Copy .env.example → .env and fill in credentials before running.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Add simplequant to path for cleaner imports
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from logger import configure_logging, get_logger

configure_logging()
log = get_logger("main")


# ── Sub-commands ───────────────────────────────────────────────────────────────

def cmd_login(args) -> None:
    """Generate a Kite access token interactively."""
    from data.kite_client import KiteClient
    kite = KiteClient()
    print("\n=== Kite Login ===")
    print(f"Open this URL in your browser:\n\n  {kite.login_url()}\n")
    request_token = input("Paste the request_token from the redirect URL: ").strip()
    if not request_token:
        print("No token provided. Aborting.")
        sys.exit(1)
    access_token = kite.generate_session(request_token)
    print(f"\nAccess token: {access_token}")
    print("\nAdd this to your .env file as:")
    print(f"  KITE_ACCESS_TOKEN={access_token}")


def cmd_live(args) -> None:
    """Start the live trading loop."""
    _assert_credentials()
    log.info("starting_live_trading")

    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from execution.zerodha_broker import ZerodhaBroker
    from portfolio.state import PortfolioState
    from risk.risk_manager import RiskManager
    from engine.trade_engine import TradeEngine

    kite = KiteClient()
    instruments = InstrumentManager()
    historical = HistoricalDataFetcher(kite, instruments)
    portfolio = PortfolioState(settings.backtest_initial_capital)
    risk = RiskManager()
    broker = ZerodhaBroker(kite)

    engine = TradeEngine(
        kite=kite,
        portfolio=portfolio,
        risk_manager=risk,
        broker=broker,
        instruments=instruments,
        historical=historical,
    )

    print("\n=== SimpleQuant AI — Live Trading ===")
    print(f"Universe : {', '.join(settings.symbol_list)}")
    print(f"Cycle    : every {settings.cycle_interval_seconds}s")
    print(f"Exchange : {settings.exchange}")
    print(f"Product  : {settings.product_type}")
    print("\nPress Ctrl+C to stop gracefully.\n")

    engine.run()


def cmd_backtest(args) -> None:
    """Run backtest on historical Kite data."""
    _assert_credentials()

    from_date: str = args.from_date
    to_date: str = args.to_date
    use_llm: bool = not args.no_llm

    log.info(
        "starting_backtest",
        from_date=from_date,
        to_date=to_date,
        use_llm=use_llm,
    )

    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from backtest.backtester import Backtester
    from analytics.reporter import PerformanceReporter

    kite = KiteClient()
    instruments = InstrumentManager()
    historical = HistoricalDataFetcher(kite, instruments)

    print("\n=== SimpleQuant AI — Backtest ===")
    print(f"Period  : {from_date} → {to_date}")
    print(f"Universe: {', '.join(settings.symbol_list)}")
    print(f"LLM     : {'enabled' if use_llm else 'disabled (technical-only)'}")
    print(f"Capital : ₹{settings.backtest_initial_capital:,.0f}\n")

    print("Downloading historical data…")
    hist_data = {}
    for symbol in settings.symbol_list:
        df = historical.fetch(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval="day",
        )
        if not df.empty:
            hist_data[symbol] = df
            print(f"  {symbol}: {len(df)} bars")
        else:
            print(f"  {symbol}: no data — skipped")

    if not hist_data:
        print("No data available. Check Kite credentials and symbol names.")
        sys.exit(1)

    print("\nRunning backtest…")
    backtester = Backtester(
        historical_data=hist_data,
        initial_capital=settings.backtest_initial_capital,
        use_llm=use_llm,
    )
    result = backtester.run()

    print("\nGenerating performance report…")
    reporter = PerformanceReporter()
    reporter.generate_report(
        equity_curve=result["equity_curve"],
        trade_journal=result["trade_journal"],
        initial_capital=result["initial_capital"],
        label=f"backtest_{from_date}_{to_date}",
    )

    final = result["final_equity"]
    initial = result["initial_capital"]
    ret = (final - initial) / initial * 100
    print(f"\nBacktest complete. Return: {ret:.2f}%  |  Final equity: ₹{final:,.0f}")
    print(f"Reports saved to: {settings.log_dir}/")


def cmd_screener(args) -> None:
    """
    Screen NSE stocks by fundamentals + technicals, or fetch one symbol's data.

    Single-symbol mode  : python main.py screener --symbol RELIANCE
    Multi-stock screen  : python main.py screener --pe-max 20 --roe-min 15 --rsi-max 35
    Export results      : python main.py screener --pe-max 25 --export results.csv
    """
    import json as _json

    # ── SINGLE-SYMBOL MODE ────────────────────────────────────────────────
    symbol: str = (args.symbol or "").upper().strip()
    _multi_flags = [
        args.pe_max, args.roe_min, args.debt_max, args.market_cap_min,
        args.promoter_min, args.div_yield_min, args.rsi_max, args.rsi_min,
        args.volume_spike, args.above_sma, args.below_sma, args.signal,
    ]
    if symbol and not any(_multi_flags):
        # Old behaviour: full fundamentals for one symbol
        from fundamentals.fetcher import get_deep_fundamentals
        force = args.force
        table_filter = args.table or ""
        log.info("screener_single_symbol", symbol=symbol, force=force)
        print(f"\nFetching fundamentals for {symbol} "
              f"{'(force refresh)' if force else '(cache-first)'}…")
        try:
            data = get_deep_fundamentals(symbol, force_refresh=force)
        except ValueError as exc:
            print(f"\nError: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"\nFailed: {exc}")
            sys.exit(1)

        _ALL_SECTIONS = [
            "key_ratios", "profit_loss", "balance_sheet",
            "quarterly_results", "shareholding", "cash_flow", "peer_comparison",
        ]
        sections = [table_filter] if table_filter else _ALL_SECTIONS
        try:
            from tabulate import tabulate as _tab
            _HAS_TAB = True
        except ImportError:
            _HAS_TAB = False

        print(f"\n{'═'*60}")
        print(f"  {symbol} — Deep Fundamentals  (source: screener.in)")
        print(f"  {'Consolidated' if data.get('metadata', {}).get('consolidated') else 'Standalone'}")
        print(f"{'═'*60}")
        about = data.get("about", "")
        if about and not table_filter:
            print(f"\nAbout:\n  {about[:300]}{'…' if len(about) > 300 else ''}\n")
        for sec in sections:
            rows = data.get(sec)
            if not rows:
                continue
            print(f"\n{'─'*60}\n  {sec.replace('_',' ').title()}\n{'─'*60}")
            if isinstance(rows, list) and rows:
                if _HAS_TAB:
                    print(_tab(rows[:25], headers="keys",
                               tablefmt="rounded_outline", floatfmt=".2f"))
                else:
                    print(_json.dumps(rows[:20], indent=2, ensure_ascii=False))
        meta = data.get("metadata", {})
        print(f"\n[Rows: {meta.get('total_rows_scraped','?')} | {data.get('url','?')}]\n")
        return

    # ── MULTI-STOCK SCREEN MODE ───────────────────────────────────────────
    log.info("screener_multi_stock", filters={
        "pe_max": args.pe_max, "roe_min": args.roe_min,
        "rsi_max": args.rsi_max, "limit": args.limit,
    })

    from screener.engine import ScreenerEngine
    engine = ScreenerEngine()

    print(f"\n{'═'*60}")
    print("  NSE Stock Screener")
    active = {k: v for k, v in {
        "P/E ≤": args.pe_max, "ROE ≥": args.roe_min, "Debt/Eq ≤": args.debt_max,
        "MCap ≥ ₹Cr": args.market_cap_min, "Promoter ≥%": args.promoter_min,
        "Div Yield ≥%": args.div_yield_min, "RSI ≤": args.rsi_max,
        "RSI ≥": args.rsi_min, "Vol spike ≥": args.volume_spike,
        "Above SMA": args.above_sma, "Below SMA": args.below_sma,
        "Signal": args.signal,
    }.items() if v is not None}
    for k, v in active.items():
        print(f"  {k}: {v}")
    print(f"{'═'*60}\n")

    try:
        df = engine.screen_by_ratios(
            pe_max=args.pe_max,
            roe_min=args.roe_min,
            debt_max=args.debt_max,
            market_cap_min_cr=args.market_cap_min,
            promoter_holding_min=args.promoter_min,
            dividend_yield_min=args.div_yield_min,
            rsi_max=args.rsi_max,
            rsi_min=args.rsi_min,
            volume_spike_min=args.volume_spike,
            price_above_sma_days=args.above_sma,
            price_below_sma_days=args.below_sma,
            ensemble_signal=args.signal,
            limit=args.limit,
            scrape_missing_fundamentals=args.scrape,
        )
    except Exception as exc:
        print(f"\nScreener error: {exc}")
        sys.exit(1)

    if df.empty:
        print("No stocks match the criteria.\n")
        return

    print(f"\nFound {len(df)} stocks:\n")
    try:
        from tabulate import tabulate as _tab
        print(_tab(df, headers="keys", tablefmt="rounded_outline",
                   floatfmt=".2f", showindex=False))
    except ImportError:
        print(df.to_string(index=False))

    if args.export:
        df.to_csv(args.export, index=False)
        print(f"\nExported {len(df)} rows → {args.export}")
    print()


def cmd_ensemble(args) -> None:
    """Print ensemble ML signal for a symbol."""
    symbol = args.symbol.upper()
    log.info("ensemble_signal_requested", symbol=symbol)

    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from ml.ensemble_signal import EnsembleSignalGenerator

    kite = KiteClient()
    instruments = InstrumentManager()
    historical = HistoricalDataFetcher(kite, instruments)

    import sys
    from datetime import date, timedelta as td
    to_d = date.today().strftime("%Y-%m-%d")
    from_d = (date.today() - td(days=400)).strftime("%Y-%m-%d")

    print(f"\nFetching data for {symbol}…")
    df = historical.fetch(symbol=symbol, from_date=from_d, to_date=to_d, interval="day")
    if df is None or df.empty:
        print(f"No data for {symbol}. Check Kite credentials.")
        sys.exit(1)

    gen = EnsembleSignalGenerator()
    sig = gen.generate_signal(df, symbol)

    print(f"\n{'─'*50}")
    print(f"  Symbol    : {sig['symbol']}")
    print(f"  Action    : {sig['action']}")
    print(f"  Confidence: {sig['confidence']:.1%}")
    print(f"  Reasoning : {sig['reasoning']}")
    details = sig.get("ensemble_details", {})
    if details:
        print("\n  Model breakdown:")
        for model_name, info in details.items():
            if isinstance(info, dict):
                print(f"    {model_name:12s}: {info.get('action','?'):4s} @ {info.get('confidence',0):.1%}")
    print(f"{'─'*50}\n")


def cmd_lgb(args) -> None:
    """LightGBM signal for a single symbol."""
    symbol = args.symbol.upper()
    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from ml.lgbm_signal import LightGBMSignalGenerator
    import sys
    from datetime import date, timedelta as td

    kite = KiteClient()
    historical = HistoricalDataFetcher(kite, InstrumentManager())
    to_d = date.today().strftime("%Y-%m-%d")
    from_d = (date.today() - td(days=400)).strftime("%Y-%m-%d")

    print(f"\nFetching data for {symbol}…")
    df = historical.fetch(symbol=symbol, from_date=from_d, to_date=to_d, interval="day")
    if df is None or df.empty:
        print(f"No data for {symbol}. Check Kite credentials.")
        sys.exit(1)

    sig = LightGBMSignalGenerator().generate_signal(df, symbol)
    print(f"\n{'─'*50}")
    print(f"  Symbol    : {sig['symbol']}")
    print(f"  Action    : {sig['action']}")
    print(f"  Confidence: {sig['confidence']:.1%}")
    print(f"  Reasoning : {sig['reasoning']}")
    print(f"{'─'*50}\n")


def cmd_multi(args) -> None:
    """Multi-horizon (1d/5d/10d) signal for a single symbol."""
    symbol = args.symbol.upper()
    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from ml.multi_horizon import MultiHorizonSignalGenerator
    import sys
    from datetime import date, timedelta as td

    kite = KiteClient()
    historical = HistoricalDataFetcher(kite, InstrumentManager())
    to_d = date.today().strftime("%Y-%m-%d")
    from_d = (date.today() - td(days=500)).strftime("%Y-%m-%d")

    print(f"\nFetching data for {symbol}…")
    df = historical.fetch(symbol=symbol, from_date=from_d, to_date=to_d, interval="day")
    if df is None or df.empty:
        print(f"No data for {symbol}. Check Kite credentials.")
        sys.exit(1)

    result = MultiHorizonSignalGenerator().generate_signals(df, symbol)
    print(f"\n{'─'*55}")
    print(f"  Multi-Horizon Signal — {symbol}")
    print(f"{'─'*55}")
    for h in ["horizon_1d", "horizon_5d", "horizon_10d"]:
        hr = result[h]
        print(f"  {h:14s}: {hr['action']:4s}  confidence={hr['confidence']:.1%}")
    cons = result["consensus"]
    print(f"{'─'*55}")
    print(f"  CONSENSUS   : {cons['action']:4s}  confidence={cons['confidence']:.1%}  "
          f"agreement={cons['agreement']}/3")
    print(f"{'─'*55}\n")


def cmd_monitor(args) -> None:
    """Compute and display model decay metrics."""
    from monitoring.decay_monitor import ModelDecayMonitor
    mon = ModelDecayMonitor()
    metrics = mon.compute_decay_metrics()

    if metrics.empty:
        print("\nNo live trades recorded in the last 30 days.\n"
              "Trades are logged automatically when the live engine closes positions.\n")
        return

    print(f"\n{'─'*70}")
    print(f"  Model Decay Monitor — rolling 30-day window")
    print(f"{'─'*70}")
    print(f"  {'Strategy':<18} {'Trades':>6} {'Live Sharpe':>12} "
          f"{'BT Sharpe':>10} {'Decay':>7} {'Alert':>6}")
    print(f"  {'─'*18} {'─'*6} {'─'*12} {'─'*10} {'─'*7} {'─'*6}")
    for _, row in metrics.iterrows():
        decay = f"{row['decay_ratio']:.2f}" if row["decay_ratio"] is not None else "N/A"
        alert = "⚠️ YES" if row["alert"] else "OK"
        print(f"  {row['strategy']:<18} {row['trade_count']:>6} "
              f"{row['live_sharpe']:>12.3f} {row['backtest_sharpe']:>10.3f} "
              f"{decay:>7} {alert:>6}")
    print(f"{'─'*70}\n")

    alerted = mon.check_and_alert(metrics)
    if alerted:
        print(f"Telegram alerts sent for: {', '.join(alerted)}\n")


def cmd_strategy(args) -> None:
    """Enable / disable / show status of individual strategies."""
    from monitoring.strategy_manager import StrategyManager
    sm = StrategyManager()

    if args.strategy_cmd == "status":
        print(f"\n{'─'*35}")
        print(f"  Strategy Kill-Switch Status")
        print(f"{'─'*35}")
        for name, enabled in sm.status().items():
            icon = "✅" if enabled else "🔴"
            print(f"  {icon}  {name:<20} {'ON' if enabled else 'OFF'}")
        print(f"{'─'*35}\n")

    elif args.strategy_cmd == "enable":
        sm.enable(args.name)
        print(f"✅ Strategy '{args.name}' enabled.")

    elif args.strategy_cmd == "disable":
        sm.disable(args.name)
        print(f"🔴 Strategy '{args.name}' disabled.")


def cmd_alerts(args) -> None:
    """Start the background signal monitor (runs until Ctrl+C)."""
    log.info("alerts_monitor_starting")
    from notify.alerts import SignalMonitor
    monitor = SignalMonitor()
    monitor.run()


def cmd_chart(args) -> None:
    """
    Generate a professional interactive chart for a symbol and save to HTML.
    Optionally show only a specific chart type (main | footprint | liquidity).
    """
    import yfinance as yf
    from pathlib import Path as _Path

    symbol = args.symbol.upper()
    period = args.period          # e.g. "3mo", "1y", "5d"
    chart_type = args.type        # "main" | "footprint" | "liquidity" | "all"

    # ── fetch data ────────────────────────────────────────────────────────
    _PERIOD_INTERVAL = {
        "1d": "5m", "5d": "30m",
        "1mo": "1d", "3mo": "1d", "6mo": "1d", "1y": "1d",
    }
    interval = _PERIOD_INTERVAL.get(period, "1d")
    ticker = f"{symbol}.NS"
    print(f"\nFetching {ticker}  period={period}  interval={interval} …")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        print(f"No data returned for {ticker}. Check the symbol spelling.")
        sys.exit(1)

    # yfinance may return MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    print(f"Got {len(df)} bars.")

    out_dir = _Path("charts")
    out_dir.mkdir(exist_ok=True)

    saved = []

    # ── main chart ────────────────────────────────────────────────────────
    if chart_type in ("main", "all"):
        from charting.engine import SmartChart
        fig = SmartChart().build(df, symbol=symbol, show_vp=True)
        out = out_dir / f"chart_{symbol}_{period}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        saved.append(str(out))

    # ── footprint chart ───────────────────────────────────────────────────
    if chart_type in ("footprint", "all"):
        from charting.footprint import FootprintAnalyzer
        fig = FootprintAnalyzer().build_figure(df, symbol=symbol)
        out = out_dir / f"footprint_{symbol}_{period}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        saved.append(str(out))

    # ── liquidity heatmap ─────────────────────────────────────────────────
    if chart_type in ("liquidity", "all"):
        from charting.liquidity import LiquidityHeatmap
        current_price = float(df["close"].iloc[-1])
        book = LiquidityHeatmap().simulate_book(current_price)
        fig = LiquidityHeatmap().build_figure(book, symbol=symbol)
        out = out_dir / f"liquidity_{symbol}_{period}.html"
        fig.write_html(str(out), include_plotlyjs="cdn")
        saved.append(str(out))

    print("\nCharts saved:")
    for path in saved:
        print(f"  {path}")
    print("\nOpen any .html file in a browser to view the interactive chart.\n")


def cmd_explain(args) -> None:
    """Print a plain-language explanation for an indicator."""
    from charting.explanations import explain, list_all
    indicator = args.indicator
    if indicator.lower() in ("list", "all", "?"):
        print("\nAvailable indicators:\n  " + "\n  ".join(list_all()) + "\n")
        return
    print(explain(indicator))


def cmd_fnolive(args) -> None:
    """Start the live trading loop with F&O execution enabled."""
    if not settings.enable_fno:
        print("F&O trading is disabled. Set ENABLE_FNO=true in .env to enable.")
        sys.exit(0)

    _assert_credentials()
    log.info("starting_fno_live_trading")

    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from execution.zerodha_broker import ZerodhaBroker
    from execution.fo_executor import FnOExecutor
    from portfolio.state import PortfolioState
    from risk.risk_manager import RiskManager
    from engine.trade_engine import TradeEngine

    print("\n" + "=" * 60)
    print("  WARNING: F&O trading enabled.")
    print("  Losses can exceed capital.")
    print("  Confirm you understand margin requirements.")
    print("=" * 60)
    confirm = input('\nType "YES" to continue, anything else to abort: ').strip()
    if confirm != "YES":
        print("Aborted.")
        sys.exit(0)

    kite = KiteClient()
    instruments = InstrumentManager()
    historical = HistoricalDataFetcher(kite, instruments)
    portfolio = PortfolioState(settings.backtest_initial_capital)
    risk = RiskManager()
    broker = ZerodhaBroker(kite)
    _fno = FnOExecutor(kite)  # available for strategy-layer use; engine unchanged

    engine = TradeEngine(
        kite=kite,
        portfolio=portfolio,
        risk_manager=risk,
        broker=broker,
        instruments=instruments,
        historical=historical,
    )

    print("\n=== SimpleQuant AI — F&O Live Trading ===")
    print(f"Universe : {', '.join(settings.symbol_list)}")
    print(f"Cycle    : every {settings.cycle_interval_seconds}s")
    print(f"Product  : {settings.fno_default_product}")
    print("\nPress Ctrl+C to stop gracefully.\n")

    engine.run()


def cmd_walkforward(args) -> None:
    """Run walk-forward validation on historical Kite data."""
    _assert_credentials()

    from_date: str = args.from_date
    to_date: str = args.to_date

    log.info(
        "starting_walk_forward",
        from_date=from_date,
        to_date=to_date,
    )

    from data.kite_client import KiteClient
    from data.instruments import InstrumentManager
    from data.historical import HistoricalDataFetcher
    from backtest.walk_forward import WalkForwardValidator

    kite = KiteClient()
    instruments = InstrumentManager()
    historical = HistoricalDataFetcher(kite, instruments)

    print("\n=== SimpleQuant AI — Walk-Forward Validation ===")
    print(f"Period  : {from_date} → {to_date}")
    print(f"Universe: {', '.join(settings.symbol_list)}")
    print(f"IS days : {settings.walkforward_is_days}  |  OOS days: {settings.walkforward_oos_days}")
    print(f"Capital : ₹{settings.backtest_initial_capital:,.0f}\n")

    print("Downloading historical data…")
    hist_data = {}
    for symbol in settings.symbol_list:
        df = historical.fetch(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            interval="day",
        )
        if not df.empty:
            hist_data[symbol] = df
            print(f"  {symbol}: {len(df)} bars")
        else:
            print(f"  {symbol}: no data — skipped")

    if not hist_data:
        print("No data available. Check Kite credentials and symbol names.")
        sys.exit(1)

    print("\nRunning walk-forward validation…")
    validator = WalkForwardValidator()
    summary = validator.run(hist_data)

    if not summary:
        print("Walk-forward produced no results — insufficient data.")
        sys.exit(1)

    print(f"\n{'─'*52}")
    print(f"  Windows completed     : {summary['total_windows']}")
    print(f"  Mean OOS Sharpe       : {summary['mean_oos_sharpe']:.3f}")
    print(f"  Mean IS  Sharpe       : {summary['mean_is_sharpe']:.3f}")
    print(f"  IS/OOS Sharpe ratio   : {summary['is_oos_sharpe_ratio']:.3f}  (>2.0 = overfitting)")
    print(f"  % Profitable OOS wins : {summary['pct_profitable_oos_windows']:.1f}%")
    print(f"{'─'*52}")

    if summary["is_oos_sharpe_ratio"] > 2.0:
        print("  WARNING: IS/OOS ratio > 2.0 — strategy may be overfit to IS data.")

    print("\nOOS Window Details:")
    for w in summary.get("window_details", []):
        print(
            f"  [{w['window']:02d}] {w['oos_start']} → {w['oos_end']} "
            f"Sharpe={w['oos_sharpe']:+.3f}  ret={w['oos_total_return_pct']:+.2f}%  "
            f"dd={w['oos_max_drawdown_pct']:.2f}%"
        )

    print("\nMost-frequent winning parameter sets:")
    for params_str, count in list(summary.get("best_params_frequency", {}).items())[:3]:
        print(f"  {count}x  {params_str}")

    print(f"\nDone. Reports available in {settings.log_dir}/")


def cmd_kill(args) -> None:
    """Write a kill switch flag file. The live engine checks this on startup."""
    flag = Path("logs/.kill_switch")
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    print("Kill switch flag written. The engine will refuse to start/continue.")


def cmd_status(args) -> None:
    """Print live portfolio status (reads Kite positions directly)."""
    _assert_credentials()
    from data.kite_client import KiteClient
    kite = KiteClient()
    try:
        positions = kite.get_positions()
        margins = kite.get_margins()

        print("\n=== Live Portfolio Status ===")
        net_positions = positions.get("net", [])
        if not net_positions:
            print("No open positions.")
        else:
            for p in net_positions:
                pnl = p.get("unrealised", 0)
                print(
                    f"  {p['tradingsymbol']:12s}  qty={p['quantity']:4d}  "
                    f"avg={p.get('average_price', 0):.2f}  "
                    f"ltp={p.get('last_price', 0):.2f}  "
                    f"pnl={pnl:+.2f}"
                )

        equity = margins.get("equity", {}).get("available", {})
        cash = equity.get("cash", 0)
        print(f"\nAvailable cash: ₹{cash:,.2f}")
    except Exception as exc:
        print(f"Failed to fetch status: {exc}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _assert_credentials() -> None:
    missing = []
    if not settings.kite_api_key:
        missing.append("KITE_API_KEY")
    if not settings.kite_api_secret:
        missing.append("KITE_API_SECRET")
    if not settings.kite_access_token:
        missing.append("KITE_ACCESS_TOKEN (run: python main.py login)")
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if missing:
        print(f"Missing credentials in .env: {', '.join(missing)}")
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simplequant",
        description="SimpleQuant AI — Zerodha + DeepSeek trading system",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("login", help="Generate Kite access token via OAuth flow")

    sub.add_parser("live", help="Start live trading loop")

    bt = sub.add_parser("backtest", help="Run event-driven backtest")
    bt.add_argument(
        "--from",
        dest="from_date",
        default=(date.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
    )
    bt.add_argument(
        "--to",
        dest="to_date",
        default=date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
    )
    bt.add_argument(
        "--no-llm",
        action="store_true",
        help="Use pure technical strategy (no DeepSeek calls — fast, free)",
    )

    wf = sub.add_parser("walkforward", help="Run walk-forward parameter validation")
    wf.add_argument(
        "--from",
        dest="from_date",
        default=(date.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
    )
    wf.add_argument(
        "--to",
        dest="to_date",
        default=date.today().strftime("%Y-%m-%d"),
        metavar="YYYY-MM-DD",
    )

    sub.add_parser("fnolive", help="Start live trading with F&O execution (requires ENABLE_FNO=true)")

    scr = sub.add_parser(
        "screener",
        help="Screen ALL NSE stocks by fundamentals + technicals, OR fetch one symbol's data",
    )
    scr.add_argument("--symbol",    metavar="SYMBOL",  default="",
                     help="Single symbol: show full fundamentals (e.g. BEL)")
    scr.add_argument("--force",     action="store_true",
                     help="Force-refresh cache for single-symbol mode")
    scr.add_argument("--table",     default="", metavar="SECTION",
                     help="(single-symbol) Print only one section: key_ratios | profit_loss | …")
    # ── multi-stock filter flags ──────────────────────────────────────────
    scr.add_argument("--pe-max",            type=float, metavar="N",
                     help="Maximum P/E ratio")
    scr.add_argument("--roe-min",           type=float, metavar="N",
                     help="Minimum ROE %%")
    scr.add_argument("--debt-max",          type=float, metavar="N",
                     help="Maximum Debt/Equity ratio")
    scr.add_argument("--market-cap-min",    type=float, metavar="CR",
                     help="Minimum market cap (₹ Crore)")
    scr.add_argument("--promoter-min",      type=float, metavar="PCT",
                     help="Minimum promoter holding %%")
    scr.add_argument("--div-yield-min",     type=float, metavar="N",
                     help="Minimum dividend yield %%")
    scr.add_argument("--rsi-max",           type=float, metavar="N",
                     help="Maximum RSI-14 (e.g. 30 = oversold)")
    scr.add_argument("--rsi-min",           type=float, metavar="N",
                     help="Minimum RSI-14 (e.g. 70 = overbought)")
    scr.add_argument("--volume-spike",      type=float, metavar="X",
                     help="Min volume ratio vs 30-day avg (e.g. 1.5)")
    scr.add_argument("--above-sma",         type=int,   metavar="DAYS",
                     help="Price must be above SMA(N) — 20 or 50")
    scr.add_argument("--below-sma",         type=int,   metavar="DAYS",
                     help="Price must be below SMA(N) — 20 or 50")
    scr.add_argument("--signal",            choices=["BUY", "SELL", "HOLD"],
                     help="Ensemble ML signal filter")
    scr.add_argument("--limit",             type=int, default=50,
                     help="Max results to display (default: 50)")
    scr.add_argument("--export",            metavar="FILE.csv",
                     help="Export results to CSV")
    scr.add_argument("--scrape",            action="store_true",
                     help="Scrape screener.in for symbols missing fundamentals (slower)")

    ens = sub.add_parser("ensemble", help="Print ensemble ML signal for a symbol")
    ens.add_argument("--symbol", required=True, metavar="SYMBOL", help="NSE symbol, e.g. RELIANCE")

    lgb_p = sub.add_parser("lgb", help="LightGBM signal for a symbol (trains if needed)")
    lgb_p.add_argument("--symbol", required=True, metavar="SYMBOL")

    multi_p = sub.add_parser("multi", help="Multi-horizon (1d/5d/10d) signal for a symbol")
    multi_p.add_argument("--symbol", required=True, metavar="SYMBOL")

    sub.add_parser("monitor", help="Show model decay metrics (live vs backtest Sharpe)")

    strat = sub.add_parser("strategy", help="Enable / disable / show strategy kill switches")
    strat_sub = strat.add_subparsers(dest="strategy_cmd", required=True)
    strat_sub.add_parser("status", help="Show all strategy on/off states")
    en_p = strat_sub.add_parser("enable",  help="Enable a strategy")
    en_p.add_argument("--name", required=True,
                      choices=["lgbm", "xgboost", "multi_horizon", "ensemble"])
    dis_p = strat_sub.add_parser("disable", help="Disable a strategy")
    dis_p.add_argument("--name", required=True,
                       choices=["lgbm", "xgboost", "multi_horizon", "ensemble"])

    sub.add_parser("alerts", help="Start background signal monitor (Telegram alerts)")

    ch = sub.add_parser("chart", help="Build professional interactive chart (saved as HTML)")
    ch.add_argument("--symbol", required=True, metavar="SYMBOL",
                    help="NSE symbol, e.g. RELIANCE")
    ch.add_argument("--period", default="3mo",
                    choices=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                    help="Data period (default: 3mo)")
    ch.add_argument("--type", default="main",
                    choices=["main", "footprint", "liquidity", "all"],
                    help="Chart type: main | footprint | liquidity | all (default: main)")

    ex = sub.add_parser("explain", help="Print plain-language explanation for an indicator")
    ex.add_argument("--indicator", required=True, metavar="INDICATOR",
                    help="Indicator name (e.g. vwap, rsi, macd) or 'list' to see all")

    sub.add_parser("kill", help="Write kill switch flag")
    sub.add_parser("status", help="Print live portfolio status from Kite")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "login":      cmd_login,
        "live":       cmd_live,
        "backtest":   cmd_backtest,
        "walkforward": cmd_walkforward,
        "fnolive":    cmd_fnolive,
        "screener":   cmd_screener,
        "ensemble":   cmd_ensemble,
        "lgb":        cmd_lgb,
        "multi":      cmd_multi,
        "monitor":    cmd_monitor,
        "strategy":   cmd_strategy,
        "alerts":     cmd_alerts,
        "chart":      cmd_chart,
        "explain":    cmd_explain,
        "kill":       cmd_kill,
        "status":     cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

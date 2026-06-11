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

    print(f"\n=== SimpleQuant AI — Backtest ===")
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
    metrics = reporter.generate_report(
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


def cmd_ledger(args) -> None:
    """Prediction Ledger management."""
    from forensics import prediction_ledger as pl

    if args.ledger_cmd == "stats":
        s = pl.get_stats()
        print("\n=== Prediction Ledger — Platform Stats ===")
        print(f"  Total predictions:   {s['total']}")
        print(f"  Pending:             {s['pending']}")
        print(f"  Correct:             {s['correct']}")
        print(f"  Incorrect:           {s['incorrect']}")
        print(f"  Inconclusive:        {s['inconclusive']}")
        if s["hit_rate"] is not None:
            print(f"  Hit Rate:            {s['hit_rate']:.1%}")
        if s["brier_score"] is not None:
            print(f"  Brier Score:         {s['brier_score']:.4f}  "
                  "(0 = perfect calibration)")

    elif args.ledger_cmd == "check":
        symbol = args.symbol
        print(f"\nRe-running analysis for {symbol} to check pending predictions…")
        from forensics import QuantRedFlagAnalyst
        from forensics import prediction_ledger as pl
        analyst = QuantRedFlagAnalyst()
        try:
            report = analyst.analyze(symbol)
        except Exception as exc:
            print(f"Analysis failed: {exc}")
            return
        baseline = pl.extract_baseline_metrics(report.layers)
        res = pl.check_outcomes(symbol, baseline)
        print(f"  Checked: {res['checked']}  "
              f"Correct: {res['correct']}  "
              f"Incorrect: {res['incorrect']}  "
              f"Inconclusive: {res['inconclusive']}")
        open_preds = pl.get_open_predictions(symbol)
        if open_preds:
            print(f"\n  Open predictions for {symbol}:")
            for p in open_preds:
                print(f"    [{p['check_date']}] ({p['confidence']:.0%}) "
                      f"{p['prediction']}")


def cmd_analyze(args) -> None:
    """Run the Quant Red Flag Analyst on one or more symbols."""
    from rich.console import Console
    from forensics import QuantRedFlagAnalyst
    from forensics.report import render
    from forensics.simple_report import render_simple
    from forensics import tearsheet as ts
    from logger import quiet_console

    pro = args.pro or args.explain
    if not pro:
        quiet_console()
    console = Console()

    analyst = QuantRedFlagAnalyst()
    for symbol in args.symbols:
        try:
            with console.status(f"Checking {symbol.upper()}…", spinner="dots"):
                report = analyst.analyze(symbol, save_predictions=args.ledger)
        except Exception as exc:
            console.print(f"\n[red]Couldn't analyze {symbol.upper()}[/red] — {exc}")
            continue
        if pro:
            render(report, explain=args.explain)
        else:
            render_simple(report, console=console)
        if report.predictions_saved:
            print(f"\n  Predictions logged: {report.predictions_saved} "
                  f"(run: python main.py ledger check {symbol})")
        if args.tearsheet:
            bundle = ts.generate(report)
            prefix = f"logs/{symbol.replace('/', '_')}"
            paths = ts.save(bundle, prefix)
            print(f"\nTear sheet saved → {', '.join(paths)}")


def cmd_replay(args) -> None:
    """Research Replay: point-in-time analysis on a historical date."""
    from datetime import date as _date
    from forensics import replay as rp
    from forensics.report import render_replay

    try:
        as_of = _date.fromisoformat(args.date)
    except ValueError:
        print(f"Invalid date '{args.date}' — use YYYY-MM-DD.")
        return
    from logger import quiet_console
    if not args.full:
        quiet_console()
    try:
        result = rp.replay(args.symbol, as_of)
    except Exception as exc:
        print(f"\n{args.symbol}: replay failed — {exc}")
        return
    render_replay(result, full=args.full)


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
        description="Check any stock for red flags: python main.py RELIANCE\n"
                    "(also a full Zerodha + DeepSeek trading system — see "
                    "commands below)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    an = sub.add_parser(
        "analyze",
        help="Quant Red Flag Analyst — forensic + quant analysis of a stock",
    )
    an.add_argument(
        "symbols",
        nargs="+",
        metavar="SYMBOL",
        help="NSE symbol (e.g. RELIANCE) or explicit ticker (e.g. AAPL, TCS.NS)",
    )
    an.add_argument(
        "--pro",
        action="store_true",
        help="Full institutional report (all panels, evidence locker, "
             "causal graph). Default is a clean plain-English summary.",
    )
    an.add_argument(
        "--explain",
        action="store_true",
        help="Include the full metric glossary (implies --pro)",
    )
    an.add_argument(
        "--tearsheet",
        action="store_true",
        help="Save institutional tear sheet, one-pager, and social card to logs/",
    )
    an.add_argument(
        "--ledger",
        action="store_true",
        help="Log testable predictions to the Prediction Ledger (logs/predictions.db)",
    )

    led = sub.add_parser("ledger", help="Prediction Ledger — track signal outcomes")
    led_sub = led.add_subparsers(dest="ledger_cmd", required=True)
    lc = led_sub.add_parser("check", help="Check outcomes for a symbol")
    lc.add_argument("symbol")
    led_sub.add_parser("stats", help="Platform-wide hit rate and Brier score")

    rp = sub.add_parser(
        "replay",
        help="Research Replay — run the analyst as of a historical date "
             "and compare against what actually happened",
    )
    rp.add_argument("symbol", help="Symbol to replay")
    rp.add_argument("date", help="As-of date, YYYY-MM-DD")
    rp.add_argument("--full", action="store_true",
                    help="Render the full point-in-time report, not just the summary")

    sub.add_parser("kill", help="Write kill switch flag")
    sub.add_parser("status", help="Print live portfolio status from Kite")

    return parser


_COMMANDS = {"login", "live", "backtest", "analyze", "ledger",
             "replay", "kill", "status"}


def _humanize_argv(argv: list) -> list:
    """Let people just type a stock name.

    `python main.py RELIANCE`        → analyze RELIANCE
    `python main.py` (no arguments)  → ask which stock to check
    Known commands and flags pass through untouched."""
    if not argv:
        try:
            answer = input("Which stock do you want to check? (e.g. RELIANCE): ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        symbols = answer.split()
        if not symbols:
            sys.exit(0)
        return ["analyze"] + symbols
    if argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        return ["analyze"] + argv
    return argv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(_humanize_argv(sys.argv[1:]))

    dispatch = {
        "login": cmd_login,
        "live": cmd_live,
        "backtest": cmd_backtest,
        "analyze": cmd_analyze,
        "ledger": cmd_ledger,
        "replay": cmd_replay,
        "kill": cmd_kill,
        "status": cmd_status,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

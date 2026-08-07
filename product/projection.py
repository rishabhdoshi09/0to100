"""Pure retail product projection for QuantTerm.

This module never owns trading state. It translates existing backend facts into
plain-language UI state and recommended actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field


TERMINOLOGY = {
    "snapshot": "Saved market data",
    "forward_eligible": "Data ready for trading",
    "evidence_card": "Why the system believes this",
    "regime": "Market condition",
    "allocation": "Paper-money amount",
    "risk_governor": "Safety checks",
    "intent": "Proposed paper trade",
    "reconciliation": "Record check",
    "in_sample": "Historical test",
    "no_eligible_intent": "No safe trade found",
}


@dataclass(frozen=True)
class ProductInputs:
    market_open: bool = False
    market_label: str = "Market closed"
    kite_connected: bool = False
    active_snapshot_id: str | None = None
    latest_market_date: str = ""
    instrument_count: int = 0
    data_ready: bool = False
    paper_auto_enabled: bool = False
    worker_running: bool = False
    paper_capital: float = 100_000.0
    paper_equity: float = 100_000.0
    open_positions: int = 0
    closed_trades: int = 0
    last_cycle_status: str = ""
    last_error: str = ""


@dataclass(frozen=True)
class SetupStep:
    label: str
    status: str
    detail: str
    action: str = ""


@dataclass(frozen=True)
class ProductState:
    headline: str
    readiness: str
    activity: str
    primary_action: str
    primary_key: str
    attention: str = ""
    setup_steps: tuple[SetupStep, ...] = field(default_factory=tuple)
    useful_actions: tuple[str, ...] = field(default_factory=tuple)


def build_product_state(i: ProductInputs) -> ProductState:
    steps = (
        SetupStep("Connect Zerodha", "Ready" if i.kite_connected else "Not connected",
                  "Daily Zerodha login is available." if i.kite_connected else
                  "Complete the normal Zerodha login once.", "connect"),
        SetupStep("Download market history", "Ready" if i.active_snapshot_id else "Not ready",
                  (f"Saved market data: {i.active_snapshot_id}" if i.active_snapshot_id else
                   "Historical candles have not been activated."), "update_data"),
        SetupStep("Check data quality", "Ready" if i.data_ready else "Needs attention",
                  (f"Data through {i.latest_market_date or 'latest session'}." if i.data_ready else
                   "New paper entries stay paused until data passes checks."), "update_data"),
        SetupStep("Run a backtest", "Available" if i.data_ready else "Waiting for data",
                  "Test a strategy on past market history.", "backtest"),
        SetupStep("Automatic paper trading", "On" if i.paper_auto_enabled else "Off",
                  "No per-trade approval is required." if i.paper_auto_enabled else
                  "Enable it once; normal restarts remember the setting.", "paper"),
    )

    useful = (
        "Run Backtest",
        "Find Momentum Stocks",
        "See F&O Momentum",
        "Prepare Tomorrow's Watchlist",
        "Review Paper Trades",
        "See What QuantTerm Learned",
    )

    if not i.kite_connected:
        return ProductState(
            headline="Waiting for Zerodha login",
            readiness="Market data is not connected yet",
            activity="QuantTerm is preserving existing paper records and will not invent data.",
            primary_action="Connect Zerodha",
            primary_key="connect",
            attention="Complete the normal daily Zerodha login.",
            setup_steps=steps,
            useful_actions=useful,
        )
    if not i.active_snapshot_id or not i.data_ready:
        return ProductState(
            headline="Get market data ready",
            readiness="Historical data needs an update",
            activity="QuantTerm can download, verify and activate genuine Kite history automatically.",
            primary_action="Update Market Data",
            primary_key="update_data",
            attention=i.last_error or "New paper entries remain paused until validation passes.",
            setup_steps=steps,
            useful_actions=useful,
        )
    if not i.paper_auto_enabled:
        return ProductState(
            headline="Data ready — paper trading is off",
            readiness="Ready for research and backtesting",
            activity="No automatic paper entries will be opened until you enable PAPER_AUTO.",
            primary_action="Enable Automatic Paper Trading",
            primary_key="paper",
            setup_steps=steps,
            useful_actions=useful,
        )
    if i.market_open:
        status = "running" if i.worker_running else "needs attention"
        return ProductState(
            headline="Automatic paper trading is running" if i.worker_running else
                     "Automatic paper trading needs attention",
            readiness=f"Data ready · PAPER_AUTO {status}",
            activity=(f"Monitoring {i.open_positions} open paper position(s)." if i.open_positions else
                      (i.last_cycle_status or "Scanning for qualified trades.")),
            primary_action="Nothing required — QuantTerm is running" if i.worker_running else
                           "Start Paper-Trading Worker",
            primary_key="none" if i.worker_running else "start_worker",
            attention=i.last_error,
            setup_steps=steps,
            useful_actions=useful,
        )
    return ProductState(
        headline="Market closed — Research mode active",
        readiness="Ready for the next NSE session",
        activity=(f"{i.open_positions} open paper position(s) will resume management next session. "
                  f"{i.closed_trades} paper trade(s) recorded."),
        primary_action="Run Backtest",
        primary_key="backtest",
        attention=i.last_error,
        setup_steps=steps,
        useful_actions=useful,
    )

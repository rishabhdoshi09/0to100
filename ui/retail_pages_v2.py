"""Retail-completion page exports."""
from ui.retail_home_momentum import render_home, render_momentum
from ui.retail_trade_market import render_market, render_paper_trading
from ui.retail_backtest_data import render_backtest, render_data_zerodha
from ui.retail_pages import (
    render_advanced,
    render_alerts,
    render_help,
    render_learned,
    render_portfolio,
    render_reports,
    render_settings,
)

__all__ = [
    "render_home", "render_momentum", "render_paper_trading", "render_market",
    "render_backtest", "render_data_zerodha", "render_advanced", "render_alerts",
    "render_help", "render_learned", "render_portfolio", "render_reports", "render_settings",
]

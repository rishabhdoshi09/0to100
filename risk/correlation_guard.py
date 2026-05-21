"""
Correlation Guard — The Dormammu Defense.

Dormammu: "Bargain with me." No. Your portfolio bargains with correlation.

In a crisis, all positions move together. Diversification fails exactly when
you need it most. This module:
  1. Computes pairwise correlation of open positions (30-day returns)
  2. Alerts if portfolio average correlation > 0.70 (Dormammu approaching)
  3. Blocks new positions that would push correlation above threshold
  4. Computes effective position count (1 / HHI) — shows true diversification

Portfolio with 5 positions but correlation=0.85 is effectively 1.2 positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_CORRELATION_ALERT_THRESHOLD = 0.70
_CORRELATION_BLOCK_THRESHOLD = 0.80
_MIN_BARS = 20


@dataclass
class CorrelationReport:
    avg_pairwise_correlation: float
    max_pairwise_correlation: float
    effective_positions: float      # 1/HHI — true diversification count
    total_positions: int
    danger_pairs: list[tuple[str, str, float]]   # (sym1, sym2, corr) above 0.75
    verdict: str                     # HEALTHY | WARNING | DANGER
    block_new_trades: bool


def compute_portfolio_correlation(
    open_symbols: list[str],
    days: int = 30,
) -> Optional[CorrelationReport]:
    """
    Fetch returns for open symbols and compute correlation matrix.
    Returns None if fewer than 2 symbols or data unavailable.
    """
    if len(open_symbols) < 2:
        return None

    returns: dict[str, np.ndarray] = {}
    for symbol in open_symbols:
        r = _fetch_returns(symbol, days)
        if r is not None and len(r) >= _MIN_BARS:
            returns[symbol] = r

    if len(returns) < 2:
        return None

    syms = list(returns.keys())
    n = len(syms)

    # Align lengths
    min_len = min(len(v) for v in returns.values())
    aligned = {s: returns[s][-min_len:] for s in syms}

    # Correlation matrix
    matrix = np.corrcoef([aligned[s] for s in syms])

    # Upper triangle pairs (excluding diagonal)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((syms[i], syms[j], float(matrix[i, j])))

    if not pairs:
        return None

    corr_values = [p[2] for p in pairs]
    avg_corr = float(np.mean(corr_values))
    max_corr = float(np.max(corr_values))
    danger_pairs = [(s1, s2, c) for s1, s2, c in pairs if c > 0.75]

    # Effective N via HHI (Herfindahl–Hirschman Index on eigenvalues)
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        weights = eigenvalues / eigenvalues.sum()
        hhi = float(np.sum(weights ** 2))
        effective_n = round(1 / hhi, 1) if hhi > 0 else float(n)
    except Exception:
        effective_n = float(n)

    if avg_corr >= _CORRELATION_BLOCK_THRESHOLD:
        verdict = "DANGER"
        block = True
    elif avg_corr >= _CORRELATION_ALERT_THRESHOLD:
        verdict = "WARNING"
        block = False
    else:
        verdict = "HEALTHY"
        block = False

    if verdict != "HEALTHY":
        log.warning("portfolio_correlation_elevated",
                    avg=round(avg_corr, 2), verdict=verdict, effective_n=effective_n)

    return CorrelationReport(
        avg_pairwise_correlation=round(avg_corr, 3),
        max_pairwise_correlation=round(max_corr, 3),
        effective_positions=effective_n,
        total_positions=n,
        danger_pairs=[(s1, s2, round(c, 2)) for s1, s2, c in danger_pairs],
        verdict=verdict,
        block_new_trades=block,
    )


def would_increase_correlation(
    new_symbol: str,
    open_symbols: list[str],
    days: int = 30,
    threshold: float = _CORRELATION_BLOCK_THRESHOLD,
) -> tuple[bool, float]:
    """
    Returns (would_breach_threshold, projected_avg_correlation).
    Use before adding a new position.
    """
    if not open_symbols:
        return False, 0.0
    trial = open_symbols + [new_symbol]
    report = compute_portfolio_correlation(trial, days)
    if report is None:
        return False, 0.0
    return report.avg_pairwise_correlation >= threshold, report.avg_pairwise_correlation


def _fetch_returns(symbol: str, days: int) -> Optional[np.ndarray]:
    """Fetch daily returns. Kite → yfinance → None."""
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period=f"{days + 5}d", interval="1d")
        if df is not None and len(df) >= _MIN_BARS:
            closes = df["Close"].values
            return np.diff(closes) / closes[:-1]
    except Exception as e:
        log.debug("correlation_fetch_failed", symbol=symbol, error=str(e))
    return None


def render_correlation_ui() -> None:
    """Streamlit widget for correlation dashboard."""
    import streamlit as st
    from portfolio.state import PortfolioState

    try:
        portfolio = PortfolioState()
        symbols = list(portfolio.positions.keys())
    except Exception:
        symbols = []

    if len(symbols) < 2:
        st.caption("Need 2+ open positions to compute correlation.")
        return

    with st.spinner("Computing portfolio correlation…"):
        report = compute_portfolio_correlation(symbols)

    if report is None:
        st.caption("Insufficient data for correlation analysis.")
        return

    color = {"HEALTHY": "#00d4a0", "WARNING": "#f59e0b", "DANGER": "#ff4b4b"}[report.verdict]

    st.markdown(
        f"<div style='background:{color}11;border:1px solid {color}44;border-radius:12px;"
        f"padding:.8rem 1rem;margin:.5rem 0'>"
        f"<div style='color:{color};font-size:.85rem;font-weight:700'>"
        f"{'✅' if report.verdict == 'HEALTHY' else '⚠️' if report.verdict == 'WARNING' else '🔴'} "
        f"Portfolio Correlation: {report.verdict}</div>"
        f"<div style='color:#c8cfe0;font-size:.8rem;margin-top:.3rem'>"
        f"Avg correlation: {report.avg_pairwise_correlation:.2f} · "
        f"Effective positions: {report.effective_positions:.1f} of {report.total_positions}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if report.danger_pairs:
        st.markdown("**Highly correlated pairs** (move together in a crash):")
        for s1, s2, c in report.danger_pairs:
            st.markdown(f"- `{s1}` ↔ `{s2}`: **{c:.2f}** correlation")

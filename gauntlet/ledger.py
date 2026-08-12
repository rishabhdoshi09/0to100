"""
E1 — Historical Trade Ledger.

Every simulated trade becomes ONE immutable record carrying every field any later
statistic could need, so nothing is ever recomputed (and so a result can never
silently depend on a value that has since changed). `TradeRecord` is frozen; the
`LedgerBuilder` is the sink handed to `signal_backtest.run_backtest(on_trade=…)`,
and `finalize()` enriches each row with the benchmark/factor return over its EXACT
holding window — computed once, from the paired index/factor series.

alpha & beta are POPULATION quantities (a regression over a set of trades), so
they are not meaningful per-trade; each row stores the raw
`benchmark_return_during_trade` the regression consumes, and the strategy-level
alpha/beta live on the gauntlet verdict, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    signal_id: str
    signals: tuple
    entry_datetime: str
    exit_datetime: str
    entry_price: float
    exit_price: float
    holding_period: int
    stop_price: float
    target_price: float
    gross_R: float
    net_R: float
    costs: float
    slippage_used: float
    exit_reason: str
    regime: str
    confidence: float
    calibration_version: str
    benchmark_return_during_trade: float | None = None    # fraction, e.g. 0.021
    factor_returns_during_trade: dict = field(default_factory=dict)

    @property
    def risk_frac(self) -> float:
        """Per-trade risk as a fraction of entry — the denominator that converts a
        market %-move into the trade's own R units (a +2% index move on a trade
        risking 4% is +0.5R of beta)."""
        if self.entry_price <= 0:
            return 0.0
        r = (self.entry_price - self.stop_price) / self.entry_price
        return r if r > 0 else 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _window_return(series, entry_dt, exit_dt):
    """Fractional return of a close `series` (date-indexed) over [entry, exit],
    using as-of lookups so a non-trading date snaps to the last available bar.
    None when either endpoint is missing or the base is non-positive."""
    if series is None or len(series) == 0:
        return None
    try:
        p0 = float(series.asof(entry_dt))
        p1 = float(series.asof(exit_dt))
    except Exception:
        return None
    import math
    if p0 is None or p1 is None or math.isnan(p0) or math.isnan(p1) or p0 <= 0:
        return None
    return p1 / p0 - 1.0


class LedgerBuilder:
    """Collects raw per-trade dicts from the backtest, then `finalize()` freezes
    them into TradeRecords enriched with benchmark & factor window-returns."""

    def __init__(self):
        self._rows: list[dict] = []

    def __call__(self, rec: dict) -> None:          # the on_trade sink
        self._rows.append(rec)

    def __len__(self) -> int:
        return len(self._rows)

    def finalize(self, index_close=None, factor_closes: dict | None = None
                 ) -> list[TradeRecord]:
        """`index_close` = the benchmark's date-indexed close series (e.g. Nifty
        50). `factor_closes` = {name: date-indexed close} for each enabled factor.
        Returns the immutable ledger."""
        factor_closes = factor_closes or {}
        out: list[TradeRecord] = []
        for r in self._rows:
            e_dt, x_dt = r["entry_datetime"], r["exit_datetime"]
            bench = _window_return(index_close, e_dt, x_dt)
            facs = {name: _window_return(s, e_dt, x_dt)
                    for name, s in factor_closes.items()}
            out.append(TradeRecord(
                symbol=r["symbol"],
                signal_id=r.get("signal_id", ""), signals=tuple(r.get("signals", ())),
                entry_datetime=str(e_dt), exit_datetime=str(x_dt),
                entry_price=r["entry_price"], exit_price=r["exit_price"],
                holding_period=r["holding_period"], stop_price=r["stop_price"],
                target_price=r["target_price"], gross_R=r["gross_R"],
                net_R=r["net_R"], costs=r["costs"], slippage_used=r["slippage_used"],
                exit_reason=r["exit_reason"], regime=r["regime"],
                confidence=r.get("confidence", 0.0),
                calibration_version=r.get("calibration_version", ""),
                benchmark_return_during_trade=bench,
                factor_returns_during_trade=facs))
        return out


def per_strategy(ledger: list[TradeRecord]) -> dict:
    """Group the ledger into per-strategy record lists. A trade that carried
    several signals is attributed to EACH — the same trade set the aggregate
    backtest measures for that signal (their non-independence is exactly what the
    effective-N / block-bootstrap stats then correct for)."""
    out: dict[str, list[TradeRecord]] = {}
    for rec in ledger:
        for s in (rec.signals or (rec.signal_id,)):
            if s:
                out.setdefault(s, []).append(rec)
    return out

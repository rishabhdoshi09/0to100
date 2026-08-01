"""Canonical current long-term shortlist service.

The technical pre-screen uses official bhavcopy history. Fundamentals are a
current Screener.in/cache snapshot and are explicitly *not* point-in-time
historical evidence. Missing data lowers coverage and can never be converted to
an optimistic pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

SUCCEEDED = "SUCCEEDED"
NO_CANDIDATES = "NO_CANDIDATES"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
FAILED = "FAILED"


@dataclass(frozen=True)
class LongTermScanReport:
    status: str
    payload: dict = field(default_factory=dict)
    error_code: str = ""
    error_message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (SUCCEEDED, NO_CANDIDATES)


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _band(value: float, bands: list[tuple[float, float]], *, reverse=False) -> float:
    """Piecewise quality points in [0,100]."""
    if reverse:
        for ceiling, points in bands:
            if value <= ceiling:
                return points
        return bands[-1][1]
    for floor, points in bands:
        if value >= floor:
            return points
    return bands[-1][1]


def score_current_fundamentals(fund: Mapping[str, Any] | None, *, sector: str = "") -> dict:
    fund = dict(fund or {})
    financial = any(word in sector.lower() for word in
                    ("bank", "finance", "financial", "insurance", "nbfc"))
    metrics: list[tuple[str, float, float]] = []
    factors: list[str] = []
    risks: list[str] = []

    def add(name: str, weight: float, value: float | None, points: float,
            good: str = "", risk: str = "") -> None:
        if value is None:
            return
        metrics.append((name, weight, max(0.0, min(100.0, points))))
        if good:
            factors.append(good)
        if risk:
            risks.append(risk)

    roce = _f(fund.get("roce"))
    if not financial:
        add("roce", 18, roce, _band(roce, [(20,100),(15,78),(10,48),(0,10)]) if roce is not None else 0,
            f"ROCE {roce:.1f}%" if roce is not None and roce >= 15 else "",
            f"Low ROCE {roce:.1f}%" if roce is not None and roce < 8 else "")
    roe = _f(fund.get("roe"))
    add("roe", 18, roe, _band(roe, [(20,100),(15,78),(10,48),(0,10)]) if roe is not None else 0,
        f"ROE {roe:.1f}%" if roe is not None and roe >= 15 else "",
        f"Low ROE {roe:.1f}%" if roe is not None and roe < 8 else "")

    sales = _f(fund.get("sales_growth_3y"))
    add("sales_growth_3y", 12, sales,
        _band(sales, [(15,100),(10,78),(5,52),(0,25),(-999,0)]) if sales is not None else 0,
        f"3y sales CAGR {sales:.1f}%" if sales is not None and sales >= 10 else "",
        f"Sales shrinking {sales:.1f}%" if sales is not None and sales < 0 else "")
    profit = _f(fund.get("profit_growth_3y"))
    add("profit_growth_3y", 12, profit,
        _band(profit, [(15,100),(10,78),(5,52),(0,25),(-999,0)]) if profit is not None else 0,
        f"3y profit CAGR {profit:.1f}%" if profit is not None and profit >= 10 else "",
        f"Profit shrinking {profit:.1f}%" if profit is not None and profit < 0 else "")

    debt = _f(fund.get("debt_to_equity"))
    if not financial:
        add("debt_to_equity", 12, debt,
            _band(debt, [(0.3,100),(0.7,82),(1.2,50),(2.0,20),(999,0)], reverse=True)
            if debt is not None else 0,
            f"Low debt/equity {debt:.2f}" if debt is not None and debt <= 0.7 else "",
            f"High debt/equity {debt:.2f}" if debt is not None and debt > 2 else "")

    conversion = _f(fund.get("cfo_to_pat"))
    if not financial:
        add("cfo_to_pat", 10, conversion,
            _band(conversion, [(1.0,100),(0.8,82),(0.5,50),(0,20),(-999,0)])
            if conversion is not None else 0,
            f"Cash conversion {conversion:.2f}× PAT" if conversion is not None and conversion >= 0.8 else "",
            f"Weak cash conversion {conversion:.2f}×" if conversion is not None and conversion < 0.5 else "")

    interest = _f(fund.get("interest_coverage"))
    if not financial:
        add("interest_coverage", 5, interest,
            _band(interest, [(5,100),(3,78),(1.5,42),(0,0)]) if interest is not None else 0,
            f"Interest coverage {interest:.1f}×" if interest is not None and interest >= 3 else "",
            f"Weak interest coverage {interest:.1f}×" if interest is not None and interest < 1.5 else "")

    pledge = _f(fund.get("promoter_pledge"))
    add("promoter_pledge", 10, pledge,
        _band(pledge, [(0,100),(5,78),(10,50),(20,20),(999,0)], reverse=True)
        if pledge is not None else 0,
        "No promoter pledge" if pledge == 0 else "",
        f"Promoter pledge {pledge:.1f}%" if pledge is not None and pledge > 10 else "")

    pe = _f(fund.get("pe"))
    add("pe", 8, pe,
        _band(pe, [(20,100),(30,80),(45,55),(60,30),(9999,10)], reverse=True)
        if pe is not None and pe > 0 else 0,
        f"P/E {pe:.1f}" if pe is not None and 0 < pe <= 30 else "",
        f"Rich valuation P/E {pe:.1f}" if pe is not None and pe > 50 else "")

    promoter = _f(fund.get("promoter_holding"))
    add("promoter_holding", 5, promoter,
        _band(promoter, [(50,100),(35,72),(20,40),(0,15)]) if promoter is not None else 0,
        f"Promoter holding {promoter:.1f}%" if promoter is not None and promoter >= 50 else "")

    coverage_weight = sum(weight for _, weight, _ in metrics)
    total_weight = 65.0 if financial else 110.0
    coverage = min(1.0, coverage_weight / total_weight) if total_weight else 0.0
    score = (sum(weight * points for _, weight, points in metrics) / coverage_weight
             if coverage_weight else 0.0)
    severe = any(
        (pledge is not None and pledge > 20,
         debt is not None and not financial and debt > 2.0,
         profit is not None and profit < -10,
         roe is not None and roe < 5,
         roce is not None and not financial and roce < 5,
         interest is not None and not financial and interest < 1.5)
    )
    return {
        "score": round(score, 1), "coverage": round(coverage, 3),
        "factors": factors[:6], "risks": risks[:6], "severe_red_flag": severe,
        "metrics": {name: fund.get(name) for name, _, _ in metrics},
        "financial_sector": financial,
    }


def _default_fundamental_provider(symbol: str, refresh: bool) -> Mapping[str, Any] | None:
    if refresh:
        from fundamentals.fetcher import get_deep_fundamentals
        return get_deep_fundamentals(symbol, force_refresh=True)
    from fundamentals.cache import FundamentalsCache
    return FundamentalsCache().get(symbol)


def _prepare_official_history() -> dict:
    """Load or build the single canonical official-history store.

    API and supervisor are separate processes, so an in-memory readiness check is
    insufficient. First load the persisted cache, then rebuild from existing local
    CSVs, and only then invoke the canonical downloader/builder.
    """
    from data.bhavcopy_runtime import ensure_loaded

    state = ensure_loaded(rebuild_from_local=True)
    if state.get("ready"):
        return state
    from data.bhavcopy_store import build_store
    build_store()
    return ensure_loaded(rebuild_from_local=True)


def run_long_term_scan(
    *,
    symbols=None,
    scope: str = "nifty500",
    refresh_fundamentals: bool = False,
    technical_scanner: Callable[..., list[dict]] | None = None,
    fundamental_provider: Callable[[str, bool], Mapping[str, Any] | None] | None = None,
    sector_lookup: Callable[[str], str] | None = None,
    save: bool = True,
    top: int = 40,
) -> LongTermScanReport:
    default_technical = technical_scanner is None
    technical_scanner = technical_scanner or __import__(
        "scan.long_term", fromlist=["scan_long_term"]).scan_long_term
    history: dict = {}
    if default_technical:
        try:
            history = _prepare_official_history()
            if not history.get("ready"):
                return LongTermScanReport(
                    DATA_UNAVAILABLE,
                    error_code="BHAVCOPY_NOT_READY",
                    error_message=(
                        "official bhavcopy history is not ready for long-term screening · "
                        f"csv_files={history.get('csv_files', 0)} · "
                        f"sessions={history.get('sessions', 0)} · "
                        f"cache={history.get('cache_exists', False)}"
                    ),
                )
        except Exception as exc:
            return LongTermScanReport(DATA_UNAVAILABLE, error_code="BHAVCOPY_STATUS_ERROR",
                                      error_message=str(exc))
    fundamental_provider = fundamental_provider or _default_fundamental_provider
    if sector_lookup is None:
        try:
            from scan.sector_heat import sector_of
            sector_lookup = sector_of
        except Exception:
            sector_lookup = lambda _symbol: ""

    if symbols is None and scope == "nifty500":
        try:
            from data.nse_universe import get_nifty500_universe
            symbols = get_nifty500_universe()
        except Exception:
            symbols = None
    technical_limit = max(top, 30) if refresh_fundamentals else max(top * 2, 60)
    try:
        technical = technical_scanner(symbols=symbols, min_score=45, top=technical_limit,
                                      include_watch=True)
    except TypeError:
        technical = technical_scanner(symbols=symbols, min_score=45, top=technical_limit)
    except Exception as exc:
        return LongTermScanReport(FAILED, error_code="LONG_TERM_TECHNICAL_ERROR",
                                  error_message=str(exc))
    if not technical:
        payload = _payload([], scope=scope, refresh=refresh_fundamentals, history=history)
        if save:
            from product.long_term_store import save_long_term_scan
            save_long_term_scan(payload)
        return LongTermScanReport(NO_CANDIDATES, payload)

    records: list[dict] = []
    for technical_row in technical:
        row = dict(technical_row)
        symbol = str(row.get("symbol", "") or "").upper()
        sector = str(sector_lookup(symbol) or "Unknown")
        raw = None
        error = ""
        try:
            raw = fundamental_provider(symbol, bool(refresh_fundamentals))
        except Exception as exc:
            error = type(exc).__name__
        fund = {}
        if raw:
            try:
                from screener.engine import _extract_fundamentals
                fund = _extract_fundamentals(dict(raw))
            except Exception as exc:
                error = type(exc).__name__
        fq = score_current_fundamentals(fund, sector=sector)
        technical_score = float(row.get("score", 0) or 0)
        combined = round(technical_score * 0.45 + fq["score"] * 0.55, 1)
        pe = _f(fund.get("pe"))

        if fq["severe_red_flag"]:
            classification = "AVOID_REVIEW"
        elif fq["coverage"] < 0.50:
            classification = "NEEDS_FUNDAMENTALS"
        elif fq["score"] >= 70 and technical_score >= 62 and combined >= 70:
            classification = "QUALITY_BUT_EXPENSIVE" if pe is not None and pe > 50 else "QUALITY_COMPOUNDER"
        elif fq["score"] >= 60 and technical_score >= 55 and combined >= 63:
            classification = "GARP_CANDIDATE"
        elif combined >= 52:
            classification = "LONG_TERM_WATCH"
        else:
            classification = "AVOID_REVIEW"

        extension = float(row.get("extension_pct", 0) or 0)
        timing = ("WAIT_FOR_BASE" if extension >= 35 else
                  "ACCUMULATE_ON_PULLBACK" if extension >= 20 else
                  "TECHNICALLY_FAVORABLE")
        factors = list(dict.fromkeys(list(row.get("factors", []) or [])[:4] + fq["factors"]))
        risks = list(dict.fromkeys(fq["risks"] +
                    (["Current fundamentals unavailable or incomplete"] if fq["coverage"] < 0.50 else []) +
                    (["Price extended above 200-DMA"] if extension >= 35 else [])))
        records.append({
            **row, "symbol": symbol, "sector": sector,
            "technical_score": round(technical_score, 1),
            "fundamental_score": fq["score"], "fundamental_coverage": fq["coverage"],
            "combined_score": combined, "classification": classification,
            "timing": timing, "fundamentals": fund, "fundamental_error": error,
            "quality_factors": factors[:8], "risk_flags": risks[:8],
            "fundamentals_point_in_time": False,
        })

    priority = {"QUALITY_COMPOUNDER": 0, "GARP_CANDIDATE": 1,
                "QUALITY_BUT_EXPENSIVE": 2, "LONG_TERM_WATCH": 3,
                "NEEDS_FUNDAMENTALS": 4, "AVOID_REVIEW": 5}
    records.sort(key=lambda r: (priority.get(r["classification"], 9),
                                -float(r["combined_score"]), r["symbol"]))
    payload = _payload(records[:top], scope=scope, refresh=refresh_fundamentals, history=history)
    if save:
        from product.long_term_store import save_long_term_scan
        save_long_term_scan(payload)
    return LongTermScanReport(SUCCEEDED if records else NO_CANDIDATES, payload)


def _payload(records: list[dict], *, scope: str, refresh: bool,
             history: Mapping[str, Any] | None = None) -> dict:
    summary = {name.lower(): sum(1 for r in records if r.get("classification") == name)
               for name in ("QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
                            "LONG_TERM_WATCH", "NEEDS_FUNDAMENTALS", "AVOID_REVIEW")}
    covered = [r for r in records if float(r.get("fundamental_coverage", 0) or 0) >= 0.50]
    summary.update({"candidates": len(records), "fundamentally_covered": len(covered),
                    "fundamental_errors": sum(1 for r in records if r.get("fundamental_error")),
                    "coverage_pct": round(len(covered) / len(records) * 100, 1) if records else 0.0})
    return {
        "schema_version": 1, "scanned_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope, "records": records, "summary": summary,
        "history": dict(history or {}),
        "fundamentals_source": "Screener.in current snapshot/cache",
        "fundamentals_refreshed": bool(refresh), "fundamentals_point_in_time": False,
        "disclaimer": ("Current long-term shortlist only. Fundamentals are not publication-dated "
                       "and must never be substituted into historical backtests."),
    }

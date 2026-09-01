"""Decision-consistency gate for the current long-term shortlist.

This module does not create a second scanner. It wraps the canonical
``scan.long_term_service.run_long_term_scan`` and enforces four product truths:

1. A Nifty-500 long-term run is actually restricted to Nifty 500 symbols even
   when the technical pre-screen reuses the saved whole-market scan.
2. Sparse fundamentals may never manufacture a high-confidence composite score.
3. Financial companies are not judged with industrial ROCE/CFO/debt rules just
   because the legacy sector map misses a symbol.
4. The final user-facing verdict reconciles technical timing, fundamental
   classification and evidence coverage; contradictory ``LONG_TERM_BUY`` labels
   are removed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

_FINANCIAL_SYMBOLS = frozenset({
    "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK",
    "BANKBARODA", "CANBK", "PNB", "UNIONBANK", "BANDHANBNK", "FEDERALBNK",
    "IDFCFIRSTB", "KARURVYSYA", "RBLBANK", "CUB", "DCBBANK", "UJJIVANSFB",
    "EQUITASBNK", "AUBANK", "ESAFSFB", "J&KBANK", "SOUTHBANK", "TMB",
    "CSBBANK", "KTKBANK", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN",
    "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM", "RECLTD", "PFC", "IRFC",
    "AAVAS", "APTUS", "HOMEFIRST", "LICHSGFIN", "CANFINHOME", "REPCO",
    "PNBHOUSING", "CREDITACC", "SPANDANA", "FUSION", "ARMANFIN", "SBFC",
    "UGROCAP", "MASFIN", "SUNDARMFIN", "M&MFIN", "SRTRANSFIN", "IIFL",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI", "ICICIGI", "NIACL", "STARHEALTH",
    "GODIGIT", "POLICYBZR",
})

_CLASS_PRIORITY = {
    "QUALITY_COMPOUNDER": 0,
    "GARP_CANDIDATE": 1,
    "QUALITY_BUT_EXPENSIVE": 2,
    "LONG_TERM_WATCH": 3,
    "NEEDS_FUNDAMENTALS": 4,
    "AVOID_REVIEW": 5,
}


def hardened_sector_of(symbol: str) -> str:
    """Best available sector with an explicit financial fallback.

    The legacy sector parser only sees names physically listed under comment
    groups inside the NIFTY500 extension block, so several NIFTY50/NIFTY100
    financials can otherwise become ``Unknown``.
    """
    clean = str(symbol or "").strip().upper()
    try:
        from scan.sector_heat import sector_of
        sector = str(sector_of(clean) or "").strip()
    except Exception:
        sector = ""
    if sector:
        return sector
    if clean in _FINANCIAL_SYMBOLS:
        return "Banking & Finance"
    return "Unknown"


def evidence_adjusted_combined(technical_score: float, fundamental_score: float, coverage: float) -> float:
    """Composite where missing evidence contributes no phantom points.

    This is deliberately a *decision-evidence* score, not an estimate of company
    quality. The raw fundamental score is preserved separately so missing data is
    not silently interpreted as a bad business.
    """
    tech = max(0.0, min(100.0, float(technical_score or 0.0)))
    fund = max(0.0, min(100.0, float(fundamental_score or 0.0)))
    cov = max(0.0, min(1.0, float(coverage or 0.0)))
    return round(0.45 * tech + 0.55 * fund * cov, 1)


def reconcile_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return one internally consistent long-term decision record."""
    row = dict(record or {})
    technical = float(row.get("technical_score") or row.get("score") or 0.0)
    fund_raw = float(row.get("fundamental_score") or 0.0)
    coverage = float(row.get("fundamental_coverage") or 0.0)
    previous_combined = row.get("combined_score")
    decision_score = evidence_adjusted_combined(technical, fund_raw, coverage)
    classification = str(row.get("classification") or "").strip().upper()
    timing = str(row.get("timing") or "").strip().upper()
    chase = bool(row.get("chase_risk"))

    if classification == "AVOID_REVIEW":
        verdict = "AVOID_REVIEW"
    elif classification == "NEEDS_FUNDAMENTALS" or coverage < 0.50:
        verdict = "NEEDS_FUNDAMENTALS"
    elif timing == "WAIT_FOR_BASE" or chase:
        verdict = "WAIT_FOR_BASE"
    elif classification == "QUALITY_BUT_EXPENSIVE":
        verdict = "WATCH_VALUATION"
    elif classification in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
        verdict = "LONG_TERM_CANDIDATE"
    else:
        verdict = "WATCH"

    row["technical_verdict"] = row.get("verdict") or ""
    row["combined_score_unadjusted"] = previous_combined
    row["combined_score"] = decision_score
    row["decision_score"] = decision_score
    row["decision_coverage"] = round(max(0.0, min(1.0, coverage)), 3)
    row["verdict"] = verdict
    if verdict != row.get("technical_verdict"):
        row["verdict_reconciled"] = True
    return row


def postprocess_report(report: Any) -> Any:
    """Reconcile records without changing the canonical report type."""
    payload = dict(getattr(report, "payload", {}) or {})
    records = [reconcile_record(item) for item in list(payload.get("records") or []) if isinstance(item, dict)]
    records.sort(key=lambda r: (
        _CLASS_PRIORITY.get(str(r.get("classification") or "").upper(), 9),
        -float(r.get("decision_score") or 0.0),
        str(r.get("symbol") or ""),
    ))
    payload["records"] = records
    payload["decision_consistency_gate"] = {
        "enabled": True,
        "coverage_adjusted_scoring": True,
        "verdict_reconciled": True,
        "financial_sector_fallback": True,
        "nifty500_scope_enforced": True,
    }
    if hasattr(report, "__dataclass_fields__"):
        try:
            return replace(report, payload=payload)
        except Exception:
            pass
    try:
        report.payload = payload
    except Exception:
        pass
    return report


def install() -> None:
    """Install once around the canonical service; no second scanner is created."""
    from scan import long_term_service as service

    if getattr(service, "_decision_consistency_installed", False):
        return
    original: Callable[..., Any] = service.run_long_term_scan

    def wrapped_run_long_term_scan(*args: Any, **kwargs: Any) -> Any:
        scope = str(kwargs.get("scope") or "nifty500").strip().lower()
        restrict_nifty500 = (
            scope == "nifty500"
            and kwargs.get("symbols") is None
            and kwargs.get("technical_scanner") is None
        )
        if kwargs.get("sector_lookup") is None:
            kwargs["sector_lookup"] = hardened_sector_of
        # Do not inject NIFTY500 into the technical projector. That turned a
        # saved-scan projection into an empty shortlist for any symbol outside
        # the static list and looked like a scanner miss. Restrict AFTER
        # projecting the saved scan, and never start a second OHLCV walk.
        report = postprocess_report(original(*args, **kwargs))
        if restrict_nifty500:
            try:
                from data.nse_universe import NIFTY500
                allowed = {str(s).strip().upper() for s in NIFTY500}
            except Exception:
                allowed = set()
            if allowed:
                payload = dict(getattr(report, "payload", {}) or {})
                payload["records"] = [
                    row for row in list(payload.get("records") or [])
                    if str((row or {}).get("symbol") or "").strip().upper() in allowed
                ]
                payload["nifty500_scope_enforced"] = True
                if hasattr(report, "__dataclass_fields__"):
                    try:
                        report = replace(report, payload=payload)
                    except Exception:
                        try:
                            report.payload = payload
                        except Exception:
                            pass
                else:
                    try:
                        report.payload = payload
                    except Exception:
                        pass
        return report

    service.run_long_term_scan = wrapped_run_long_term_scan
    service._decision_consistency_installed = True

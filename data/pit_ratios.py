"""Derived PIT ratios. Computed at read time from point-in-time inputs.

Never persist a ratio without the underlying inputs + calculation version.
Never treat growth as an earnings surprise.
"""
from __future__ import annotations

from typing import Any

CALC_VERSION = "pit_ratios.v2"

# Canonical input aliases → schema fields on a fundamentals row
_REV = "revenue_from_operations"
_PAT = "profit_after_tax"
_PBT = "profit_before_tax"
_EPS = "basic_eps"
_EQUITY = "paid_up_equity_capital"
_DE = "debt_equity_ratio"


def _f(row: dict[str, Any] | None, key: str) -> float | None:
    if not row:
        return None
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _pct(cur: float | None, prev: float | None) -> float | None:
    if cur is None or prev is None or prev == 0:
        return None
    return cur / prev - 1.0


def _margin(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def derive_ratios(
    current: dict[str, Any] | None,
    prior: dict[str, Any] | None = None,
    *,
    calc_version: str = CALC_VERSION,
) -> dict[str, Any]:
    """Ratios + provenance. Missing stays None (never 0-filled)."""
    rev = _f(current, _REV)
    pat = _f(current, _PAT)
    pbt = _f(current, _PBT)
    eps = _f(current, _EPS)
    equity = _f(current, _EQUITY)
    de = _f(current, _DE)
    prev_rev = _f(prior, _REV)
    prev_eps = _f(prior, _EPS)
    prev_pat = _f(prior, _PAT)

    # YoY when prior is the same fiscal slot a year earlier; otherwise the
    # nearest earlier known period (may be QoQ). Callers must not treat
    # mixed-span growth as a single named signal.
    qoq_ok = _qoq_meaningful(current, prior)
    yoy_ok = _yoy_meaningful(current, prior)
    values = {
        "revenue_growth": _pct(rev, prev_rev),
        "revenue_growth_yoy": _pct(rev, prev_rev) if yoy_ok else None,
        "revenue_growth_qoq": _pct(rev, prev_rev) if qoq_ok else None,
        "eps_growth": _pct(eps, prev_eps),
        "pat_growth": _pct(pat, prev_pat),
        "pat_growth_yoy": _pct(pat, prev_pat) if yoy_ok else None,
        "operating_margin": _margin(_f(current, "operating_profit"), rev),
        "pat_margin": _margin(pat, rev),
        "pbt_margin": _margin(pbt, rev),
        "roe": _margin(pat, equity),  # crude: paid-up equity, not average book
        "debt_equity": de,
        "cfo_pat": None,  # requires cash-flow ledger
        "fcf_margin": None,
        "roce": None,
    }
    return {
        "calc_version": calc_version,
        "values": values,
        "inputs": {
            "current_row_id": (current or {}).get("row_id"),
            "prior_row_id": (prior or {}).get("row_id"),
            "current_available_at": (current or {}).get("available_at"),
            "prior_available_at": (prior or {}).get("available_at"),
            "current_period_end": (current or {}).get("period_end"),
            "prior_period_end": (prior or {}).get("period_end"),
            "fields_used": {
                "revenue": rev,
                "pat": pat,
                "pbt": pbt,
                "eps": eps,
                "paid_up_equity": equity,
                "debt_equity": de,
            },
        },
        "quality": {
            "roe": "PIT_DEGRADED" if values["roe"] is not None else "UNKNOWN",
            "revenue_growth": "PIT_STRONG" if values["revenue_growth"] is not None else "UNKNOWN",
            "eps_growth": "PIT_STRONG" if values["eps_growth"] is not None else "UNKNOWN",
            "cfo_pat": "UNUSABLE",
            "fcf_margin": "UNUSABLE",
            "roce": "UNUSABLE",
            "note": (
                "ROE uses paid-up equity when book equity is absent (degraded). "
                "CFO/FCF/ROCE require a cash-flow + capital-employed ledger. "
                "QoQ/YoY named fields are None unless period alignment is meaningful."
            ),
        },
        "not_earnings_surprise": True,
        "period_alignment": {
            "qoq_meaningful": qoq_ok,
            "yoy_meaningful": yoy_ok,
            "current_kind": (current or {}).get("period_kind"),
            "prior_kind": (prior or {}).get("period_kind"),
        },
    }


def _span_days(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    try:
        from data.period_alignment import span_days
        return span_days(row.get("period_start"), row.get("period_end"))
    except Exception:
        return None


def _yoy_meaningful(current: dict | None, prior: dict | None) -> bool:
    if not current or not prior:
        return False
    if str(current.get("period_kind") or "") == "annual" and str(prior.get("period_kind") or "") == "annual":
        return True
    if current.get("quarterly_usable") is False or prior.get("quarterly_usable") is False:
        return False
    try:
        from datetime import date
        a = date.fromisoformat(str(current.get("period_end"))[:10])
        b = date.fromisoformat(str(prior.get("period_end"))[:10])
        gap = abs((a - b).days)
    except Exception:
        return False
    return 330 <= gap <= 400


def _qoq_meaningful(current: dict | None, prior: dict | None) -> bool:
    if not current or not prior:
        return False
    if current.get("quarterly_usable") is not True or prior.get("quarterly_usable") is not True:
        return False
    try:
        from datetime import date
        a = date.fromisoformat(str(current.get("period_end"))[:10])
        b = date.fromisoformat(str(prior.get("period_end"))[:10])
        gap = abs((a - b).days)
    except Exception:
        return False
    return 70 <= gap <= 110


def lineage_for(ratio_name: str, derived: dict[str, Any]) -> dict[str, Any]:
    """Lightweight provenance for a named derived feature."""
    return {
        "feature": ratio_name,
        "calc_version": derived.get("calc_version"),
        "value": (derived.get("values") or {}).get(ratio_name),
        "source_rows": {
            "current": (derived.get("inputs") or {}).get("current_row_id"),
            "prior": (derived.get("inputs") or {}).get("prior_row_id"),
        },
        "availability": {
            "current": (derived.get("inputs") or {}).get("current_available_at"),
            "prior": (derived.get("inputs") or {}).get("prior_available_at"),
        },
        "quality": (derived.get("quality") or {}).get(ratio_name),
        "not_earnings_surprise": True,
    }

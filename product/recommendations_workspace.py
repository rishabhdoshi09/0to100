"""Reco-style recommendation desk — projections only, no new scans or scrapes.

Maps QuantTerm evidence into named research categories (Wealth Builders,
Super Trends, Momentum Breakouts, Recovery Setups) plus Active/Closed
lifecycle strips from long-term + signal-outcome trackers.

Honest rules:
  - Empty category → empty list (never fabricate picks)
  - Prices/upside from real entry/target/price fields only
  - CMP freshness tags come from live_technicals when available
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.radar_workspace import (
    enrich_long_term_row,
    enrich_scan_row,
    is_long_term_pick,
    is_sniper_breakout_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "logs" / "product" / "market_reports"

# Reco-Wealth analogues — QuantTerm evidence names (not a brand clone).
CATEGORIES: tuple[dict[str, str], ...] = (
    {
        "id": "wealth_builders",
        "label": "Wealth Builders",
        "blurb": "Quality compounders with readable fundamentals — long-term research, not day trades.",
        "icon": "compound",
    },
    {
        "id": "super_trends",
        "label": "Super Trends",
        "blurb": "Names riding trend and relative strength without chase/extension flags.",
        "icon": "trend",
    },
    {
        "id": "momentum_breakouts",
        "label": "Momentum Breakouts",
        "blurb": "Sniper-grade and confirmed breakouts — volume, structure, RSI hard ceiling.",
        "icon": "breakout",
    },
    {
        "id": "recovery_setups",
        "label": "Recovery Setups",
        "blurb": "Accumulation / double-bottom style recovery patterns when the scan finds them.",
        "icon": "recovery",
    },
)

_RECOVERY_SIGNALS = frozenset({
    "DOUBLE_BOTTOM", "ACCUMULATION", "CUP_HANDLE", "POCKET_PIVOT", "NR7",
})
_TREND_SIGNALS = frozenset({
    "MOMENTUM", "GOLDEN_CROSS", "TRENDING", "HIGH_TIGHT_FLAG",
})


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(x).upper() for x in (row.get("signals") or [])}


def risk_tier(row: Mapping[str, Any]) -> str:
    """Low / Medium / High from existing flags — never invented."""
    if bool(row.get("chase_risk")):
        return "High"
    label = str(row.get("risk_label") or "").lower()
    if "chase" in label or "avoid" in label or "high" in label:
        return "High"
    rsi = _f(row.get("rsi"))
    if rsi >= 75:
        return "High"
    if rsi >= 65 or "pullback" in label or "review" in label or "medium" in label:
        return "Medium"
    flags = row.get("risk_flags") or []
    if flags:
        return "Medium"
    return "Low"


def upside_metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
    """% move from entry and remaining room to target — None when inputs missing."""
    price = _f(row.get("price"))
    entry = _f(row.get("entry") or row.get("entry_price"))
    target = _f(row.get("target") or row.get("target_price"))
    from_entry = None
    to_target = None
    if entry > 0 and price > 0:
        from_entry = round((price / entry - 1.0) * 100.0, 1)
    if target > 0 and price > 0:
        to_target = round((target / price - 1.0) * 100.0, 1)
    return {
        "upside_from_entry_pct": from_entry,
        "upside_to_target_pct": to_target,
        "entry": entry or None,
        "target": target or None,
        "cmp": price or None,
    }


def _action_badge(row: Mapping[str, Any]) -> str:
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
        return "Buy"
    if "breakout" in status.lower() or str(row.get("breakout_grade") or "").upper() in {"A", "B"}:
        return "Watch"
    cls = str(row.get("classification") or "")
    if cls in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE"}:
        return "Hold / Research"
    return "Watch"


def card_from_row(row: Mapping[str, Any], *, category_id: str, category_label: str) -> dict[str, Any]:
    ups = upside_metrics(row)
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "category_id": category_id,
        "category_label": category_label,
        "action_badge": _action_badge(row),
        "risk_tier": risk_tier(row),
        "risk_label": str(row.get("risk_label") or risk_tier(row)),
        "setup_label": str(row.get("setup_label") or row.get("status") or row.get("classification") or ""),
        "sector": str(row.get("sector") or "—"),
        "score": _f(row.get("score") or row.get("combined_score")),
        "rsi": _f(row.get("rsi")) or None,
        "volume_ratio": _f(row.get("volume_ratio")) or None,
        "price_tag": str(row.get("price_tag") or ""),
        "tech_source": str(row.get("tech_source") or ""),
        "reason": str(row.get("reason") or ""),
        "lifecycle": "active",
        **ups,
    }


def _is_super_trend(row: Mapping[str, Any]) -> bool:
    if bool(row.get("chase_risk")):
        return False
    sigs = _signals(row)
    if not (sigs & _TREND_SIGNALS) and "MOMENTUM" not in sigs:
        # Momentum state from enrich also qualifies.
        state = str(row.get("momentum_state") or "")
        if state not in {"strong_actionable", "steady_leadership", "improving"}:
            return False
    if str(row.get("verdict") or "").upper() == "AVOID":
        return False
    # Prefer names still in trend structure.
    if row.get("above_sma50") is False and row.get("above_sma200") is False:
        return False
    return _f(row.get("score")) >= 50 or _f(row.get("momentum_5d")) >= 3


def _is_recovery(row: Mapping[str, Any]) -> bool:
    return bool(_signals(row) & _RECOVERY_SIGNALS)


def _bucket_rows(
    scan_rows: Sequence[Mapping[str, Any]],
    lt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    wealth: list[dict[str, Any]] = []
    for r in lt_rows:
        if not is_long_term_pick(r):
            continue
        wealth.append(card_from_row(r, category_id="wealth_builders", category_label="Wealth Builders"))
    wealth.sort(key=lambda c: (-_f(c.get("score")), c["symbol"]))

    trends: list[dict[str, Any]] = []
    breakouts: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    for r in scan_rows:
        if _is_super_trend(r):
            trends.append(card_from_row(r, category_id="super_trends", category_label="Super Trends"))
        if is_sniper_breakout_candidate(r) or str(r.get("breakout_grade") or "").upper() in {"A", "B"}:
            breakouts.append(card_from_row(r, category_id="momentum_breakouts", category_label="Momentum Breakouts"))
        if _is_recovery(r):
            recovery.append(card_from_row(r, category_id="recovery_setups", category_label="Recovery Setups"))

    trends.sort(key=lambda c: (-_f(c.get("score")), c["symbol"]))
    breakouts.sort(key=lambda c: (-_f(c.get("score")), c["symbol"]))
    recovery.sort(key=lambda c: (-_f(c.get("score")), c["symbol"]))
    return {
        "wealth_builders": wealth[:40],
        "super_trends": trends[:40],
        "momentum_breakouts": breakouts[:40],
        "recovery_setups": recovery[:40],
    }


def _tracker_lifecycle() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    try:
        from core.long_term_tracker import active_picks, exited_picks
        for p in active_picks():
            price = _f(p.get("last_price") or p.get("entry_price"))
            entry = _f(p.get("entry_price"))
            card = {
                "symbol": str(p.get("symbol") or "").upper(),
                "company": str(p.get("symbol") or ""),
                "category_id": "wealth_builders",
                "category_label": "Wealth Builders",
                "action_badge": "Tracked",
                "risk_tier": "Medium",
                "risk_label": "Long-term tracked",
                "setup_label": "ACTIVE long-term pick",
                "sector": "—",
                "score": _f(p.get("score")),
                "lifecycle": "active",
                "source": "long_term_tracker",
                "cmp": price or None,
                "entry": entry or None,
                "target": None,
                "upside_from_entry_pct": (
                    round((price / entry - 1.0) * 100.0, 1) if entry > 0 and price > 0 else None
                ),
                "upside_to_target_pct": None,
                "reason": str(p.get("thesis") or "")[:160],
            }
            active.append(card)
        for p in exited_picks(limit=40):
            entry = _f(p.get("entry_price"))
            exit_px = _f(p.get("exit_price") or p.get("last_price"))
            closed.append({
                "symbol": str(p.get("symbol") or "").upper(),
                "company": str(p.get("symbol") or ""),
                "category_id": "wealth_builders",
                "category_label": "Wealth Builders",
                "action_badge": "Closed",
                "risk_tier": "—",
                "risk_label": "Exited",
                "setup_label": "EXITED long-term pick",
                "sector": "—",
                "score": _f(p.get("score")),
                "lifecycle": "closed",
                "source": "long_term_tracker",
                "cmp": exit_px or None,
                "entry": entry or None,
                "target": None,
                "upside_from_entry_pct": _f(p.get("return_pct")) if p.get("return_pct") is not None else (
                    round((exit_px / entry - 1.0) * 100.0, 1) if entry > 0 and exit_px > 0 else None
                ),
                "upside_to_target_pct": None,
                "reason": str(p.get("exit_reason") or p.get("thesis") or "")[:160],
            })
    except Exception:
        pass

    try:
        from core.signal_outcome_tracker import get_recent_signals
        for s in get_recent_signals(limit=80):
            worked = s.get("worked")
            entry = _f(s.get("entry_price"))
            target = _f(s.get("target_price"))
            outcome_px = _f(s.get("outcome_price"))
            base = {
                "symbol": str(s.get("symbol") or "").upper(),
                "company": str(s.get("symbol") or ""),
                "category_id": "momentum_breakouts",
                "category_label": "Momentum Breakouts",
                "risk_tier": "Medium",
                "risk_label": str(s.get("signal_type") or "signal"),
                "setup_label": str(s.get("signal_type") or "Signal"),
                "sector": "—",
                "score": 0.0,
                "source": "signal_outcome_tracker",
                "entry": entry or None,
                "target": target or None,
                "reason": f"Logged {s.get('logged_at') or ''}".strip(),
            }
            if worked is None:
                active.append({
                    **base,
                    "action_badge": "Open",
                    "lifecycle": "active",
                    "cmp": entry or None,
                    "upside_from_entry_pct": None,
                    "upside_to_target_pct": (
                        round((target / entry - 1.0) * 100.0, 1) if entry > 0 and target > 0 else None
                    ),
                })
            else:
                closed.append({
                    **base,
                    "action_badge": "Win" if worked == 1 else ("Void" if worked == -1 else "Loss"),
                    "lifecycle": "closed",
                    "cmp": outcome_px or None,
                    "upside_from_entry_pct": (
                        round(_f(s.get("outcome_pct")), 1) if s.get("outcome_pct") is not None else None
                    ),
                    "upside_to_target_pct": None,
                })
    except Exception:
        pass

    active.sort(key=lambda c: (c.get("symbol") or ""))
    closed.sort(key=lambda c: (c.get("symbol") or ""))
    return active, closed


def build_recommendations_workspace(
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    refresh_technicals: bool = True,
) -> dict[str, Any]:
    """Project recommendation categories + lifecycle from persisted product state."""
    scan = dict(scan_payload or {})
    lt = dict(long_term_payload or {})
    scan_at = str(scan.get("scanned_at") or "")
    lt_at = str(lt.get("scanned_at") or "")

    scan_rows = [enrich_scan_row(dict(r), scanned_at=scan_at) for r in (scan.get("records") or [])]
    lt_rows = [enrich_long_term_row(dict(r), scanned_at=lt_at) for r in (lt.get("records") or [])]

    if refresh_technicals and scan_rows:
        try:
            from product.live_technicals import refresh_rows_technicals
            # Refresh head for CMP/RSI honesty on the desk.
            head = refresh_rows_technicals(scan_rows[:120], bulk_overlay=True)
            by = {str(r.get("symbol")).upper(): r for r in head}
            scan_rows = [by.get(str(r.get("symbol")).upper(), r) for r in scan_rows]
        except Exception:
            pass

    buckets = _bucket_rows(scan_rows, lt_rows)
    active, closed = _tracker_lifecycle()

    categories = []
    for meta in CATEGORIES:
        cards = list(buckets.get(meta["id"]) or [])
        categories.append({
            **meta,
            "count": len(cards),
            "cards": cards,
            "empty_detail": (
                "No matching setups in the current scan / long-term shortlist."
                if not cards else ""
            ),
        })

    cmp_note = (
        "CMP uses live Kite/NSE overlay when available; otherwise last official EOD. "
        "Scan signals are research snapshots — check scanned_at."
    )
    if str(scan.get("records_status") or "") == "PRIOR_DAY_SNAPSHOT":
        cmp_note = "Scan file is a PRIOR-DAY SNAPSHOT — run a fresh market scan before acting. " + cmp_note

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": lt_at,
        "records_status": str(scan.get("records_status") or ""),
        "same_ist_day": bool(scan.get("same_ist_day")),
        "cmp_note": cmp_note,
        "categories": categories,
        "lifecycle": {
            "active": active[:60],
            "closed": closed[:60],
            "active_count": len(active),
            "closed_count": len(closed),
        },
        "disclaimer": (
            "Research categories from QuantTerm evidence — not broker recommendations, "
            "not a promise of returns, not Reco Wealth."
        ),
    }


def _persist_pulse(pulse: Mapping[str, Any]) -> Path | None:
    try:
        from core.market_clock import today_ist
        day = today_ist().isoformat()
    except Exception:
        day = datetime.now(timezone.utc).date().isoformat()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"market_pulse_{day}.json"
    try:
        import json
        payload = {
            "id": f"market_pulse_{day}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": day,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pulse": dict(pulse),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path
    except Exception:
        return None


def _list_saved_reports(limit: int = 30) -> list[dict[str, Any]]:
    if not REPORTS_DIR.exists():
        return []
    import json
    rows: list[dict[str, Any]] = []
    for path in sorted(REPORTS_DIR.glob("market_pulse_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "id": str(data.get("id") or path.stem),
                "title": str(data.get("title") or "Market Pulse"),
                "kind": str(data.get("kind") or "market_pulse"),
                "date": str(data.get("date") or ""),
                "created_at": str(data.get("created_at") or ""),
                "is_new": False,
                "summary": _pulse_summary(data.get("pulse") or {}),
                "path": str(path),
            })
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


def _pulse_summary(pulse: Mapping[str, Any]) -> str:
    takes = [str(t) for t in (pulse.get("takeaways") or [])[:2]]
    if takes:
        return " · ".join(takes)
    return str(pulse.get("date") or "Market overview")


def build_market_reports_workspace(*, persist_today: bool = True) -> dict[str, Any]:
    """Chronological Market Pulse list from live street pulse + saved day files."""
    pulse: dict[str, Any] = {}
    error = ""
    try:
        from reports.street_pulse import build_pulse
        pulse = build_pulse() or {}
    except Exception as exc:
        error = str(exc)[:200]

    if pulse and persist_today:
        _persist_pulse(pulse)

    reports = _list_saved_reports()
    if reports:
        reports[0]["is_new"] = True
        reports[0]["badge"] = "New market report"
    elif pulse:
        try:
            from core.market_clock import today_ist
            day = today_ist().isoformat()
        except Exception:
            day = datetime.now(timezone.utc).date().isoformat()
        reports = [{
            "id": f"market_pulse_{day}",
            "title": "Market Pulse",
            "kind": "market_pulse",
            "date": day,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_new": True,
            "badge": "New market report",
            "summary": _pulse_summary(pulse),
            "path": "",
        }]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "Stay on top of the markets",
        "blurb": (
            "Daily Market Pulse from QuantTerm scanners — trends, sector movers, "
            "and breakout context. Assembled from live system state, never invented."
        ),
        "reports": reports,
        "today_pulse": pulse,
        "error": error,
        "disclaimer": "Market reports are research summaries, not trade instructions.",
    }

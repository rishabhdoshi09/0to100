"""Shared breakout quality gates for sniper / best-pick / Telegram / autopilot.

Hard rejects thin tape and obvious technical/fundamental red flags so a
"best candidate" list cannot be filled with 0.1× volume noise. Optional
order-book and concall context is attached when available — never invented,
never a sole reason to promote.
"""
from __future__ import annotations

from typing import Any, Mapping

MIN_VOLUME_RATIO = 1.0
RSI_BLOWOFF = 70.0
MIN_FUND_COVERAGE = 0.50
AVOID_CLASSES = frozenset({"AVOID_REVIEW"})
QUALITY_CLASSES = frozenset({
    "QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
})


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def volume_ratio(row: Mapping[str, Any]) -> float:
    return _f(row.get("volume_ratio") or row.get("rvol") or 0)


def passes_volume_floor(row: Mapping[str, Any], *, min_ratio: float = MIN_VOLUME_RATIO) -> bool:
    """Hard floor: known volume must be ≥ min_ratio. Unknown/zero fails closed."""
    vol = volume_ratio(row)
    if vol <= 0:
        return False
    return vol >= float(min_ratio)


def gate_breakout_quality(
    row: Mapping[str, Any],
    *,
    require_fundamentals: bool = False,
    min_volume: float = MIN_VOLUME_RATIO,
) -> tuple[bool, list[str], dict[str, str]]:
    """Return (ok, reject_reasons, gate_status).

    Hard rejects (ok=False):
      - volume missing/zero or < min_volume
      - RSI > blow-off
      - chase_risk
      - classification AVOID_REVIEW when fund context is present
      - require_fundamentals and coverage < MIN_FUND_COVERAGE

    Soft status values: pass | fail | unknown (unknown ≠ reject unless required).
    """
    status: dict[str, str] = {}
    reasons: list[str] = []

    vol = volume_ratio(row)
    if vol <= 0:
        status["volume"] = "fail"
        reasons.append("volume missing/zero — not a breakout")
    elif vol < min_volume:
        status["volume"] = "fail"
        reasons.append(f"volume {vol:.2f}× < {min_volume:.1f}× floor")
    else:
        status["volume"] = "pass"

    rsi = _f(row.get("rsi"))
    if rsi > RSI_BLOWOFF:
        status["rsi"] = "fail"
        reasons.append(f"RSI {rsi:.0f} > {RSI_BLOWOFF:.0f} blow-off")
    elif rsi > 0:
        status["rsi"] = "pass"
    else:
        status["rsi"] = "unknown"

    if bool(row.get("chase_risk")):
        status["extension"] = "fail"
        reasons.append("chase/extension risk")
    else:
        status["extension"] = "pass"

    cls = str(row.get("classification") or "")
    cov = _f(row.get("fundamental_coverage"))
    has_fund = row.get("fundamental_score") is not None or bool(cls) or cov > 0
    if cls in AVOID_CLASSES:
        status["fundamentals"] = "fail"
        reasons.append("fundamentals AVOID_REVIEW")
    elif has_fund and cov > 0 and cov < MIN_FUND_COVERAGE:
        status["fundamentals"] = "fail" if require_fundamentals else "unknown"
        if require_fundamentals:
            reasons.append(f"fundamental coverage {cov:.0%} < {MIN_FUND_COVERAGE:.0%}")
    elif cls in QUALITY_CLASSES and cov >= MIN_FUND_COVERAGE:
        status["fundamentals"] = "pass"
    elif has_fund:
        status["fundamentals"] = "pass" if cov >= MIN_FUND_COVERAGE else "unknown"
    else:
        status["fundamentals"] = "unknown"
        if require_fundamentals:
            reasons.append("fundamentals required but missing")

    # Technical structure: soft unless clearly broken
    above50 = row.get("above_sma50")
    above200 = row.get("above_sma200")
    if above50 is False and above200 is False:
        status["trend"] = "fail"
        reasons.append("below SMA50 and SMA200")
    elif above50 is True or above200 is True:
        status["trend"] = "pass"
    else:
        status["trend"] = "unknown"

    ok = not reasons
    return ok, reasons, status


def enrich_optional_context(symbol: str) -> dict[str, Any]:
    """Best-effort order-book + concall context. Never fabricates a pass.

    Returns only what can be verified from live Kite depth or cached evidence.
    Missing sources stay ``unavailable`` so the UI stays honest.
    """
    out: dict[str, Any] = {
        "order_book": {"status": "unavailable", "note": "No live depth for this symbol."},
        "concall": {"status": "unavailable", "note": "No traced earnings-call evidence on file."},
    }
    sym = str(symbol or "").strip().upper()
    if not sym:
        return out

    # Order book — Kite full quote depth when logged in
    try:
        from data.kite_client import KiteClient
        from config import settings
        if settings.kite_access_token:
            kite = KiteClient()
            key = f"{settings.exchange}:{sym}"
            raw = kite.raw.quote([key]) or {}
            quote = raw.get(key) or next(iter(raw.values()), {}) or {}
            depth = quote.get("depth") or {}
            buys = depth.get("buy") or []
            sells = depth.get("sell") or []
            bid_qty = sum(float(level.get("quantity") or 0) for level in buys[:5])
            ask_qty = sum(float(level.get("quantity") or 0) for level in sells[:5])
            if bid_qty > 0 or ask_qty > 0:
                imbalance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1.0)
                if imbalance >= 0.15:
                    book_status = "bid_heavy"
                    note = f"Top-5 bid qty {bid_qty:,.0f} > ask {ask_qty:,.0f}"
                elif imbalance <= -0.15:
                    book_status = "ask_heavy"
                    note = f"Top-5 ask qty {ask_qty:,.0f} > bid {bid_qty:,.0f}"
                else:
                    book_status = "balanced"
                    note = f"Top-5 bid {bid_qty:,.0f} ≈ ask {ask_qty:,.0f}"
                out["order_book"] = {
                    "status": book_status,
                    "note": note,
                    "bid_qty": round(bid_qty),
                    "ask_qty": round(ask_qty),
                    "imbalance": round(imbalance, 3),
                }
    except Exception:
        pass

    # Concall / earnings evidence — structured intake only (no scrape invention)
    try:
        from pathlib import Path
        import json
        root = Path(__file__).resolve().parents[1]
        for path in (
            root / "logs" / "evidence" / f"{sym}.json",
            root / "logs" / "intelligence" / "earnings" / f"{sym}.json",
            root / "logs" / "product" / "earnings_cache" / f"{sym}.json",
        ):
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                continue
            # Accept only explicitly dated / sourced transcript notes
            dated = (
                payload.get("concall_date")
                or payload.get("call_date")
                or payload.get("as_of")
                or payload.get("fetched_at")
            )
            if dated or payload.get("transcript") or payload.get("summary"):
                out["concall"] = {
                    "status": "present",
                    "note": str(payload.get("summary") or payload.get("headline") or "Cached earnings-call note")[:180],
                    "as_of": str(dated or ""),
                    "source": str(payload.get("source") or path.name),
                }
                break
    except Exception:
        pass
    return out


def attach_best_pick_meta(row: Mapping[str, Any], *, with_context: bool = True) -> dict[str, Any]:
    """Copy row + quality gate status (+ optional book/concall context)."""
    out = dict(row)
    ok, reasons, status = gate_breakout_quality(out)
    out["quality_ok"] = ok
    out["quality_gates"] = status
    out["quality_reject_reasons"] = reasons
    if with_context:
        out["breakout_context"] = enrich_optional_context(str(out.get("symbol") or ""))
    return out

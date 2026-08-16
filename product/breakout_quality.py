"""Breakout gates — technical for candidates; fundamentals only for best-among.

Breakout / sniper / Telegram arming use tape + structure only (volume, chase,
hard RSI blow-off). Fundamentals never zero the breakout lane.

A separate best-among pick may require readable fund context and rejects
AVOID_REVIEW. Optional order-book / concall context is attached when available
— never invented, never a sole reason to promote.
"""
from __future__ import annotations

from typing import Any, Mapping

# Technical sniper / Telegram lane — eased so mid-day on-pace prints still arm.
# Best-among can still pass a stricter min_volume explicitly.
MIN_VOLUME_RATIO = 0.7
BEST_MIN_VOLUME_RATIO = 1.0
# Soft ceiling: prefer / best-among. Hard: scanner blow-off (CLAUDE.md).
RSI_BLOWOFF = 70.0
RSI_HARD = 82.0
# Live structure: a breakout that has already given back the high is not "best".
FADE_20D_PCT = 5.0
ROLLOVER_20D_PCT = 2.5
ROLLOVER_RSI = 50.0
# Hero card only: a "best breakout" must be near the 52-week high.
BEST_52W_PCT = 8.0
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


def _pct_field(row: Mapping[str, Any], key: str) -> float | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def pct_below_20d_high(row: Mapping[str, Any]) -> float | None:
    return _pct_field(row, "pct_below_20d_high")


def pct_below_52w_high(row: Mapping[str, Any]) -> float | None:
    return _pct_field(row, "pct_below_52w_high")


def live_breakout_intact(row: Mapping[str, Any], *, for_best: bool = False) -> tuple[bool, list[str]]:
    """Whether the live tape still looks like a breakout — not a faded scan tag.

    Scan grade A/B is sticky. After a multi-day pullback the name can still
    carry BREAKOUT_52W + Grade B while price sits well under the 20-day high
    and RSI has rolled over. That is not a live breakout.

    ``for_best=False`` (sniper lane): fail-open when structure is unknown;
    reject when the live bar has already faded.
    ``for_best=True`` (BEST pick): fail-closed without structure, and chase
    is a hard reject.
    """
    reasons: list[str] = []
    below = pct_below_20d_high(row)
    rsi = _f(row.get("rsi"))

    if below is None:
        if for_best:
            reasons.append("live structure unknown — not ranked as best")
            return False, reasons
        return True, []

    if below > FADE_20D_PCT:
        reasons.append(f"{below:.1f}% below 20-day high — breakout faded")
    elif rsi > 0 and rsi < ROLLOVER_RSI and below > ROLLOVER_20D_PCT:
        reasons.append(f"RSI {rsi:.0f} and {below:.1f}% off 20-day high — rolled over")

    if for_best:
        below52 = pct_below_52w_high(row)
        if below52 is not None and below52 > BEST_52W_PCT:
            reasons.append(f"{below52:.1f}% below 52-week high — not a leading breakout")
        if bool(row.get("chase_risk")):
            reasons.append("chase/extension risk")

    return not reasons, reasons


def gate_breakout_quality(
    row: Mapping[str, Any],
    *,
    for_best: bool = False,
    require_fundamentals: bool = False,
    min_volume: float = MIN_VOLUME_RATIO,
) -> tuple[bool, list[str], dict[str, str]]:
    """Return (ok, reject_reasons, gate_status).

    Technical path (``for_best=False``, default — sniper / Telegram / lane):
      - volume missing/zero or < min_volume (default 0.7×)
      - RSI > RSI_HARD (82)
      - chase_risk is status-only (soft) — still arms alerts
      Fundamentals are status-only — never reject.

    Best-among path (``for_best=True``):
      - same technicals, but RSI > RSI_BLOWOFF (70)
      - chase_risk hard-rejects
      - AVOID_REVIEW rejected
      - optional require_fundamentals coverage floor
      - both SMAs below → reject
    """
    status: dict[str, str] = {}
    reasons: list[str] = []
    want_best = bool(for_best or require_fundamentals)

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
    rsi_cap = RSI_BLOWOFF if want_best else RSI_HARD
    if rsi > rsi_cap:
        status["rsi"] = "fail"
        reasons.append(f"RSI {rsi:.0f} > {rsi_cap:.0f} blow-off")
    elif rsi > 0:
        status["rsi"] = "pass" if rsi <= RSI_BLOWOFF else "elevated"
    else:
        status["rsi"] = "unknown"

    if bool(row.get("chase_risk")):
        status["extension"] = "fail"
        # Soft on the technical lane (Telegram/sniper); hard only for best-among.
        if want_best:
            reasons.append("chase/extension risk")
    else:
        status["extension"] = "pass"

    cls = str(row.get("classification") or "")
    cov = _f(row.get("fundamental_coverage"))
    has_fund = row.get("fundamental_score") is not None or bool(cls) or cov > 0
    if cls in AVOID_CLASSES:
        status["fundamentals"] = "fail"
        if want_best:
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

    above50 = row.get("above_sma50")
    above200 = row.get("above_sma200")
    if above50 is False and above200 is False:
        status["trend"] = "fail"
        if want_best:
            reasons.append("below SMA50 and SMA200")
    elif above50 is True or above200 is True:
        status["trend"] = "pass"
    else:
        status["trend"] = "unknown"

    ok = not reasons
    return ok, reasons, status


def has_usable_fundamentals(row: Mapping[str, Any]) -> bool:
    """True when long-term fund context is good enough for best-among."""
    cls = str(row.get("classification") or "")
    if cls in AVOID_CLASSES:
        return False
    cov = _f(row.get("fundamental_coverage"))
    if cov < MIN_FUND_COVERAGE:
        return False
    if cls in QUALITY_CLASSES:
        return True
    return row.get("fundamental_score") is not None


def enrich_optional_context(symbol: str) -> dict[str, Any]:
    """Best-effort order-book + concall context. Never fabricates a pass."""
    out: dict[str, Any] = {
        "order_book": {"status": "unavailable", "note": "No live depth for this symbol."},
        "concall": {"status": "unavailable", "note": "No traced earnings-call evidence on file."},
    }
    sym = str(symbol or "").strip().upper()
    if not sym:
        return out

    try:
        from data.kite_client import KiteClient, _fresh_env
        if _fresh_env("KITE_ACCESS_TOKEN"):
            from config import settings
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


def attach_best_pick_meta(
    row: Mapping[str, Any],
    *,
    with_context: bool = True,
    for_best: bool = True,
) -> dict[str, Any]:
    """Copy row + quality gate status (+ optional book/concall context)."""
    out = dict(row)
    ok, reasons, status = gate_breakout_quality(out, for_best=for_best)
    out["quality_ok"] = ok
    out["quality_gates"] = status
    out["quality_reject_reasons"] = reasons
    if with_context:
        out["breakout_context"] = enrich_optional_context(str(out.get("symbol") or ""))
    return out

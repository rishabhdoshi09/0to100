"""Minervini SEPA / Trend Template — published rules, scored from official OHLCV.

Seven Stage-2 checks, 100 points. Missing history awards 0 and is not a pass.
This is a research overlay (Reco-style monitor), not a measured QuantTerm edge
and never a buy instruction.

Strict 8/8 qualification (AND-gate + cross-sectional RS + structural VCP +
buy-zone) lives in ``research.sepa`` (SEPA-001). This module stays the Ideas
scorer; do not treat score >= 40 as SEPA eligibility.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

SEPA_TOTAL = 7
SEPA_MAX_SCORE = 100
NEAR_52W_PCT = 25.0
OFF_52W_LOW_PCT = 25.0
SMA200_LOOKBACK = 21  # ~1 month of sessions
_RANK_CACHE: dict[str, tuple[float, list[tuple[dict[str, Any], dict[str, Any]]], str]] = {}
_RANK_TTL = 180.0

CRITERIA: tuple[dict[str, Any], ...] = (
    {
        "id": "near_52w_high",
        "title": "52-Week High Proximity",
        "points": 15,
        "rule": "Price within 25% of the 52-week high.",
    },
    {
        "id": "off_52w_low",
        "title": "52-Week Low Distance",
        "points": 10,
        "rule": "Price at least 25% above the 52-week low.",
    },
    {
        "id": "above_150_200",
        "title": "Price vs 150 & 200 DMA",
        "points": 20,
        "rule": "Close above both the 150-day and 200-day simple averages.",
    },
    {
        "id": "sma150_gt_200",
        "title": "150 DMA vs 200 DMA",
        "points": 10,
        "rule": "150-day average is above the 200-day average.",
    },
    {
        "id": "sma200_rising",
        "title": "200 DMA trending up",
        "points": 10,
        "rule": "200-day average is higher than it was ~1 month ago.",
    },
    {
        "id": "sma50_leads",
        "title": "50 DMA above 150 & 200",
        "points": 20,
        "rule": "50-day average is above both longer averages.",
    },
    {
        "id": "above_sma50",
        "title": "Price vs 50 DMA",
        "points": 15,
        "rule": "Close above the 50-day simple average.",
    },
)

_POINTS = {row["id"]: int(row["points"]) for row in CRITERIA}


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out:  # NaN
            return None
        return out
    except (TypeError, ValueError):
        return None


def _sma(close, window: int) -> float | None:
    if close is None or len(close) < window:
        return None
    try:
        val = float(close.iloc[-window:].mean())
    except Exception:
        return None
    if val != val:
        return None
    return val


def _criterion(
    spec_id: str,
    *,
    passed: bool | None,
    detail: str,
    note: str,
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = next(row for row in CRITERIA if row["id"] == spec_id)
    points = int(spec["points"])
    awarded = points if passed is True else 0
    return {
        "id": spec_id,
        "title": spec["title"],
        "rule": spec["rule"],
        "points": points,
        "awarded": awarded,
        "passed": passed,
        "detail": detail,
        "note": note,
        "values": dict(values or {}),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    criteria = [
        _criterion(
            spec["id"],
            passed=None,
            detail="Not enough official history.",
            note=reason,
        )
        for spec in CRITERIA
    ]
    return {
        "available": False,
        "score": 0,
        "max_score": SEPA_MAX_SCORE,
        "passed": 0,
        "total": SEPA_TOTAL,
        "unknown": SEPA_TOTAL,
        "verdict": "INCOMPLETE",
        "headline": "INCOMPLETE — NOT ENOUGH HISTORY",
        "advice": reason,
        "method": "minervini_trend_template_7",
        "disclaimer": (
            "SEPA here is Mark Minervini's published Stage-2 trend template "
            "scored on NSE bhavcopy. It is not a QuantTerm backtest edge and not a buy order."
        ),
        "criteria": criteria,
        "quote": None,
    }


def _verdict(score: int, passed: int, unknown: int) -> tuple[str, str, str]:
    if unknown >= 4:
        return (
            "INCOMPLETE",
            "INCOMPLETE — NOT ENOUGH HISTORY",
            "Need ~200 sessions of official OHLCV before the 7 Stage-2 checks are readable.",
        )
    if score >= 80 and passed >= 6:
        return (
            "STRONG",
            "STRONG — MEETS SEPA",
            "Stage-2 template is intact: price is behaving like a leadership name near highs. "
            "This is a research qualify, not a buy ticket — still check stop, volume and chase risk.",
        )
    if score >= 60 and passed >= 5:
        return (
            "CONSTRUCTIVE",
            "CONSTRUCTIVE — SETUP FORMING",
            "Most of the trend template is in place. Wait for the failed checks to flip, "
            "or treat this as a watch — not an all-in swing.",
        )
    if score >= 40:
        return (
            "MIXED",
            "MIXED — WAIT FOR STRUCTURE",
            "Some Stage-2 pieces are present, but this is not a clean Minervini setup yet. "
            "Better candidates sit closer to 52-week highs with a rising 200-day average.",
        )
    return (
        "WEAK",
        "WEAK — NOT IDEAL FOR SWING",
        "This stock currently does not meet Minervini's SEPA criteria. "
        "Either wait for it to set up properly or look for better candidates near "
        "52-week highs with strong relative strength.",
    )


def _quote_stats(frame: Any) -> dict[str, Any] | None:
    try:
        data = frame.sort_index()
        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) >= 2 else None
        close = float(last["close"])
        prev_close = float(prev["close"]) if prev is not None else None
        change_pct = None
        if prev_close and prev_close > 0:
            change_pct = round((close / prev_close - 1.0) * 100.0, 2)
        return {
            "open": round(float(last["open"]), 2) if "open" in data.columns else None,
            "high": round(float(last["high"]), 2) if "high" in data.columns else None,
            "low": round(float(last["low"]), 2) if "low" in data.columns else None,
            "close": round(close, 2),
            "prev_close": round(prev_close, 2) if prev_close else None,
            "change_pct": change_pct,
            "as_of": str(getattr(data.index[-1], "date", lambda: data.index[-1])()),
        }
    except Exception:
        return None


def score_sepa(frame: Any, *, bench_frame: Any = None) -> dict[str, Any]:
    """Score one OHLCV frame. Never fabricates moving averages or 52-week levels."""
    if frame is None:
        return _unavailable("Official price history is unavailable.")
    try:
        if len(frame) == 0:
            return _unavailable("Official price history is unavailable.")
        data = frame.sort_index().copy()
        close = data["close"].astype(float).dropna()
    except Exception:
        return _unavailable("Official price history is unreadable.")
    if len(close) < 50:
        return _unavailable(
            f"Only {len(close)} sessions on file — SEPA needs at least 50 days, 200 for the full template."
        )

    price = float(close.iloc[-1])
    high_col = data["high"].astype(float) if "high" in data.columns else close
    low_col = data["low"].astype(float) if "low" in data.columns else close
    win = min(252, len(data))
    high_52w = float(high_col.tail(win).max())
    low_52w = float(low_col.tail(win).min())
    sma50 = _sma(close, 50)
    sma150 = _sma(close, 150)
    sma200 = _sma(close, 200)
    sma200_prev = None
    if len(close) >= 200 + SMA200_LOOKBACK:
        sma200_prev = _sma(close.iloc[: -SMA200_LOOKBACK], 200)

    below_high_pct = None
    if high_52w > 0:
        below_high_pct = round((1.0 - price / high_52w) * 100.0, 1)
    above_low_pct = None
    if low_52w > 0:
        above_low_pct = round((price / low_52w - 1.0) * 100.0, 1)

    # 1. 52-week high proximity
    if below_high_pct is None:
        c1 = _criterion(
            "near_52w_high", passed=None,
            detail="52-week high is unavailable.",
            note="Cannot score proximity without a 52-week high.",
        )
    else:
        ok = below_high_pct <= NEAR_52W_PCT
        c1 = _criterion(
            "near_52w_high",
            passed=ok,
            detail=f"{below_high_pct:.1f}% below 52WH (₹{high_52w:,.2f})",
            note=(
                "Acceptable — still within 25% of highs."
                if ok
                else "Laggard zone — more than 25% below the 52-week high."
            ),
            values={"below_high_pct": below_high_pct, "high_52w": round(high_52w, 2)},
        )

    # 2. 52-week low distance
    if above_low_pct is None:
        c2 = _criterion(
            "off_52w_low", passed=None,
            detail="52-week low is unavailable.",
            note="Cannot score distance from lows.",
        )
    else:
        ok = above_low_pct >= OFF_52W_LOW_PCT
        c2 = _criterion(
            "off_52w_low",
            passed=ok,
            detail=f"{above_low_pct:.1f}% above 52WL (₹{low_52w:,.2f})",
            note=(
                "Enough distance from the low — not a falling knife."
                if ok
                else "Too close to the 52-week low for a Stage-2 swing."
            ),
            values={"above_low_pct": above_low_pct, "low_52w": round(low_52w, 2)},
        )

    # 3. Price vs 150 & 200
    if sma150 is None or sma200 is None:
        c3 = _criterion(
            "above_150_200", passed=None,
            detail="Need 200 sessions for 150/200 DMA.",
            note="History is too short for the long averages.",
            values={"sma150": sma150, "sma200": sma200, "price": round(price, 2)},
        )
    else:
        ok = price > sma150 and price > sma200
        c3 = _criterion(
            "above_150_200",
            passed=ok,
            detail=(
                f"Close ₹{price:,.2f} vs 150-DMA ₹{sma150:,.2f} / 200-DMA ₹{sma200:,.2f}"
            ),
            note="Price is above both long averages." if ok else "Price is below a long-term average.",
            values={"sma150": round(sma150, 2), "sma200": round(sma200, 2), "price": round(price, 2)},
        )

    # 4. 150 > 200
    if sma150 is None or sma200 is None:
        c4 = _criterion(
            "sma150_gt_200", passed=None,
            detail="Need 200 sessions to compare 150 vs 200 DMA.",
            note="History is too short.",
        )
    else:
        ok = sma150 > sma200
        c4 = _criterion(
            "sma150_gt_200",
            passed=ok,
            detail=f"150-DMA ₹{sma150:,.2f} vs 200-DMA ₹{sma200:,.2f}",
            note="Intermediate average leads the long average." if ok else "150-DMA is still below 200-DMA.",
            values={"sma150": round(sma150, 2), "sma200": round(sma200, 2)},
        )

    # 5. 200 rising
    if sma200 is None or sma200_prev is None:
        c5 = _criterion(
            "sma200_rising", passed=None,
            detail="Need ~221 sessions to see whether the 200-DMA is rising.",
            note="Not enough history for the one-month slope.",
        )
    else:
        ok = sma200 > sma200_prev
        delta = sma200 - sma200_prev
        c5 = _criterion(
            "sma200_rising",
            passed=ok,
            detail=f"200-DMA ₹{sma200:,.2f} vs ₹{sma200_prev:,.2f} a month ago ({delta:+.2f})",
            note="Long average is rising." if ok else "200-DMA is flat or falling — Stage 1/4, not Stage 2.",
            values={"sma200": round(sma200, 2), "sma200_prev": round(sma200_prev, 2)},
        )

    # 6. 50 above 150 & 200
    if sma50 is None or sma150 is None or sma200 is None:
        c6 = _criterion(
            "sma50_leads", passed=None,
            detail="Need 200 sessions for the full average stack.",
            note="History is too short.",
        )
    else:
        ok = sma50 > sma150 and sma50 > sma200
        c6 = _criterion(
            "sma50_leads",
            passed=ok,
            detail=f"50-DMA ₹{sma50:,.2f} vs 150 ₹{sma150:,.2f} / 200 ₹{sma200:,.2f}",
            note="Short average leads the stack." if ok else "50-DMA is not leading the longer averages.",
            values={"sma50": round(sma50, 2), "sma150": round(sma150, 2), "sma200": round(sma200, 2)},
        )

    # 7. Price vs 50
    if sma50 is None:
        c7 = _criterion(
            "above_sma50", passed=None,
            detail="Need 50 sessions for the 50-DMA.",
            note="History is too short.",
        )
    else:
        ok = price > sma50
        c7 = _criterion(
            "above_sma50",
            passed=ok,
            detail=f"Close ₹{price:,.2f} vs 50-DMA ₹{sma50:,.2f}",
            note="Price holds above the 50-day average." if ok else "Price is below the 50-day average.",
            values={"sma50": round(sma50, 2), "price": round(price, 2)},
        )

    criteria = [c1, c2, c3, c4, c5, c6, c7]
    score = int(sum(int(c["awarded"]) for c in criteria))
    passed_n = sum(1 for c in criteria if c["passed"] is True)
    unknown_n = sum(1 for c in criteria if c["passed"] is None)
    verdict, headline, advice = _verdict(score, passed_n, unknown_n)
    quote = _quote_stats(data)
    session = _session_label()
    result = {
        "available": True,
        "score": score,
        "max_score": SEPA_MAX_SCORE,
        "passed": passed_n,
        "total": SEPA_TOTAL,
        "unknown": unknown_n,
        "verdict": verdict,
        "headline": headline,
        "advice": advice,
        "method": "minervini_trend_template_7",
        "disclaimer": (
            "SEPA here is Mark Minervini's published Stage-2 trend template "
            "scored on NSE bhavcopy. It is not a QuantTerm backtest edge and not a buy order."
        ),
        "criteria": criteria,
        "quote": quote,
        "session": session,
        "levels": {
            "price": round(price, 2),
            "sma50": round(sma50, 2) if sma50 is not None else None,
            "sma150": round(sma150, 2) if sma150 is not None else None,
            "sma200": round(sma200, 2) if sma200 is not None else None,
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
        },
    }
    try:
        from product.monitor_context import attach_context
        attach_context(result, data, bench_frame)
    except Exception:
        result.setdefault(
            "stage",
            {"id": "unknown", "label": "STAGE ?", "note": "Stage unavailable."},
        )
        result.setdefault(
            "rs",
            {
                "available": False,
                "label": "UNKNOWN",
                "note": "Need official stock and Nifty history for relative strength.",
            },
        )
    return result


def _session_label() -> dict[str, Any]:
    try:
        from core.market_clock import now_ist
        from core.market_session import in_market_open
        ts = now_ist()
        open_now = bool(in_market_open(ts))
        return {
            "label": "MARKET OPEN" if open_now else "MARKET CLOSED",
            "open": open_now,
            "clock": ts.strftime("%H:%M:%S IST"),
        }
    except Exception:
        return {"label": "SESSION UNKNOWN", "open": None, "clock": ""}


def sepa_card_fields(sepa: Mapping[str, Any]) -> dict[str, Any]:
    """Compact fields for Ideas cards — never a fabricated score."""
    if not sepa or not sepa.get("available"):
        return {}
    stage = dict(sepa.get("stage") or {})
    rs = dict(sepa.get("rs") or {})
    excess = rs.get("excess_pp")
    try:
        excess_pp = float(excess) if excess is not None else None
    except (TypeError, ValueError):
        excess_pp = None
    return {
        "sepa_score": int(sepa.get("score") or 0),
        "sepa_max": int(sepa.get("max_score") or SEPA_MAX_SCORE),
        "sepa_passed": int(sepa.get("passed") or 0),
        "sepa_total": int(sepa.get("total") or SEPA_TOTAL),
        "sepa_verdict": str(sepa.get("verdict") or ""),
        "sepa_headline": str(sepa.get("headline") or ""),
        "sepa_advice": str(sepa.get("advice") or ""),
        "setup_label": str(sepa.get("headline") or ""),
        "stage_id": stage.get("id"),
        "stage_label": stage.get("label"),
        "stage_note": stage.get("note"),
        "rs_available": bool(rs.get("available")),
        "rs_label": str(rs.get("label") or ""),
        "rs_excess_pp": excess_pp,
        "rs_stock_pct": rs.get("stock_pct"),
        "rs_benchmark_pct": rs.get("benchmark_pct"),
        "rs_note": str(rs.get("note") or ""),
        "volume_label": str((sepa.get("volume") or {}).get("label") or ""),
        "volume_rvol": (sepa.get("volume") or {}).get("rvol"),
        "breakout_score": (sepa.get("breakout") or {}).get("score"),
        "breakout_label": str((sepa.get("breakout") or {}).get("label") or ""),
    }


def _candidate_rank(row: Mapping[str, Any]) -> tuple:
    chase = 1 if bool(row.get("chase_risk")) else 0
    rsi = _f(row.get("rsi")) or 0.0
    blowoff = 1 if rsi > 82 else 0
    verdict = str(row.get("verdict") or "").upper()
    v = 0 if verdict in {"BUY", "STRONG BUY"} else 1
    score = float(row.get("score") or 0.0)
    return (chase, blowoff, v, -score, str(row.get("symbol") or ""))


def select_sepa_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Scan names worth an OHLCV SEPA pass — not the whole universe."""
    pool = [dict(r) for r in rows if str(r.get("symbol") or "").strip()]
    pool.sort(key=_candidate_rank)
    return pool[: max(1, int(limit))]


def rank_best_setups(
    rows: Sequence[Mapping[str, Any]],
    *,
    load_frame: Callable[[str], Any] | None = None,
    limit: int = 40,
    score_cap: int = 80,
    min_score: int = 40,
    cache_key: str = "",
    max_seconds: float | None = 45.0,
) -> tuple[list[dict[str, Any]], str]:
    """Score a scan shortlist on SEPA. Empty list if history cannot be read.

    Names below ``min_score`` stay out of the Best Setups lane so a 20/100
    downtrend is not dressed up as a setup. If nobody clears the floor, the
    note explains that — we do not invent winners.

    ``max_seconds`` is a hard budget on OHLCV scoring (default 45s). Ideas
    must not sit on a request thread while 80 frames load. Incomplete
    rankings are not cached.
    """
    import time
    if cache_key:
        hit = _RANK_CACHE.get(cache_key)
        if hit and (time.time() - hit[0]) < _RANK_TTL:
            return hit[1], hit[2]
    loader = load_frame or _default_frame
    try:
        from product.monitor_context import nifty_frame, rs_rank
    except Exception:
        def nifty_frame():
            return None

        def rs_rank(_sepa):
            return 0
    try:
        bench = nifty_frame()
    except Exception:
        bench = None
    scored: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    truncated = False
    attempted = 0
    deadline = None
    if max_seconds is not None and float(max_seconds) > 0:
        deadline = time.monotonic() + float(max_seconds)
    for row in select_sepa_candidates(rows, limit=score_cap):
        if deadline is not None and time.monotonic() >= deadline:
            truncated = True
            break
        attempted += 1
        symbol = str(row.get("symbol") or "").upper()
        try:
            frame = loader(symbol)
        except Exception:
            frame = None
        sepa = score_sepa(frame, bench_frame=bench)
        if not sepa.get("available"):
            continue
        score = int(sepa.get("score") or 0)
        passed = int(sepa.get("passed") or 0)
        if score < min_score:
            continue
        scored.append((score, passed, sepa, row))
    def _below_high(sepa: Mapping[str, Any]) -> float:
        for item in sepa.get("criteria") or []:
            if item.get("id") == "near_52w_high":
                try:
                    return float((item.get("values") or {}).get("below_high_pct") or 99.0)
                except (TypeError, ValueError):
                    return 99.0
        return 99.0
    def _fund_rank(row: Mapping[str, Any]) -> int:
        try:
            from product.top_stocks import fund_rank
            return int(fund_rank(row))
        except Exception:
            return 0
    scored.sort(key=lambda item: (
        -item[0],
        -item[1],
        -rs_rank(item[2]),
        -_fund_rank(item[3]),
        _below_high(item[2]),
        -float(item[3].get("score") or 0.0),
        str(item[3].get("symbol") or ""),
    ))
    top = scored[:limit]
    budget = f"{float(max_seconds):.0f}s" if max_seconds and float(max_seconds) > 0 else ""
    if top:
        note = (
            "Best Setups = last-scan names ranked on Minervini's 7-rule Stage-2 template "
            f"(need ≥{min_score}/100). Stage and RS vs Nifty 50 are official-history "
            "context. Score is research, not a buy."
        )
        if truncated:
            note = (
                f"{note} Ranking stopped at the {budget} history budget after "
                f"{attempted} name(s) — the list may still grow."
            )
        out = [(sepa, row) for _, _, sepa, row in top]
        if cache_key and not truncated:
            _RANK_CACHE[cache_key] = (time.time(), out, note)
        return out, note
    n = attempted if truncated else len(select_sepa_candidates(rows, limit=score_cap))
    if truncated:
        note = (
            f"History budget ({budget}) ended after {n} name(s); none had cleared "
            f"{min_score}/100 yet. Ranking continues in the background."
        )
        return [], note
    note = (
        f"Scored {n} scan name(s) on the SEPA template; none cleared {min_score}/100. "
        "Open a stock to see which of the 7 rules failed."
    )
    if cache_key:
        _RANK_CACHE[cache_key] = (time.time(), [], note)
    return [], note


def public_best_setups(
    scan_payload: Mapping[str, Any] | None,
    *,
    limit: int = 8,
    score_cap: int = 24,
    max_seconds: float = 8.0,
    min_score: int = 40,
    load_frame: Callable[[str], Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """RecoWealth Today cards from the saved scan. Research overlay only."""
    records = list((scan_payload or {}).get("records") or [])
    if not records:
        return [], "No saved scan yet — SEPA ranking needs the last whole-market scan."
    cache_key = f"{scan_payload.get('scanned_at')}:{limit}:{score_cap}:{min_score}"
    ranked, note = rank_best_setups(
        records,
        load_frame=load_frame,
        limit=limit,
        score_cap=score_cap,
        min_score=min_score,
        max_seconds=max_seconds,
        cache_key=cache_key,
    )
    cards: list[dict[str, Any]] = []
    for sepa, row in ranked:
        card = dict(row)
        card.update(sepa_card_fields(sepa))
        quote = dict(sepa.get("quote") or {})
        if quote.get("close") and not card.get("price"):
            card["price"] = quote.get("close")
        cards.append(card)
    return cards, note


def _default_frame(symbol: str) -> Any:
    from data.bhavcopy_runtime import get_ohlcv
    return get_ohlcv(symbol)

"""Desk posture for the path that actually runs.

``core.brain.assess`` still probes the legacy auto-scan store and the
trades journal. The canonical stack (market ops scan, autonomy paper
cycle, product API) never starts those. This module joins the same pure
posture function to the paper book, the saved scan context, and curated
news — without a second signal family and without loosening the 1% / 10%
/ 5% rails.

The dashboard overlay is read-only and does not walk the bhavcopy cache.
Breadth, edge and correlation are refreshed on the scan / paper-cycle
jobs, then read back here if the snapshot is still fresh.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

STALE_HOURS = 36.0

_ACTIONS = {
    "STAND_ASIDE": "No new paper entries.",
    "DEFENSIVE": "Only high-conviction setups. Size stays at the 1% rail.",
    "NORMAL": "Normal paper entries. The 1% / 10% / 5% rails are unchanged.",
    "AGGRESSIVE": "Lean-in is allowed only inside the existing 1% and 5% rails.",
}


def _utcnow(now: datetime | None = None) -> datetime:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        return clock.replace(tzinfo=timezone.utc)
    return clock


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def snapshot_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[1] / "logs" / "product" / "desk_brain.json"


def read_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    target = snapshot_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_snapshot(payload: Mapping[str, Any], path: str | Path | None = None) -> None:
    target = snapshot_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)


def fresh_section(key: str, *, path: str | Path | None = None,
                  now: datetime | None = None) -> dict[str, Any]:
    """One snapshot section, or {} when the file is missing or older than 36h."""
    snap = read_snapshot(path)
    if not snap or _age_hours(snap.get("generated_at"), now) is None:
        return {}
    if _is_stale(snap, now):
        return {}
    section = snap.get(key)
    return dict(section) if isinstance(section, Mapping) else {}


def _age_hours(stamp: Any, now: datetime | None = None) -> float | None:
    parsed = _parse_dt(stamp)
    if parsed is None:
        return None
    return (_utcnow(now) - parsed).total_seconds() / 3600.0


def _is_stale(snap: Mapping[str, Any], now: datetime | None = None) -> bool:
    age = _age_hours(snap.get("generated_at"), now)
    return age is None or age > STALE_HOURS


def rows_from_positions(positions) -> list[dict]:
    rows: list[dict] = []
    for pos in positions or []:
        if isinstance(pos, Mapping):
            symbol = pos.get("symbol")
            qty = pos.get("qty") if pos.get("qty") is not None else pos.get("quantity")
            entry = pos.get("entry_price") if pos.get("entry_price") is not None else pos.get("entry")
            stop = pos.get("stop_price") if pos.get("stop_price") is not None else pos.get("stop")
            sector = pos.get("sector") or ""
        else:
            symbol = getattr(pos, "symbol", "")
            qty = getattr(pos, "qty", 0)
            entry = getattr(pos, "entry_price", 0)
            stop = getattr(pos, "stop_price", 0)
            sector = getattr(pos, "sector", "") or ""
        text = str(symbol or "").strip().upper()
        if not text:
            continue
        rows.append({
            "symbol": text,
            "qty": qty,
            "entry": entry,
            "stop": stop,
            "sector": str(sector or ""),
        })
    return rows


def product_open_rows() -> list[dict]:
    """Open rows on the modern paper book. Empty when the book cannot be read."""
    try:
        from product.paper_status import read_paper_status
        return rows_from_positions(read_paper_status().open_positions)
    except Exception:
        return []


def rows_from_book(book) -> list[dict]:
    if book is None:
        return product_open_rows()
    opens = getattr(book, "open", None)
    if isinstance(opens, Mapping):
        return rows_from_positions(opens.values())
    if isinstance(opens, (list, tuple)):
        return rows_from_positions(opens)
    return product_open_rows()


def _account_limits(book=None) -> tuple[float, int]:
    from risk.portfolio_risk import _capital, _max_positions
    capital = float(_capital())
    max_n = int(_max_positions())
    book_max = getattr(book, "max_positions", None) if book is not None else None
    if book_max is None and book is None:
        try:
            from product.paper_status import read_paper_status
            book_max = read_paper_status().max_positions
        except Exception:
            book_max = None
    try:
        if book_max:
            max_n = min(max_n, int(book_max))
    except (TypeError, ValueError):
        pass
    return capital, max_n


def compose_desk_read(
    *,
    book_rows: list[dict] | None = None,
    capital: float | None = None,
    max_positions: int | None = None,
    regime: str = "",
    market_risk_mode: str = "",
    edge: Mapping[str, Any] | None = None,
    breadth: Mapping[str, Any] | None = None,
    macro: Mapping[str, Any] | None = None,
    correlation: Mapping[str, Any] | None = None,
    snapshot_generated_at: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure posture. Missing context degrades; it does not invent an edge."""
    from core.brain import build_directives, decide_posture
    from risk.portfolio_risk import assess_open_rows

    book = assess_open_rows(list(book_rows or []), capital=capital, max_positions=max_positions)
    edge_map = dict(edge or {})
    breadth_map = dict(breadth or {})
    macro_map = dict(macro or {})
    corr_map = dict(correlation or {})
    measured_edge = "expectancy_r" in edge_map or "closed" in edge_map
    expectancy = float(edge_map.get("expectancy_r") or 0.0) if measured_edge else 0.0
    edge_trend = str(edge_map.get("edge_trend") or "") if measured_edge else ""
    closed = int(edge_map.get("closed") or 0) if measured_edge else 0
    breadth_verdict = str(breadth_map.get("verdict") or "")
    macro_risk_off = bool(macro_map.get("risk_off"))
    posture, reason = decide_posture(
        str(regime or ""),
        edge_trend or "stable",
        expectancy,
        str(book.get("verdict") or "OK"),
        closed,
        breadth=breadth_verdict,
        macro_risk_off=macro_risk_off,
    )
    risk_mode = str(market_risk_mode or "").upper()
    # The paper gate already hard-blocks RISK_OFF. The label has to match
    # that block. Book DANGER stays the more specific stand-aside reason.
    if risk_mode == "RISK_OFF" and posture != "STAND_ASIDE":
        posture = "STAND_ASIDE"
        reason = "Cached regime is RISK_OFF. New paper longs stand aside."

    corr_measured = bool(corr_map.get("measured"))
    corr_positions = int(corr_map.get("n_positions") or 0) if corr_measured else 0
    corr_bets = int(corr_map.get("n_bets") or 0) if corr_measured else 0
    biggest = list(corr_map.get("biggest") or []) if corr_measured else []
    vitals = {
        "book_verdict": book.get("verdict") or "OK",
        "open_risk_pct": float(book.get("open_risk_pct") or 0.0),
        "regime": str(regime or "UNKNOWN"),
        "edge_trend": edge_trend or "stable",
        "expectancy_r": expectancy,
        "recent_r": float(edge_map.get("recent_avg_r") or 0.0) if measured_edge else 0.0,
        "closed": closed,
        "breadth": breadth_verdict,
        "breadth_line": str(breadth_map.get("line") or ""),
        "macro_mood": str(macro_map.get("mood") or ""),
        "macro_note": str(macro_map.get("note") or ""),
        "macro_themes": list(macro_map.get("themes") or []),
        "corr_positions": corr_positions,
        "corr_bets": corr_bets,
        "corr_biggest": biggest,
        "posture": posture,
        "stale_min": None,
        "n_high_conviction": 0,
        "autopilot_armed": True,
    }
    try:
        directives = build_directives(vitals)
    except Exception:
        directives = []
    generated_at = snapshot_generated_at or _utcnow(now).isoformat()
    return {
        "generated_at": generated_at,
        "posture": posture,
        "posture_reason": reason,
        "action": _ACTIONS.get(posture, _ACTIONS["NORMAL"]),
        "book_verdict": book.get("verdict") or "OK",
        "open_risk_pct": book.get("open_risk_pct"),
        "open_risk": book.get("open_risk"),
        "n_positions": book.get("n_positions"),
        "regime": str(regime or ""),
        "regime_measured": bool(str(regime or "").strip() or risk_mode),
        "market_risk_mode": risk_mode,
        "breadth_verdict": breadth_verdict,
        "breadth": {
            "verdict": breadth_verdict,
            "line": str(breadth_map.get("line") or ""),
            "pct_above_50": breadth_map.get("pct_above_50"),
            "n": breadth_map.get("n"),
        },
        "edge": {
            "expectancy_r": expectancy if measured_edge else None,
            "edge_trend": edge_trend,
            "closed": closed if measured_edge else None,
            "recent_avg_r": edge_map.get("recent_avg_r") if measured_edge else None,
            "measured": measured_edge,
        },
        "macro_mood": str(macro_map.get("mood") or ""),
        "macro": {
            "mood": str(macro_map.get("mood") or ""),
            "risk_off": macro_risk_off,
            "note": str(macro_map.get("note") or ""),
            "heat": macro_map.get("heat"),
            "measured": bool(macro_map),
        },
        "correlation_measured": corr_measured,
        "correlation_positions": int(corr_map.get("n_positions") or 0),
        "correlation_bets": int(corr_map.get("n_bets") or 0) if corr_measured else None,
        "correlation": {
            "n_positions": int(corr_map.get("n_positions") or 0),
            "n_bets": int(corr_map.get("n_bets") or 0) if corr_measured else None,
            "biggest": biggest,
            "measured": corr_measured,
            "clusters": list(corr_map.get("clusters") or []) if corr_measured else [],
        },
        "directives": directives,
        "snapshot_stale": False,
        "live_locked": True,
        "size_rails": "1% per trade, 10% per name, 5% open risk — unchanged",
    }


def execution_controls(read: Mapping[str, Any] | None) -> dict[str, str]:
    """What the paper cycle is allowed to pass through. Empty read = no new block."""
    payload = dict(read or {})
    posture = str(payload.get("posture") or "").upper()
    if posture not in {"STAND_ASIDE", "DEFENSIVE", "NORMAL", "AGGRESSIVE"}:
        posture = ""
    risk_mode = str(payload.get("market_risk_mode") or "").upper()
    return {
        "regime": "RISK_OFF" if risk_mode == "RISK_OFF" else "RISK_ON",
        "posture": posture,
        "posture_reason": str(payload.get("posture_reason") or ""),
    }


def _peek_regime() -> tuple[str, str]:
    try:
        from core.regime_engine import peek_cached_regime
        cached = peek_cached_regime()
    except Exception:
        return "", ""
    if cached is None:
        return "", ""
    regime = str(getattr(cached, "market_regime", "") or "")
    risk_mode = str(getattr(cached, "risk_mode", "") or "")
    return regime, risk_mode


def _safe_edge(capital: float) -> dict:
    try:
        from core.brain import _probe_edge
        data = _probe_edge(capital) or {}
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


_breadth_memo: dict[str, Any] = {"ts": 0.0, "data": {}}


def _safe_breadth() -> dict:
    import time
    if _breadth_memo["ts"] and time.time() - float(_breadth_memo["ts"]) < 900:
        data = _breadth_memo.get("data") or {}
        return dict(data) if isinstance(data, Mapping) else {}
    try:
        from scan.breadth import breadth_from_cache
        data = breadth_from_cache() or {}
    except Exception:
        data = {}
    _breadth_memo["ts"] = time.time()
    _breadth_memo["data"] = data
    return dict(data) if isinstance(data, Mapping) else {}


def _safe_macro() -> dict:
    try:
        from core.macro_pulse import macro_pulse
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(
            Path(__file__).resolve().parents[1] / "logs" / "news_curator.sqlite3"
        )
        try:
            articles = [item.as_dict() for item in store.recent(hours=12, limit=80)]
        finally:
            store.close()
        if not articles:
            return {}
        return dict(macro_pulse(articles) or {})
    except Exception:
        return {}


def _safe_correlation(symbols: list[str]) -> dict:
    if len(symbols) < 2:
        return {"n_positions": len(symbols), "n_bets": len(symbols),
                "clusters": [[s] for s in symbols], "biggest": None, "measured": False}
    try:
        from risk.correlation import report_for_symbols
        return dict(report_for_symbols(symbols) or {})
    except Exception:
        return {"n_positions": len(symbols), "measured": False}


def load_for_execution(
    book=None,
    *,
    breadth_fn: Callable[[], Mapping[str, Any]] | None = None,
    edge_fn: Callable[[float], Mapping[str, Any]] | None = None,
    macro_fn: Callable[[], Mapping[str, Any]] | None = None,
    corr_fn: Callable[[list[str]], Mapping[str, Any]] | None = None,
    regime_fn: Callable[[], tuple[str, str]] | None = None,
    now: datetime | None = None,
    persist: bool = True,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Job-path read. Safe to call from scan and paper cycle; not from Home."""
    rows = rows_from_book(book)
    capital, max_n = _account_limits(book)
    regime, risk_mode = (regime_fn or _peek_regime)()
    edge = dict((edge_fn or _safe_edge)(capital) or {})
    breadth = dict((breadth_fn or _safe_breadth)() or {})
    macro = dict((macro_fn or _safe_macro)() or {})
    symbols = [row["symbol"] for row in rows]
    correlation = dict((corr_fn or _safe_correlation)(symbols) or {})
    read = compose_desk_read(
        book_rows=rows,
        capital=capital,
        max_positions=max_n,
        regime=regime,
        market_risk_mode=risk_mode,
        edge=edge,
        breadth=breadth,
        macro=macro,
        correlation=correlation,
        now=now,
    )
    if persist:
        try:
            write_snapshot(read, path)
        except Exception:
            pass
    return read


def overlay_market(
    market: Mapping[str, Any] | None,
    *,
    paper: Mapping[str, Any] | None = None,
    articles: list | None = None,
    now: datetime | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Attach the current posture to a dashboard market payload.

    Uses the paper rows and news articles the request already loaded, plus
    a fresh on-disk snapshot for breadth / edge / correlation. Does not
    fetch prices or unpickle the bhav cache.
    """
    base = dict(market or {})
    try:
        from risk.portfolio_risk import _capital, _max_positions
        snap = read_snapshot(path)
        stale = _is_stale(snap, now) if snap else True
        paper_map = dict(paper or {})
        rows = rows_from_positions(paper_map.get("open_positions") or [])
        try:
            from risk.portfolio_risk import legacy_open_rows
            rows = list(legacy_open_rows()) + rows
        except Exception:
            pass
        capital = float(_capital())
        max_n = int(_max_positions())
        try:
            if paper_map.get("max_positions"):
                max_n = min(max_n, int(paper_map.get("max_positions") or max_n))
        except (TypeError, ValueError):
            pass
        details = dict(base.get("technical_details") or {})
        regime = str(details.get("market_regime") or "")
        risk_mode = str(details.get("risk_mode") or "")
        if articles:
            from core.macro_pulse import macro_pulse
            macro = macro_pulse([a for a in articles if isinstance(a, Mapping)])
        elif not stale and isinstance(snap.get("macro"), Mapping):
            macro = dict(snap.get("macro") or {})
        else:
            macro = {}
        breadth = dict(snap.get("breadth") or {}) if not stale else {}
        edge = dict(snap.get("edge") or {}) if not stale else {}
        if edge.get("measured") is False:
            edge = {}
        correlation = dict(snap.get("correlation") or {}) if not stale else {}
        read = compose_desk_read(
            book_rows=rows,
            capital=capital,
            max_positions=max_n,
            regime=regime,
            market_risk_mode=risk_mode,
            edge=edge,
            breadth=breadth,
            macro=macro,
            correlation=correlation,
            snapshot_generated_at=str(snap.get("generated_at") or ""),
            now=now,
        )
        read["snapshot_stale"] = bool(stale)
        posture = str(read.get("posture") or "")
        action = str(read.get("action") or "")
        reason = str(read.get("posture_reason") or "")
        if posture == "STAND_ASIDE":
            base["trade_stance"] = f"{action} {reason}".strip()
        elif posture == "DEFENSIVE":
            prior = str(base.get("trade_stance") or "").strip()
            base["trade_stance"] = f"{action} {prior}".strip()
        base["brain"] = {
            "posture": posture,
            "posture_reason": reason,
            "action": action,
            "book_verdict": read.get("book_verdict"),
            "open_risk_pct": read.get("open_risk_pct"),
            "breadth_verdict": read.get("breadth_verdict") or "",
            "macro_mood": read.get("macro_mood") or "",
            "correlation_measured": bool(read.get("correlation_measured")),
            "correlation_positions": read.get("correlation_positions"),
            "correlation_bets": read.get("correlation_bets"),
            "directives": list(read.get("directives") or [])[:4],
            "snapshot_stale": bool(stale),
            "regime_measured": bool(read.get("regime_measured")),
            "live_locked": True,
        }
    except Exception:
        return base
    return base


def journal_cycle(cycle: Mapping[str, Any] | None) -> int:
    """Write TAKEN / REJECTED / WAIT rows from the reco paper cycle.

    The decision journal is the feedback store. Logging here does not place
    an order. Rows without a reference price are skipped by the journal.
    """
    from core.decision_journal import log_decision

    payload = dict(cycle or {})
    written = 0

    def _one(row: Mapping[str, Any], decision: str) -> None:
        nonlocal written
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            return
        entry = row.get("entry_fill") if row.get("entry_fill") not in (None, "") else row.get("entry")
        try:
            entry_ref = float(entry or 0.0)
            stop_ref = float(row.get("stop") or 0.0)
            score = float(row.get("selection_score") or row.get("score") or 0.0)
        except (TypeError, ValueError):
            return
        ev = row.get("ev_pct")
        p_win = row.get("p_win")
        try:
            ev_pct = None if ev in (None, "") else float(ev)
        except (TypeError, ValueError):
            ev_pct = None
        try:
            p_win_f = None if p_win in (None, "") else float(p_win)
        except (TypeError, ValueError):
            p_win_f = None
        log_decision(
            symbol,
            decision,
            reason=str(row.get("reason_code") or ""),
            source="reco_paper_cycle",
            entry_ref=entry_ref,
            stop_ref=stop_ref,
            score=score,
            ev_pct=ev_pct,
            p_win=p_win_f,
            confidence=str(row.get("tier") or "") or None,
        )
        written += 1

    for row in payload.get("taken") or []:
        if isinstance(row, Mapping):
            _one(row, "TAKEN")
    for row in payload.get("rejections") or []:
        if isinstance(row, Mapping):
            _one(row, "REJECTED")
    for row in payload.get("waits") or []:
        if isinstance(row, Mapping):
            _one(row, "WAIT")
    return written

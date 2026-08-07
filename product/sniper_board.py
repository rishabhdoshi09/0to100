"""Confirmed sniper breakouts → durable board → focused evaluation shortlist.

When the live breakout sniper confirms a hold above pivot, the hit is appended
here (real fields only). Owners can then run a focused evaluation across only
those symbols — joining the latest market scan, measured backtest edge, and a
long-term fundamentals/technical screen — to rank candidates for tomorrow's
watchlist or longer-horizon research.

Honesty rules:
  - never invent prices, edge, fundamentals, or verdicts
  - missing evidence stays missing and lowers coverage
  - evaluation is research ranking, not a buy instruction
  - paper-first: this board does not place orders
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BOARD_PATH = Path("logs/product/latest_sniper_board.json")
_SCHEMA = 1
_lock = threading.Lock()

# Keep the working set manageable for focused screens.
_MAX_HITS = 200
_DEFAULT_LOOKBACK_DAYS = 14


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    stamp = dt or _utc_now()
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.isoformat()


def _empty_board() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA,
        "updated_at": _iso(),
        "hits": [],
        "evaluation": None,
    }


def load_board(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_BOARD_PATH)
    if not target.exists():
        return _empty_board()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != _SCHEMA:
            return _empty_board()
        if not isinstance(payload.get("hits"), list):
            return _empty_board()
        return payload
    except Exception:
        return _empty_board()


def save_board(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path or DEFAULT_BOARD_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["schema_version"] = _SCHEMA
    body["updated_at"] = _iso()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _hit_key(hit: Mapping[str, Any]) -> tuple[str, str]:
    symbol = str(hit.get("symbol") or "").strip().upper()
    session = str(hit.get("session_date") or "").strip()
    return symbol, session


def normalize_hit(raw: Mapping[str, Any], *, confirmed_at: str | None = None) -> dict[str, Any] | None:
    """Keep only real sniper fields. Returns None when symbol/trigger/ltp missing."""
    symbol = str(raw.get("symbol") or "").strip().upper()
    trigger = _f(raw.get("trigger"))
    ltp = _f(raw.get("ltp"))
    if not symbol or trigger is None or trigger <= 0 or ltp is None or ltp <= 0:
        return None
    stamp = confirmed_at or str(raw.get("confirmed_at") or "") or _iso()
    try:
        session_date = datetime.fromisoformat(stamp.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        session_date = _utc_now().date().isoformat()
    if raw.get("session_date"):
        session_date = str(raw.get("session_date"))
    avg_vol = _f(raw.get("avg_vol"))
    cum_vol = _f(raw.get("cum_vol"))
    vol_pace = None
    if avg_vol and avg_vol > 0 and cum_vol is not None:
        vol_pace = round(cum_vol / avg_vol, 2)
    return {
        "symbol": symbol,
        "trigger": round(trigger, 4),
        "ltp": round(ltp, 4),
        "held_s": int(_f(raw.get("held_s")) or 0),
        "stop": _f(raw.get("stop")),
        "target": _f(raw.get("target")),
        "cum_vol": cum_vol,
        "avg_vol": avg_vol,
        "vol_pace": vol_pace,
        "confirmed_at": stamp,
        "session_date": session_date,
        "source": "breakout_sniper",
    }


def append_hits(
    hits: list[Mapping[str, Any]] | Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Append confirmed sniper hits. One hit per symbol per session_date."""
    target = Path(path or DEFAULT_BOARD_PATH)
    raw_list = [hits] if isinstance(hits, Mapping) else list(hits or [])
    normalized = [h for h in (normalize_hit(item) for item in raw_list) if h]
    if not normalized:
        return []
    with _lock:
        board = load_board(target)
        existing = list(board.get("hits") or [])
        seen = {_hit_key(h) for h in existing}
        fresh: list[dict[str, Any]] = []
        for hit in normalized:
            key = _hit_key(hit)
            if not key[0] or key in seen:
                continue
            existing.append(hit)
            seen.add(key)
            fresh.append(hit)
        if not fresh:
            return []
        # Newest first; cap size.
        existing.sort(key=lambda h: str(h.get("confirmed_at") or ""), reverse=True)
        board["hits"] = existing[:_MAX_HITS]
        # Stale evaluation until the owner re-ranks.
        board["evaluation"] = None
        save_board(board, target)
        return fresh


def board_symbols(
    board: Mapping[str, Any] | None = None,
    *,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    path: str | Path | None = None,
) -> list[str]:
    payload = dict(board) if board is not None else load_board(path)
    hits = list(payload.get("hits") or [])
    if lookback_days > 0:
        cutoff = _utc_now().date().toordinal() - int(lookback_days)
        kept = []
        for hit in hits:
            try:
                day = datetime.fromisoformat(str(hit.get("session_date"))).date().toordinal()
            except Exception:
                try:
                    day = datetime.fromisoformat(
                        str(hit.get("confirmed_at") or "").replace("Z", "+00:00")
                    ).date().toordinal()
                except Exception:
                    continue
            if day >= cutoff:
                kept.append(hit)
        hits = kept
    ordered: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        sym = str(hit.get("symbol") or "").upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered


def _scan_index(scan_payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not scan_payload:
        return out
    for row in list(scan_payload.get("records") or []):
        sym = str(row.get("symbol") or "").upper()
        if sym:
            out[sym] = dict(row)
    return out


def _long_term_index(records: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in records:
        sym = str(row.get("symbol") or "").upper()
        if sym:
            out[sym] = dict(row)
    return out


def _latest_hit_for(symbol: str, hits: list[Mapping[str, Any]]) -> dict[str, Any]:
    for hit in hits:
        if str(hit.get("symbol") or "").upper() == symbol:
            return dict(hit)
    return {}


def _breakout_quality(hit: Mapping[str, Any]) -> float | None:
    """0–100 quality from hold time + volume pace. None if no usable inputs."""
    held = _f(hit.get("held_s"))
    pace = _f(hit.get("vol_pace"))
    if held is None and pace is None:
        return None
    held_pts = 0.0
    if held is not None:
        # 45s baseline → ~60; longer holds score higher, capped.
        held_pts = max(0.0, min(100.0, (held / 45.0) * 60.0))
    pace_pts = 0.0
    if pace is not None:
        # 1.2× pace baseline → ~60; 2× → ~100.
        pace_pts = max(0.0, min(100.0, ((pace - 0.8) / 1.2) * 100.0))
    if held is not None and pace is not None:
        return round(0.55 * held_pts + 0.45 * pace_pts, 1)
    return round(held_pts if held is not None else pace_pts, 1)


def _composite_and_verdict(row: dict[str, Any]) -> tuple[float | None, str, list[str], list[str]]:
    """Return (rank_score, verdict, reasons, risks). Missing evidence stays visible."""
    reasons: list[str] = []
    risks: list[str] = []
    parts: list[tuple[str, float, float]] = []  # name, weight, points 0-100

    mom = _f(row.get("momentum_score"))
    if mom is not None:
        parts.append(("momentum", 0.30, max(0.0, min(100.0, mom))))
        if mom >= 70:
            reasons.append(f"Strong momentum score {mom:.0f}")
        elif mom < 45:
            risks.append(f"Weak momentum score {mom:.0f}")

    fund = _f(row.get("fundamental_score"))
    cov = _f(row.get("fundamental_coverage"))
    if fund is not None and cov is not None and cov >= 0.35:
        parts.append(("fundamentals", 0.30, max(0.0, min(100.0, fund))))
        if fund >= 70 and cov >= 0.5:
            reasons.append(f"Solid fundamentals {fund:.0f} · coverage {cov * 100:.0f}%")
        elif fund < 40:
            risks.append(f"Soft fundamentals {fund:.0f}")
    elif fund is None or cov is None or cov < 0.35:
        risks.append("Fundamental coverage incomplete")

    edge = _f(row.get("edge_r"))
    if edge is not None:
        # Map typical edge band ~[-0.3, +0.4] → 0–100.
        edge_pts = max(0.0, min(100.0, 50.0 + edge * 100.0))
        parts.append(("measured_edge", 0.25, edge_pts))
        if edge <= -0.05:
            risks.append(f"Measured LOSER edge {edge:+.2f}R")
        elif edge >= 0.08:
            reasons.append(f"Measured edge {edge:+.2f}R on backtest combo")
        else:
            reasons.append(f"Measured edge {edge:+.2f}R (near flat)")
    else:
        risks.append("No measured backtest edge for this signal combo")

    bq = _f(row.get("breakout_quality"))
    if bq is not None:
        parts.append(("breakout_quality", 0.15, max(0.0, min(100.0, bq))))
        if bq >= 70:
            reasons.append(f"Clean confirmed breakout quality {bq:.0f}")
        elif bq < 40:
            risks.append(f"Weak breakout confirmation quality {bq:.0f}")

    if row.get("chase_risk"):
        risks.append("Scanner flagged chase / extension risk")

    for flag in list(row.get("risk_flags") or [])[:3]:
        text = str(flag)
        if text and text not in risks:
            risks.append(text)

    coverage = len(parts) / 4.0
    row["evidence_coverage"] = round(coverage, 2)

    if not parts:
        return None, "INCOMPLETE", reasons or ["No momentum, fundamentals, or edge available"], risks

    weight_sum = sum(w for _, w, _ in parts)
    score = sum(w * pts for _, w, pts in parts) / weight_sum if weight_sum else None
    if score is not None:
        score = round(score, 1)

    # Verdict — conservative, research-oriented.
    if edge is not None and edge <= -0.05:
        return score, "AVOID", reasons, risks
    if coverage < 0.4:
        return score, "INCOMPLETE", reasons, risks
    if row.get("chase_risk") and (edge is None or edge < 0.05):
        return score, "WATCH", reasons, risks
    if score is not None and score >= 72 and coverage >= 0.6 and (edge is None or edge >= 0):
        return score, "PRIORITY", reasons, risks
    if score is not None and score >= 55:
        return score, "CANDIDATE", reasons, risks
    if score is not None and score >= 40:
        return score, "WATCH", reasons, risks
    return score, "WEAK", reasons, risks


def _consider_for(verdict: str, row: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    if verdict in {"PRIORITY", "CANDIDATE"}:
        tags.append("tomorrow_watch")
    fund = _f(row.get("fundamental_score"))
    cov = _f(row.get("fundamental_coverage"))
    classification = str(row.get("classification") or "")
    if (
        verdict in {"PRIORITY", "CANDIDATE"}
        and fund is not None
        and cov is not None
        and cov >= 0.45
        and fund >= 55
        and classification in {"QUALITY_COMPOUNDER", "GARP_CANDIDATE", ""}
    ):
        tags.append("long_term_shortlist")
    if verdict == "WATCH":
        tags.append("needs_pullback_or_more_evidence")
    if verdict == "AVOID":
        tags.append("do_not_chase")
    if verdict == "INCOMPLETE":
        tags.append("refresh_scan_or_fundamentals")
    return tags


def evaluate_board(
    *,
    path: str | Path | None = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    refresh_fundamentals: bool = False,
    progress=None,
    save: bool = True,
) -> dict[str, Any]:
    """Rank sniper-board symbols using scan + fundamentals + measured edge.

    Does not invent missing evidence. Persists evaluation onto the board file.
    """
    target = Path(path or DEFAULT_BOARD_PATH)

    def _progress(current: int, total: int, message: str) -> None:
        if not callable(progress):
            return
        try:
            progress(int(current), int(total), str(message))
        except Exception:
            pass

    board = load_board(target)
    hits = list(board.get("hits") or [])
    symbols = board_symbols(board, lookback_days=lookback_days, path=target)
    if not symbols:
        evaluation = {
            "evaluated_at": _iso(),
            "lookback_days": int(lookback_days),
            "symbols": [],
            "records": [],
            "summary": {
                "hits": len(hits),
                "unique_symbols": 0,
                "priority": 0,
                "candidate": 0,
                "watch": 0,
                "avoid": 0,
                "incomplete": 0,
                "with_measured_edge": 0,
                "with_fundamentals": 0,
            },
            "honesty": (
                "No confirmed sniper breakouts in the lookback window. "
                "The board fills when live sniper confirms a held pivot break."
            ),
            "places_orders": False,
            "live_locked": True,
        }
        if save:
            with _lock:
                board = load_board(target)
                board["evaluation"] = evaluation
                save_board(board, target)
        return evaluation

    _progress(0, len(symbols), f"Loading market scan context for {len(symbols)} sniper symbols")
    try:
        from product.scan_store import load_scan

        scan_payload = load_scan()
    except Exception:
        scan_payload = None
    scan_by_sym = _scan_index(scan_payload)

    _progress(0, len(symbols), "Running focused long-term screen on sniper list only")
    lt_records: list[dict[str, Any]] = []
    lt_status = "SKIPPED"
    lt_error = ""
    try:
        from scan.long_term_service import run_long_term_scan

        report = run_long_term_scan(
            symbols=symbols,
            scope="sniper_board",
            refresh_fundamentals=bool(refresh_fundamentals),
            save=False,
            top=max(40, len(symbols)),
            progress=progress,
        )
        lt_status = str(getattr(report, "status", "") or "")
        lt_error = str(getattr(report, "error_message", "") or "")
        payload = dict(getattr(report, "payload", {}) or {})
        lt_records = [dict(r) for r in (payload.get("records") or [])]
    except Exception as exc:
        lt_status = "FAILED"
        lt_error = f"{type(exc).__name__}: {exc}"
    lt_by_sym = _long_term_index(lt_records)

    # Newest hits first for latest_hit lookup.
    hits_sorted = sorted(hits, key=lambda h: str(h.get("confirmed_at") or ""), reverse=True)

    records: list[dict[str, Any]] = []
    for idx, symbol in enumerate(symbols):
        _progress(idx + 1, len(symbols), f"Ranking · {idx + 1}/{len(symbols)} · {symbol}")
        scan = scan_by_sym.get(symbol) or {}
        lt = lt_by_sym.get(symbol) or {}
        hit = _latest_hit_for(symbol, hits_sorted)
        signals = list(scan.get("signals") or [])
        edge = scan.get("edge_r")
        if edge is None and signals:
            try:
                from scan.signal_backtest import combo_edge

                edge = combo_edge([str(s) for s in signals])
            except Exception:
                edge = None

        row: dict[str, Any] = {
            "symbol": symbol,
            "company": scan.get("company") or lt.get("company") or symbol,
            "confirmed_at": hit.get("confirmed_at"),
            "session_date": hit.get("session_date"),
            "trigger": hit.get("trigger"),
            "confirm_ltp": hit.get("ltp"),
            "held_s": hit.get("held_s"),
            "vol_pace": hit.get("vol_pace"),
            "stop": hit.get("stop") if hit.get("stop") is not None else scan.get("stop"),
            "target": hit.get("target") if hit.get("target") is not None else scan.get("target"),
            "scan_verdict": scan.get("verdict"),
            "scan_status": scan.get("status"),
            "momentum_score": _f(scan.get("score")),
            "momentum_5d": _f(scan.get("momentum_5d")),
            "rsi": _f(scan.get("rsi")),
            "volume_ratio": _f(scan.get("volume_ratio")),
            "chase_risk": bool(scan.get("chase_risk")),
            "signals": signals,
            "price": _f(scan.get("price")) or _f(lt.get("price")) or _f(hit.get("ltp")),
            "edge_r": float(edge) if edge is not None else None,
            "classification": lt.get("classification"),
            "technical_score": _f(lt.get("technical_score")),
            "fundamental_score": _f(lt.get("fundamental_score")),
            "fundamental_coverage": _f(lt.get("fundamental_coverage")),
            "combined_score": _f(lt.get("combined_score")),
            "timing": lt.get("timing"),
            "sector": lt.get("sector") or scan.get("sector"),
            "quality_factors": list(lt.get("quality_factors") or []),
            "risk_flags": list(lt.get("risk_flags") or []),
            "breakout_quality": _breakout_quality(hit),
        }
        rank_score, verdict, reasons, risks = _composite_and_verdict(row)
        row["rank_score"] = rank_score
        row["verdict"] = verdict
        row["reasons"] = reasons
        row["risks"] = risks
        row["consider_for"] = _consider_for(verdict, row)
        records.append(row)

    vrank = {"PRIORITY": 4, "CANDIDATE": 3, "WATCH": 2, "WEAK": 1, "INCOMPLETE": 0, "AVOID": -1}
    records.sort(
        key=lambda r: (
            vrank.get(str(r.get("verdict") or ""), -2),
            float(r.get("rank_score") or -1),
            float(r.get("edge_r") or -99),
            str(r.get("symbol") or ""),
        ),
        reverse=True,
    )

    summary = {
        "hits": len(hits),
        "unique_symbols": len(symbols),
        "priority": sum(1 for r in records if r.get("verdict") == "PRIORITY"),
        "candidate": sum(1 for r in records if r.get("verdict") == "CANDIDATE"),
        "watch": sum(1 for r in records if r.get("verdict") == "WATCH"),
        "avoid": sum(1 for r in records if r.get("verdict") == "AVOID"),
        "incomplete": sum(1 for r in records if r.get("verdict") == "INCOMPLETE"),
        "weak": sum(1 for r in records if r.get("verdict") == "WEAK"),
        "with_measured_edge": sum(1 for r in records if r.get("edge_r") is not None),
        "with_fundamentals": sum(
            1 for r in records
            if r.get("fundamental_score") is not None
            and (_f(r.get("fundamental_coverage")) or 0) >= 0.35
        ),
        "scan_context_available": bool(scan_payload),
        "long_term_status": lt_status,
    }
    evaluation = {
        "evaluated_at": _iso(),
        "lookback_days": int(lookback_days),
        "symbols": symbols,
        "records": records,
        "summary": summary,
        "long_term_error": lt_error or None,
        "honesty": (
            "Ranked from confirmed sniper hits only. Missing scan, fundamentals, or "
            "backtest edge stays missing — never invented. Research ranking for "
            "tomorrow-watch / long-term shortlist consideration; not a buy order."
        ),
        "places_orders": False,
        "live_locked": True,
    }
    if save:
        with _lock:
            board = load_board(target)
            board["evaluation"] = evaluation
            save_board(board, target)
    return evaluation


def board_api_payload(
    *,
    path: str | Path | None = None,
    record_limit: int = 80,
) -> dict[str, Any]:
    """Lean payload for /api/sniper-board and optional dashboard embed."""
    board = load_board(path)
    hits = list(board.get("hits") or [])[: max(0, int(record_limit))]
    evaluation = board.get("evaluation")
    eval_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    evaluated_at = None
    if isinstance(evaluation, Mapping):
        evaluated_at = evaluation.get("evaluated_at")
        summary = dict(evaluation.get("summary") or {})
        eval_records = list(evaluation.get("records") or [])[: max(0, int(record_limit))]
    try:
        from scan.breakout_sniper import sniper_status

        runtime = sniper_status()
    except Exception:
        runtime = {"started": False, "watching": 0, "fired_today": []}
    return {
        "available": True,
        "updated_at": board.get("updated_at"),
        "hits": hits,
        "hit_count": len(board.get("hits") or []),
        "symbols": board_symbols(board, path=path),
        "evaluated_at": evaluated_at,
        "evaluation_summary": summary,
        "evaluation_records": eval_records,
        "evaluation": evaluation if isinstance(evaluation, Mapping) else None,
        "sniper_runtime": runtime,
        "places_orders": False,
        "live_locked": True,
        "honesty": (
            "Confirmed live sniper breakouts only. Evaluation joins saved market scan, "
            "measured edge, and focused long-term screen — never invents missing evidence."
        ),
    }

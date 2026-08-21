"""Trade desk: ready queue, backtest lab, paper→live journey.

Research (Ideas) stays a monitor. This module is the production path.

  Ready  — ticket board: Stage-2 SEPA overlay + scanner BUY/Ready plans,
           ranked by Asymmetric Ticket Quality (ATQ). Prime is an overlay
           when those gates actually pass — not the only lane.
  Lab    — walk-forward backtest as use cases, not a dump
  Journey — paper autopilot → live, earned, never toggled

Never invents EV, never arms LIVE, never hard-wires a ticker.
GET Ready is cache-only: last scan + last Ideas SEPA ranking. It does not
re-run rank_best_setups.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.breakout_quality import RSI_HARD
from product.production_ladder import (
    PAPER_E4_N,
    PAPER_TRANSITION_N,
    build_production_ladder,
    live_arm_allowed,
)

READY_EDGE_FLOOR = 0.05  # R; below this is thin, not a money claim
RSI_SOFT_DEMOTE = 72.0
SEPA_READY_FLOOR = 40
SEPA_WATCH_FLOOR = 70
STAGE2_TICKET_CAP = 12
READY_TICKET_CAP = 16
PRIME_TICKET_CAP = 8
ATQ_METHOD = (
    "ATQ v1 — Asymmetric Ticket Quality: reward/risk × scan score × "
    "SEPA Stage-2 overlay × volume vs 20d × RSI room. Conservative EV "
    "is an overlay only; missing EV never zeros a complete ticket. "
    "Not a win-rate claim."
)


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        n = float(value)
    except (TypeError, ValueError):
        return None
    if n != n:  # NaN
        return None
    return n


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _upside(entry: float | None, target: float | None) -> float | None:
    if entry is None or target is None or entry <= 0:
        return None
    return round((target / entry - 1.0) * 100.0, 1)


def reward_risk(entry: Any, stop: Any, target: Any) -> float | None:
    e, s, t = _f(entry), _f(stop), _f(target)
    if e is None or s is None or t is None:
        return None
    risk = e - s
    reward = t - e
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 2)


def ticket_quality(row: Mapping[str, Any]) -> float:
    """ATQ in 0..~2. Structure-first; missing EV does not wipe the score."""
    rr = reward_risk(row.get("entry"), row.get("stop"), row.get("target"))
    r_star = _clip((rr or 1.0) / 2.0, 0.5, 1.5)

    score = _f(row.get("score")) or 0.0
    scan_frac = _clip(score / 100.0, 0.0, 1.0)

    sepa = _f(row.get("sepa_score"))
    sepa_frac = _clip(0.55 + 0.45 * (sepa / 100.0), 0.55, 1.0) if sepa is not None else 0.70

    vol_ratio = _f(row.get("volume_ratio"))
    if vol_ratio is None:
        vol_frac = 0.80
    else:
        vol_frac = _clip(0.55 + 0.20 * vol_ratio, 0.55, 1.15)

    rsi = _f(row.get("rsi"))
    if rsi is None:
        rsi_frac = 0.90
    elif rsi >= RSI_HARD:
        rsi_frac = 0.0
    elif rsi >= RSI_SOFT_DEMOTE:
        rsi_frac = 0.55
    else:
        rsi_frac = 1.0

    edge_r = _f(row.get("edge_r"))
    if edge_r is None:
        edge_frac = 1.0
    elif edge_r <= -READY_EDGE_FLOOR:
        edge_frac = 0.0
    else:
        edge_frac = _clip(0.85 + 0.08 * edge_r, 0.85, 1.20)

    return round(
        max(0.0, r_star * scan_frac * sepa_frac * vol_frac * rsi_frac * edge_frac),
        4,
    )


def _breadth_verdict(market: Mapping[str, Any] | None) -> str:
    market = dict(market or {})
    raw = str(market.get("breadth") or market.get("breadth_verdict") or "").strip()
    if raw:
        return raw.split()[0].upper()
    health = str(market.get("health") or "").strip().lower()
    if health == "narrow":
        return "NARROW"
    return raw.upper()


def _enrich_scan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    cats = {str(c) for c in (out.get("categories") or []) if c}
    sigs = [str(s) for s in (out.get("signals") or [])]
    joined = " ".join(sigs).upper()
    if any("MOMENTUM" in s.upper() for s in sigs) or "Momentum" in cats:
        cats.add("Momentum")
    if "BREAKOUT" in joined and "PRE_BREAKOUT" not in joined:
        cats.add("Breakout")
    out["categories"] = sorted(cats)
    return out


def _has_plan(row: Mapping[str, Any]) -> bool:
    entry = _f(row.get("entry"))
    stop = _f(row.get("stop"))
    return bool(entry and stop and entry > 0 and stop > 0 and stop < entry)


def _loser_edge(row: Mapping[str, Any]) -> bool:
    edge = _f(row.get("edge_r"))
    if edge is not None and edge <= -READY_EDGE_FLOOR:
        return True
    ev_lb = _f(row.get("ev_lb_pct"))
    if ev_lb is not None and ev_lb <= 0:
        return True
    return False


def _rsi_hard_reject(row: Mapping[str, Any]) -> bool:
    rsi = _f(row.get("rsi"))
    return rsi is not None and rsi >= RSI_HARD


def _is_chase(row: Mapping[str, Any]) -> bool:
    return bool(row.get("chase_risk") or row.get("extended"))


def _actionable(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Complete BUY/Ready ticket. Pattern / PreBreakout / Pullback count.

    Prime still requires Momentum/Breakout via prime_check. Ready used to
    copy that veto and emptied the board on real scans.
    """
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict not in {"BUY", "STRONG BUY"} and status != "Ready to trade":
        return False, "Scanner did not mark BUY / Ready to trade"
    if _is_chase(row):
        return False, "Chase / extension — late entry"
    if not _has_plan(row):
        return False, "No entry/stop — no risk plan"
    if _loser_edge(row):
        return False, "Measured edge or conservative EV is not positive"
    if _rsi_hard_reject(row):
        return False, "RSI blow-off — no room to run"
    return True, "Complete ticket; not a return promise"


def _honesty(lane: str) -> str:
    if lane == "prime":
        return (
            "Prime means every evidence gate passed — not a guarantee. "
            "Paper first. Live stays locked until the journey earns it."
        )
    if lane == "stage2":
        return (
            "Stage-2 SEPA overlay from the last Ideas ranking on official "
            "history. Template fit, not a win rate."
        )
    return (
        "Complete ticket ranked by ATQ. Missing EV is not invented — "
        "not a high-chance claim."
    )


def _card(row: Mapping[str, Any], *, lane: str, why: Sequence[str]) -> dict[str, Any]:
    entry = _f(row.get("entry"))
    stop = _f(row.get("stop"))
    target = _f(row.get("target"))
    ev_n = int(row.get("ev_n") or 0)
    ev: dict[str, Any] = {}
    if ev_n >= 30 and row.get("ev_pct") is not None:
        ev = {
            "ev_pct": _f(row.get("ev_pct")),
            "ev_lb_pct": _f(row.get("ev_lb_pct")),
            "ev_n": ev_n,
            "ev_conf": str(row.get("ev_conf") or ""),
            "p_win": _f(row.get("p_win")),
        }
    sepa = _f(row.get("sepa_score"))
    passed = row.get("sepa_passed")
    total = row.get("sepa_total")
    try:
        sepa_passed = int(passed) if passed is not None else None
    except (TypeError, ValueError):
        sepa_passed = None
    try:
        sepa_total = int(total) if total is not None else None
    except (TypeError, ValueError):
        sepa_total = None
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "lane": lane,
        "verdict": str(row.get("verdict") or ""),
        "status": str(row.get("status") or ""),
        "sector": str(row.get("sector") or "—"),
        "score": _f(row.get("score")),
        "atq": ticket_quality(row),
        "reward_risk": reward_risk(entry, stop, target),
        "edge_r": _f(row.get("edge_r")),
        "entry": entry,
        "stop": stop,
        "target": target,
        "cmp": _f(row.get("price") or row.get("cmp")),
        "upside_from_buy_pct": _upside(entry, target),
        "volume_ratio": _f(row.get("volume_ratio")),
        "rsi": _f(row.get("rsi")),
        "sepa_score": int(sepa) if sepa is not None else None,
        "sepa_passed": sepa_passed,
        "sepa_total": sepa_total,
        "sepa_verdict": str(row.get("sepa_verdict") or "") or None,
        "sepa_headline": str(row.get("sepa_headline") or "") or None,
        "stage_label": str(row.get("stage_label") or row.get("sepa_stage") or "") or None,
        "categories": [str(c) for c in (row.get("categories") or []) if c],
        "signals": [str(s) for s in (row.get("signals") or []) if s],
        "source": str(row.get("source") or "scan"),
        "why": [str(x) for x in why if x],
        "honesty": _honesty(lane),
        **ev,
    }


def _extract_best_setups(workspace: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not workspace:
        return []
    for cat in workspace.get("categories") or []:
        if isinstance(cat, dict) and cat.get("id") == "best_setups":
            return [c for c in (cat.get("cards") or []) if isinstance(c, dict)]
    cards = workspace.get("best_setups")
    if isinstance(cards, list):
        return [c for c in cards if isinstance(c, dict)]
    return []


_WORKSPACE_UNSET = object()


def _load_sepa_workspace(workspace: Any) -> Mapping[str, Any] | None:
    if workspace is not _WORKSPACE_UNSET:
        return workspace
    try:
        from product.recommendations_workspace import load_last_recommendations_workspace
        return load_last_recommendations_workspace()
    except Exception:
        return None


def _merge_sepa_row(card: Mapping[str, Any], scan_by_symbol: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    symbol = str(card.get("symbol") or "").upper().strip()
    scan_row = dict(scan_by_symbol.get(symbol) or {})
    merged = dict(scan_row)
    merged["symbol"] = symbol
    merged["company"] = card.get("company") or scan_row.get("company") or symbol
    merged["source"] = "sepa"
    for key in ("entry", "stop", "target"):
        if _f(merged.get(key)) is None:
            merged[key] = card.get(key)
    if _f(merged.get("score")) is None:
        merged["score"] = card.get("score") or card.get("sepa_score")
    merged["sepa_score"] = card.get("sepa_score")
    merged["sepa_passed"] = card.get("sepa_passed")
    merged["sepa_total"] = card.get("sepa_total")
    merged["sepa_verdict"] = card.get("sepa_verdict")
    merged["sepa_headline"] = card.get("sepa_headline") or card.get("setup_label")
    merged["stage_label"] = card.get("stage_label")
    if not merged.get("sector"):
        merged["sector"] = card.get("sector")
    if _f(merged.get("rsi")) is None:
        merged["rsi"] = card.get("rsi")
    if _f(merged.get("volume_ratio")) is None:
        merged["volume_ratio"] = card.get("volume_ratio")
    return merged


def _stage2_ok(row: Mapping[str, Any]) -> bool:
    sepa = _f(row.get("sepa_score")) or 0.0
    if sepa < SEPA_READY_FLOOR:
        return False
    if not _has_plan(row) or _is_chase(row) or _loser_edge(row) or _rsi_hard_reject(row):
        return False
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict in {"BUY", "STRONG BUY"} or status == "Ready to trade":
        return True
    return sepa >= SEPA_WATCH_FLOOR


def _rank_card(card: Mapping[str, Any]) -> tuple:
    atq = _f(card.get("atq")) or 0.0
    sepa = _f(card.get("sepa_score")) or 0.0
    has_ev = 1 if card.get("ev_pct") is not None and int(card.get("ev_n") or 0) >= 30 else 0
    lb = _f(card.get("ev_lb_pct"))
    edge = _f(card.get("edge_r")) or 0.0
    score = _f(card.get("score")) or 0.0
    return (atq, sepa, has_ev, lb if lb is not None else -999.0, edge, score)


def build_ready_queue(
    *,
    scan: Mapping[str, Any] | None = None,
    market: Mapping[str, Any] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    workspace: Any = _WORKSPACE_UNSET,
) -> dict[str, Any]:
    """Ticket board from last scan + cached SEPA. Empty is a valid answer."""
    scan = dict(scan or {})
    if records is None:
        records = list(scan.get("records") or [])
    rows = [_enrich_scan_row(r) for r in records if isinstance(r, Mapping)]
    try:
        from scan.ev_engine import tag_ev
        tag_ev(rows)
    except Exception:
        pass

    breadth = _breadth_verdict(market)
    demoted: set[str] = set()
    prime: list[dict[str, Any]] = []
    actionable: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    scan_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if symbol:
            scan_by_symbol[symbol] = row

    try:
        from scan.prime_filter import prime_check
    except Exception:
        prime_check = None  # type: ignore[assignment]

    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        ok_plan, plan_why = _actionable(row)
        is_prime = False
        prime_why: list[str] = []
        fail = ""
        if prime_check is not None:
            is_prime, prime_why, fail = prime_check(row, breadth, demoted)
        if (
            is_prime
            and _has_plan(row)
            and not _loser_edge(row)
            and not _is_chase(row)
            and not _rsi_hard_reject(row)
        ):
            prime.append(_card(row, lane="prime", why=prime_why))
            continue
        if ok_plan:
            actionable.append(_card(row, lane="actionable", why=[plan_why]))
            continue
        if str(row.get("verdict") or "").upper() in {"BUY", "STRONG BUY"} or str(row.get("status") or "") == "Ready to trade":
            rejected.append({"symbol": symbol, "reason": fail or plan_why})

    sepa_cards = _extract_best_setups(_load_sepa_workspace(workspace))
    stage2: list[dict[str, Any]] = []
    prime_symbols = {str(c.get("symbol") or "").upper() for c in prime}
    for card in sepa_cards:
        symbol = str(card.get("symbol") or "").upper().strip()
        if not symbol or symbol in prime_symbols:
            continue
        merged = _merge_sepa_row(card, scan_by_symbol)
        if not _stage2_ok(merged):
            continue
        why = [merged.get("sepa_headline") or "SEPA Stage-2"]
        if merged.get("sepa_passed") is not None and merged.get("sepa_total"):
            why.append(f"{merged['sepa_passed']}/{merged['sepa_total']} rules")
        if merged.get("stage_label"):
            why.append(str(merged["stage_label"]))
        stage2.append(_card(merged, lane="stage2", why=why))

    stage2_symbols = {str(c.get("symbol") or "").upper() for c in stage2}
    actionable = [
        c for c in actionable if str(c.get("symbol") or "").upper() not in stage2_symbols
    ]

    prime.sort(key=_rank_card, reverse=True)
    stage2.sort(key=_rank_card, reverse=True)
    actionable.sort(key=_rank_card, reverse=True)
    prime = prime[:PRIME_TICKET_CAP]
    stage2 = stage2[:STAGE2_TICKET_CAP]
    actionable = actionable[:READY_TICKET_CAP]

    empty = not prime and not stage2 and not actionable
    empty_why: list[str] = []
    if not rows and not stage2:
        empty_why.append("No scan is on file. Run a market scan first.")
    elif empty:
        empty_why.append(
            "No complete tickets today — BUY/Ready with entry and stop, "
            "not a chase, not a measured loser, RSI under 82."
        )
        if rejected:
            empty_why.append(
                f"{len(rejected)} BUY name(s) failed a gate (chase, missing stop, loser edge, or blow-off RSI)."
            )
    if breadth == "NARROW" and not empty:
        empty_why.append("Breadth is NARROW — size down. Prime stays vetoed.")

    return {
        "schema_version": 2,
        "places_orders": False,
        "live_locked": True,
        "method": ATQ_METHOD,
        "scanned_at": scan.get("scanned_at") or "",
        "universe_size": int(scan.get("universe_size") or len(rows)),
        "breadth": breadth or "—",
        "stage2": stage2,
        "prime": prime,
        "actionable": actionable,
        "rejected_n": len(rejected),
        "rejected_sample": rejected[:8],
        "empty": empty,
        "empty_why": empty_why,
        "disclaimer": (
            "Ready is not a broker order and not a return promise. "
            "Stage-2 is Minervini SEPA on official history (cached Ideas ranking). "
            "ATQ ranks structure; it is not a win rate. "
            "Prime still uses Telegram 💎 gates when they actually pass. "
            "Missing numbers stay missing. Lab losers are demoted on the scan — never inflated."
        ),
        "next": (
            "Open Lab to see which signals earned in this tape, then start the paper classroom."
            if not empty
            else "Fill the desk (scan + Ideas SEPA) before expecting a Ready name."
        ),
        "lab_applied": _ready_lab_applied(),
    }


def _ready_lab_applied() -> dict[str, Any]:
    """Tiny strip for Ready: Lab keep/skip already in the scan. Never invents."""
    try:
        from scan.signal_backtest import trading_playbook
        playbook = dict(trading_playbook() or {})
    except Exception:
        playbook = {}
    keep = [str(r.get("signal") or r) for r in (playbook.get("best") or []) if r][:5]
    skip = [str(x) for x in (playbook.get("avoid") or []) if x][:8]
    applied = bool(keep or skip)
    return {
        "applied": applied,
        "regime": playbook.get("regime") or "",
        "keep": keep,
        "skip": skip,
        "plain": (
            "Lab learning is already in this board: proven losers were demoted on the scan. "
            "Missing EV stays missing."
            if applied
            else "No Lab keep/skip list yet — this board is structure-only, not a win-rate claim."
        ),
    }


def _kid_verdict(verdict: str) -> dict[str, str]:
    v = str(verdict or "").upper()
    if v in {"PROVEN", "POSITIVE"}:
        return {
            "kid_lane": "keep",
            "kid_label": "Passed",
            "kid_hint": "This signal made money in practice. Keep using it.",
        }
    if v == "LOSER":
        return {
            "kid_lane": "skip",
            "kid_label": "Failed",
            "kid_hint": "This signal lost money in practice. Skip it.",
        }
    return {
        "kid_lane": "quiet",
        "kid_label": "Too few tries",
        "kid_hint": "Under 30 practice trades we stay quiet. No bragging.",
    }


def _signal_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, raw in dict(report.get("signals") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        verdict = str(raw.get("verdict") or "THIN")
        row = {
            "signal": str(key),
            "trades": int(raw.get("trades") or 0),
            "closed": int(raw.get("closed") or raw.get("trades") or 0),
            "win_rate": _f(raw.get("win_rate")),
            "expectancy_r": _f(raw.get("expectancy_r")),
            "verdict": verdict,
        }
        row.update(_kid_verdict(verdict))
        out.append(row)
    out.sort(key=lambda r: (r["verdict"] != "PROVEN", -(r["expectancy_r"] or -99.0)))
    return out


def _lab_scoreboard(signals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keep = [s for s in signals if s.get("kid_lane") == "keep"]
    skip = [s for s in signals if s.get("kid_lane") == "skip"]
    quiet = [s for s in signals if s.get("kid_lane") == "quiet"]
    if not signals:
        headline = (
            "No practice test on file yet. Press Run the practice test — "
            "official NSE history, never spends money."
        )
    else:
        headline = (
            f"{len(keep)} passed · {len(skip)} failed · {len(quiet)} too quiet to claim. "
            "R is rupees won per rupee risked, on average."
        )
    return {
        "headline": headline,
        "keep": list(keep),
        "skip": list(skip),
        "quiet": list(quiet),
    }


def _lab_lesson(*, running: bool, actionable: bool, has_report: bool) -> dict[str, Any]:
    if running:
        now = "The practice test is running. Wait. Do not treat Ready names as proven yet."
    elif actionable:
        now = "Practice is big enough to read. Green = keep. Red = skip. Grey = stay quiet."
    elif has_report:
        now = "A report exists, but the sample is too small or truncated. Run it again on all stocks."
    else:
        now = "No practice test yet. Press the big button. Official NSE history only — no fake scores."
    return {
        "title": "What is a backtest?",
        "plain": (
            "A backtest is cricket nets, not the match. We go back in official NSE history, "
            "pretend the scanner said BUY that day, and check: win, lose, or nowhere in the next "
            "10 sessions. We never peek at tomorrow. That is cheating. We never place an order. "
            "That would be the real match."
        ),
        "r_plain": (
            "R means: if you risk ₹1, +0.22R is 22 paise extra on average. "
            "−0.15R is 15 paise lost on average. Do that 100 times and the loser hurts."
        ),
        "now": now,
        "steps": [
            {
                "n": 1,
                "title": "Press the button",
                "body": "Run the practice test on all stocks in official history.",
            },
            {
                "n": 2,
                "title": "Wait for the bar",
                "body": "It walks day by day. No peeking at the next close.",
            },
            {
                "n": 3,
                "title": "Read the scoreboard",
                "body": "Passed = keep. Failed = skip. Too few tries = we do not brag.",
            },
            {
                "n": 4,
                "title": "Then paper, not live",
                "body": "Journey still uses fake money. This page cannot arm live.",
            },
        ],
        "rules": [
            "Under 30 practice trades = no claim.",
            "A truncated run is homework half-done.",
            "This never buys or sells. Live stays locked.",
        ],
        "cta": "Run the practice test",
        "cta_running": "Practicing…",
    }


def _live_edge_snapshot() -> dict[str, Any]:
    """Demote-only live outcomes. Empty when the sample is thin — never invented."""
    try:
        from scan.live_edge import profile_edge

        prof = profile_edge() or {}
    except Exception:
        return {"n": 0, "expectancy_r": None, "win_rate": None, "signals": []}
    overall = dict(prof.get("overall") or {})
    n = int(overall.get("n") or 0)
    signals = []
    for key, raw in dict(prof.get("signals") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        sn = int(raw.get("n") or 0)
        if sn < 30:
            continue
        signals.append({
            "signal": str(key),
            "n": sn,
            "win_rate": _f(raw.get("win_rate")),
            "expectancy_r": _f(raw.get("expectancy_r")),
        })
    signals.sort(key=lambda r: (r.get("expectancy_r") is None, -(r.get("expectancy_r") or -99)))
    return {
        "n": n,
        "expectancy_r": _f(overall.get("expectancy_r")) if n else None,
        "win_rate": _f(overall.get("win_rate")) if n else None,
        "claim": n >= 30,
        "signals": signals[:8],
        "plain": (
            f"{n} closed tracked outcomes — demote-only into the next scan."
            if n >= 30
            else (
                f"{n} closed tracked outcomes — under 30 we stay quiet. "
                "No live learning claim yet."
                if n
                else "No closed tracked outcomes yet. Paper classroom fills this."
            )
        ),
    }


def _lab_learning(*, playbook: Mapping[str, Any], signals: Sequence[Mapping[str, Any]],
                  live: Mapping[str, Any] | None = None) -> dict[str, Any]:
    keep = [s["signal"] for s in signals if s.get("kid_lane") == "keep"]
    skip = [s["signal"] for s in signals if s.get("kid_lane") == "skip"]
    avoid = [str(x) for x in (playbook.get("avoid") or skip) if x]
    best = list(playbook.get("best") or [])[:5]
    live = dict(live or {})
    applied = bool(keep or avoid or live.get("claim"))
    return {
        "applied": applied,
        "demote_only": True,
        "regime": playbook.get("regime") or "UNKNOWN",
        "keep": keep[:8],
        "skip": avoid[:12],
        "best": best,
        "live": live,
        "plain": (
            "Next Ideas / Ready scan already uses this. Proven losers are "
            "demoted. Missing numbers stay missing — we never inflate a win rate."
            if applied
            else (
                "No Lab learning on file yet. Run the practice test, then paper. "
                "Recommendations will not invent a win rate in the meantime."
            )
        ),
    }


def _lab_pulse(*, running: bool, actionable: bool, classroom: Mapping[str, Any]) -> dict[str, Any]:
    cls = dict(classroom or {})
    if running:
        return {"label": "Practice test running", "tone": "run", "hint": "Official NSE history. No orders."}
    if cls.get("open_n"):
        return {"label": "Paper classroom in session", "tone": "live", "hint": "Fake money. Live stays locked."}
    if cls.get("armed") and cls.get("in_window"):
        return {"label": "Armed — waiting for a Ready ticket", "tone": "live", "hint": "Gates still apply."}
    if cls.get("armed"):
        return {"label": "Armed — waiting for the next session", "tone": "wait", "hint": cls.get("headline") or ""}
    if actionable:
        return {"label": "Learning on file — start paper", "tone": "ready", "hint": "Practice passed. Paper is the match."}
    return {"label": "Classroom idle", "tone": "idle", "hint": "Run the practice test, then arm paper."}


def _lab_loop(*, running: bool, actionable: bool, evidence_note: str,
              learning: Mapping[str, Any], classroom: Mapping[str, Any]) -> list[dict[str, Any]]:
    learn = dict(learning or {})
    cls = dict(classroom or {})
    paper_state = "WAIT"
    if cls.get("open_n") or (cls.get("armed") and cls.get("in_window")):
        paper_state = "LIVE"
    elif cls.get("armed"):
        paper_state = "ARMED"
    elif cls.get("closed_n"):
        paper_state = "READY"
    return [
        {
            "id": "practice",
            "title": "Practice test",
            "n": 1,
            "state": "RUN" if running else ("READY" if actionable else "IDLE"),
            "detail": evidence_note,
        },
        {
            "id": "teach",
            "title": "What we learned",
            "n": 2,
            "state": "READY" if learn.get("applied") else "IDLE",
            "detail": (
                f"{len(learn.get('keep') or [])} keep · {len(learn.get('skip') or [])} skip"
                if learn.get("applied")
                else "No keep/skip list yet"
            ),
        },
        {
            "id": "paper",
            "title": "Paper classroom",
            "n": 3,
            "state": paper_state,
            "detail": cls.get("headline") or "Disarmed — default OFF",
        },
        {
            "id": "recos",
            "title": "Next recommendations",
            "n": 4,
            "state": "READY" if learn.get("applied") else "IDLE",
            "detail": learn.get("plain") or "",
        },
    ]


def _lab_classroom(
    *,
    diagnose: Mapping[str, Any] | None = None,
    autopilot: Mapping[str, Any] | None = None,
    report_card: Mapping[str, Any] | None = None,
    pnl: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if diagnose is None:
        try:
            from execution.autopilot import diagnose_silence
            diagnose = diagnose_silence()
        except Exception:
            diagnose = {}
    diagnose = dict(diagnose or {})
    if autopilot is None:
        try:
            from execution.autopilot import get_status
            autopilot = get_status()
        except Exception:
            autopilot = {}
    autopilot = dict(autopilot or {})
    if report_card is None:
        try:
            from execution.autopilot import report_card as _rc
            report_card = _rc()
        except Exception:
            report_card = {}
    report_card = dict(report_card or {})
    if pnl is None:
        try:
            from execution.autopilot import pnl_snapshot
            pnl = pnl_snapshot()
        except Exception:
            pnl = {}
    pnl = dict(pnl or {})
    stats = dict(report_card.get("stats") or {})
    positions = [p for p in (pnl.get("positions") or []) if isinstance(p, Mapping)]
    closed_n = int(stats.get("n") or stats.get("paper_n") or 0)
    armed = bool(autopilot.get("armed"))
    mode = str(autopilot.get("mode") or "PAPER").upper()
    in_window = bool(diagnose.get("in_window"))
    blockers = [str(b) for b in (diagnose.get("blockers") or []) if b]
    notes = [str(n) for n in (diagnose.get("notes") or []) if n]
    activity = [str(a) for a in (diagnose.get("activity") or autopilot.get("activity") or []) if a][:8]
    rejects = dict(diagnose.get("rejects_today") or {})
    funnel = [{"reason": str(k), "n": int(v)} for k, v in sorted(rejects.items(), key=lambda kv: -int(kv[1] or 0))[:6]]
    next_action = ""
    if not armed:
        next_action = "Arm paper on this page. Default is OFF so nothing auto-trades."
    elif mode != "PAPER":
        next_action = "Mode is not PAPER — this desk will not arm live."
    elif not in_window:
        next_action = "Window is closed. Paper takes Ready tickets 09:30–15:20 IST on weekdays."
    elif blockers:
        next_action = blockers[0]
    elif int(diagnose.get("considered_today") or 0) == 0:
        next_action = "Armed, but no ticket has been fed yet. Press Feed Ready tickets."
    else:
        next_action = diagnose.get("headline") or "Waiting for a Ready ticket that clears gates."
    return {
        "armed": armed,
        "mode": mode,
        "allocation": _f(autopilot.get("allocation")) or 0.0,
        "in_window": in_window,
        "headline": diagnose.get("headline") or ("Armed" if armed else "Disarmed — default OFF"),
        "blockers": blockers,
        "notes": notes,
        "funnel": funnel,
        "considered_today": int(diagnose.get("considered_today") or 0),
        "trades_today": int(diagnose.get("trades_today") or autopilot.get("trades_today_count") or 0),
        "buy_setups": int(diagnose.get("buy_setups_in_last_scan") or 0),
        "open_n": len(positions),
        "open": [
            {
                "symbol": str(p.get("symbol") or ""),
                "qty": int(p.get("qty") or 0),
                "entry": _f(p.get("entry")),
                "live": _f(p.get("live")),
                "pnl": _f(p.get("pnl")),
                "stop": _f(p.get("stop")),
                "target": _f(p.get("target")),
            }
            for p in positions[:8]
        ],
        "closed_n": closed_n,
        "expectancy_r": _f(stats.get("expectancy_r")) if closed_n else None,
        "win_rate": _f(stats.get("win_rate")) if closed_n else None,
        "profit_factor": _f(stats.get("profit_factor")) if closed_n else None,
        "day_pnl": _f(pnl.get("day_pnl")),
        "activity": activity,
        "next_action": next_action,
        "places_orders": False,
        "live_locked": True,
    }


def _ready_rows_for_paper(scan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Ready tickets as scanner-shaped rows. Never invents volume, EV, or prices."""
    out: list[dict[str, Any]] = []
    try:
        ready = build_ready_queue(scan=scan, workspace=_WORKSPACE_UNSET)
    except Exception:
        return out
    seen: set[str] = set()
    for card in list(ready.get("prime") or []) + list(ready.get("actionable") or []) + list(ready.get("stage2") or []):
        if not isinstance(card, Mapping):
            continue
        symbol = str(card.get("symbol") or "").upper().strip()
        if not symbol or symbol in seen:
            continue
        entry = _f(card.get("entry"))
        stop = _f(card.get("stop"))
        if entry is None or stop is None or entry <= 0 or stop <= 0 or stop >= entry:
            continue
        score = _f(card.get("score")) or _f(card.get("sepa_score"))
        seen.add(symbol)
        row: dict[str, Any] = {
            "symbol": symbol,
            "verdict": "BUY",
            "status": "Ready to trade",
            "entry": entry,
            "stop": stop,
            "target": card.get("target"),
            "price": card.get("cmp") or entry,
            "score": score if score is not None else 0.0,
            "edge_r": card.get("edge_r"),
            "sector": card.get("sector") or "",
            "rsi": card.get("rsi"),
            "ev_pct": card.get("ev_pct"),
            "p_win": card.get("p_win"),
            "ev_conf": card.get("ev_conf"),
        }
        vol = _f(card.get("volume_ratio"))
        if vol is not None:
            row["volume_ratio"] = vol
        out.append(row)
    return out


def feed_paper_classroom() -> dict[str, Any]:
    """Push last scan + Ready tickets into armed PAPER autopilot.

    No-op if disarmed or not PAPER. Never invents quotes, volume, or EV.
    """
    try:
        from execution.autopilot import diagnose_silence, get_status, on_setups
    except Exception as exc:
        return {
            "ok": False,
            "fed": 0,
            "message": f"Autopilot unread: {exc}",
            "places_orders": False,
            "live_locked": True,
        }
    status = get_status()
    mode = str(status.get("mode") or "PAPER").upper()
    if mode != "PAPER":
        return {
            "ok": False,
            "armed": bool(status.get("armed")),
            "mode": mode,
            "fed": 0,
            "message": "Paper feed aborted — mode is not PAPER.",
            "places_orders": False,
            "live_locked": True,
        }
    if not status.get("armed"):
        return {
            "ok": False,
            "armed": False,
            "mode": "PAPER",
            "fed": 0,
            "message": "Arm paper first. Default is OFF so nothing auto-trades.",
            "places_orders": False,
            "live_locked": True,
        }
    scan: dict[str, Any] = {}
    try:
        from product.scan_store import load_scan
        scan = dict(load_scan() or {})
    except Exception:
        scan = {}
    rows: list[dict[str, Any]] = [
        dict(r) for r in (scan.get("records") or []) if isinstance(r, Mapping)
    ]
    overlay = _ready_rows_for_paper(scan)
    have_buy = {
        str(r.get("symbol") or "").upper()
        for r in rows
        if str(r.get("verdict") or "") in {"BUY", "STRONG BUY"}
        or str(r.get("status") or "") == "Ready to trade"
    }
    for row in overlay:
        if row["symbol"] not in have_buy:
            rows.append(row)
            have_buy.add(row["symbol"])
    buy_n = len(have_buy)
    if rows:
        on_setups(rows)
    diagnose = diagnose_silence()
    after = get_status()
    taken = int(after.get("trades_today_count") or 0)
    if not rows:
        message = "No scan or Ready ticket on file. Fill the desk first."
    elif not diagnose.get("in_window"):
        message = (
            f"Fed {buy_n} Ready/BUY name(s). Window is closed — "
            "paper takes them 09:30–15:20 IST on weekdays."
        )
    elif taken:
        message = (
            f"Fed {buy_n} Ready/BUY name(s). Paper book today: {taken} trade(s). "
            "Live stays locked."
        )
    else:
        headline = diagnose.get("headline") or "waiting on gates"
        message = f"Fed {buy_n} Ready/BUY name(s). No fill yet — {headline}"
    return {
        "ok": True,
        "armed": True,
        "mode": "PAPER",
        "fed": buy_n,
        "trades_today": taken,
        "open_trades": len(after.get("open_trades") or []),
        "headline": diagnose.get("headline") or "",
        "blockers": list(diagnose.get("blockers") or []),
        "message": message,
        "places_orders": False,
        "live_locked": True,
    }


def build_backtest_lab(
    *,
    report: Mapping[str, Any] | None = None,
    status: Mapping[str, Any] | None = None,
    playbook: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Backtest as four use cases a user can actually follow."""
    if report is None:
        try:
            from scan.signal_backtest import load_report
            report = load_report() or {}
        except Exception:
            report = {}
    report = dict(report or {})
    # report_is_actionable({}) currently falls through to disk. An explicit
    # empty mapping must stay empty (tests + honest Lab).
    if "signals" not in report:
        report = {**report, "signals": {}}
    if status is None:
        try:
            from product.full_universe_backtest import backtest_status
            status = backtest_status()
        except Exception:
            status = {}
    status = dict(status or {})
    if playbook is None:
        try:
            from scan.signal_backtest import trading_playbook
            playbook = trading_playbook() or {}
        except Exception:
            playbook = {}
    playbook = dict(playbook or {})

    actionable = False
    evidence_note = "no measured backtest yet"
    try:
        from scan.signal_backtest import report_is_actionable, universe_evidence_note
        actionable = report_is_actionable(report)
        evidence_note = universe_evidence_note(report)
    except Exception:
        pass

    signals = _signal_rows(report)
    proven = [s for s in signals if s["verdict"] in {"PROVEN", "POSITIVE"}]
    losers = [s for s in signals if s["verdict"] == "LOSER"]
    running = bool(status.get("running"))

    use_cases = [
        {
            "id": "trust_scanner",
            "title": "Should I trust today's scanner?",
            "kid_title": "Did we practice enough?",
            "when": "Before you treat any Ready name as money.",
            "how": (
                "Run Backtest all stocks on official NSE history. "
                "A report is only actionable at ≥100 symbols and a non-truncated sample."
            ),
            "status": (
                "RUNNING" if running else
                "READY" if actionable else
                "MISSING" if not report else
                "PARTIAL"
            ),
            "result": evidence_note,
            "control": "RUN_FULL_UNIVERSE_BACKTEST_NOW",
        },
        {
            "id": "regime_lean",
            "title": "Which signals earn in this tape?",
            "kid_title": "Who is good on today's pitch?",
            "when": "When two Ready names compete and you need the measured one.",
            "how": "The playbook uses today's regime bucket, not a vibe ranking.",
            "status": "READY" if playbook.get("best") else ("MISSING" if not report else "THIN"),
            "result": playbook.get("regime") or "regime unknown",
            "best": list(playbook.get("best") or [])[:5],
        },
        {
            "id": "avoid_losers",
            "title": "Which signals should I skip?",
            "kid_title": "Who failed the practice test?",
            "when": "Whenever a card still looks exciting but the combo is a proven loser.",
            "how": "Loser verdicts auto-demote the next scan. Do not override them by hand.",
            "status": "READY" if losers or playbook.get("avoid") else ("MISSING" if not report else "NONE"),
            "result": f"{len(losers)} loser signal(s)",
            "avoid": list(playbook.get("avoid") or [s["signal"] for s in losers])[:12],
        },
        {
            "id": "paper_loop",
            "title": "Did paper autopilot earn the next rupee?",
            "kid_title": "Now try with fake money",
            "when": "Only after Lab is actionable and Ready has (or honestly has not) a name.",
            "how": "Journey reads autopilot's own closed paper trades — not a simulated backtest P&L.",
            "status": "OPEN",
            "result": "Open Trade → Journey. Live is earned there, not here.",
            "goto": "Live Journey",
        },
    ]

    has_report = bool(report.get("signals") or report.get("generated_at") or status.get("has_report"))
    live = _live_edge_snapshot()
    learning = _lab_learning(playbook=playbook, signals=signals, live=live)
    classroom = _lab_classroom()
    lesson = _lab_lesson(running=running, actionable=actionable, has_report=has_report)
    if running:
        current = str(status.get("current") or "")
        done = int(status.get("progress") or 0)
        total = int(status.get("total") or 0)
        lesson["now"] = (
            f"Testing {current or 'the next stock'} · {done}/{total or '—'}. "
            "Official NSE history only. No order is placed."
        )
    elif classroom.get("armed"):
        lesson["now"] = (
            (lesson.get("now") or "")
            + " Paper classroom is armed — Ready tickets are practiced with fake money."
        ).strip()
    pulse = _lab_pulse(running=running, actionable=actionable, classroom=classroom)
    loop = _lab_loop(
        running=running, actionable=actionable, evidence_note=evidence_note,
        learning=learning, classroom=classroom,
    )
    for item in use_cases:
        if item.get("id") != "paper_loop":
            continue
        if classroom.get("open_n"):
            item["status"] = "LIVE"
            item["result"] = f"{classroom['open_n']} open paper trade(s). Live locked."
        elif classroom.get("armed"):
            item["status"] = "ARMED"
            item["result"] = classroom.get("headline") or "Armed in PAPER"
        elif classroom.get("closed_n"):
            item["status"] = "READY"
            item["result"] = f"{classroom['closed_n']} closed paper trade(s) on the report card."
        else:
            item["status"] = "WAIT"
            item["result"] = classroom.get("next_action") or "Arm paper. Default is OFF."
    return {
        "schema_version": 3,
        "places_orders": False,
        "live_locked": True,
        "running": running,
        "progress": status.get("progress") or 0,
        "total": status.get("total") or 0,
        "current": status.get("current") or "",
        "actionable": actionable,
        "evidence_note": evidence_note,
        "generated_at": report.get("generated_at") or status.get("generated_at") or "",
        "universe": dict(report.get("universe") or status.get("universe") or {}),
        "playbook": {
            "regime": playbook.get("regime"),
            "best": list(playbook.get("best") or [])[:5],
            "avoid": list(playbook.get("avoid") or [])[:12],
            "recommended_target_pct": playbook.get("recommended_target_pct"),
        },
        "signals": signals[:24],
        "proven_n": len(proven),
        "loser_n": len(losers),
        "scoreboard": _lab_scoreboard(signals),
        "lesson": lesson,
        "pulse": pulse,
        "loop": loop,
        "learning": learning,
        "classroom": classroom,
        "use_cases": use_cases,
        "disclaimer": (
            "The practice test never places paper or live orders. "
            "Paper classroom uses fake money on Ready tickets. "
            "<30 trades on a signal = no claim. Truncated samples stay partial. "
            "Learning is demote-only — losers drop on the next scan, winners are never inflated. "
            "Live stays locked."
        ),
    }


def _step(sid: str, title: str, detail: str, status: str, *, next_action: str = "") -> dict[str, Any]:
    return {
        "id": sid,
        "title": title,
        "detail": detail,
        "status": status,  # PASS | WAIT | BLOCK | LOCKED
        "next_action": next_action,
    }


def build_live_journey(
    *,
    data: Mapping[str, Any] | None = None,
    scan: Mapping[str, Any] | None = None,
    paper: Mapping[str, Any] | None = None,
    autonomy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    paper_closed: int | None = None,
    ladder: Mapping[str, Any] | None = None,
    autopilot: Mapping[str, Any] | None = None,
    report_card: Mapping[str, Any] | None = None,
    diagnose: Mapping[str, Any] | None = None,
    lab: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single checklist from observe → paper → live. Never arms LIVE."""
    if ladder is None:
        ladder = build_production_ladder(
            data=data,
            scan=scan,
            paper=paper,
            autonomy=autonomy,
            env=env,
            paper_closed=paper_closed,
        )
    ladder = dict(ladder)
    n = int(ladder.get("paper_closed") or 0)
    e4 = int(ladder.get("paper_e4_n") or PAPER_E4_N)
    alpha = dict(ladder.get("alpha") or {})
    alpha_n = int(alpha.get("level") or 0)
    live_ok = bool(ladder.get("live_unlocked"))
    rung = dict(ladder.get("rung") or {})

    if autopilot is None:
        try:
            from execution.autopilot import get_status
            autopilot = get_status()
        except Exception:
            autopilot = {}
    autopilot = dict(autopilot or {})

    if report_card is None:
        try:
            from execution.autopilot import report_card as _rc
            report_card = _rc()
        except Exception:
            report_card = {}
    report_card = dict(report_card or {})
    stats = dict(report_card.get("stats") or {})
    verdict = str(report_card.get("verdict") or "COLLECTING_EVIDENCE")

    if diagnose is None:
        try:
            from execution.autopilot import diagnose_silence
            diagnose = diagnose_silence()
        except Exception:
            diagnose = {}
    diagnose = dict(diagnose or {})

    if lab is None:
        lab = build_backtest_lab()
    lab = dict(lab)

    scaling = None
    try:
        from core.sim_lab import scaling_advice
        pf = float(stats.get("profit_factor") or 0)
        dd = float(stats.get("max_drawdown") or 0)
        capital = float(autopilot.get("allocation") or 0) or 1.0
        dd_pct = (dd / capital * 100.0) if capital else 0.0
        scaling = scaling_advice(pf, dd_pct, int(stats.get("n") or 0))
    except Exception:
        scaling = None

    tape_ok = str((next((s for s in ladder.get("subsystems") or [] if s.get("id") == "tape"), {}) or {}).get("status") or "") == "READY"
    scan_ok = bool((scan or {}).get("available") or (scan or {}).get("records") or (scan or {}).get("scanned_at"))
    armed = bool(autopilot.get("armed"))
    mode = str(autopilot.get("mode") or "PAPER").upper()
    allocation = _f(autopilot.get("allocation")) or 0.0
    paper_mode = mode == "PAPER"

    ok_live, live_why = live_arm_allowed(
        env=env,
        paper_closed=n,
        alpha_level=alpha_n,
        institutional_live_allowed=live_ok,
    )

    steps = [
        _step(
            "tape",
            "Official tape is fresh",
            "NSE bhavcopy ≥ 60 sessions, not stale.",
            "PASS" if tape_ok else "BLOCK",
            next_action="" if tape_ok else "Home → Fill today's desk / System → Prepare market data",
        ),
        _step(
            "scan",
            "Whole-market scan on file",
            "Ready names come from the last scan, never a hand-picked list.",
            "PASS" if scan_ok else "WAIT",
            next_action="" if scan_ok else "Run a market scan",
        ),
        _step(
            "lab",
            "Walk-forward backtest is actionable",
            lab.get("evidence_note") or "no measured backtest yet",
            "PASS" if lab.get("actionable") else ("WAIT" if lab.get("running") else "WAIT"),
            next_action="" if lab.get("actionable") else "Trade → Lab → Backtest all stocks",
        ),
        _step(
            "paper_alloc",
            "Paper allocation ≥ ₹5,000",
            f"Current paper pool ₹{allocation:,.0f}. Autopilot will not arm below ₹5,000.",
            "PASS" if allocation >= 5000 else "WAIT",
            next_action="" if allocation >= 5000 else "Set paper allocation on this page, then arm paper.",
        ),
        _step(
            "paper_arm",
            "Autopilot armed in PAPER",
            diagnose.get("headline") or ("Armed" if armed and paper_mode else "Disarmed — default OFF"),
            "PASS" if armed and paper_mode else "WAIT",
            next_action="" if (armed and paper_mode) else "Arm paper on this page. This button cannot go live.",
        ),
        _step(
            "paper_30",
            f"Forward paper sample {n}/{PAPER_TRANSITION_N}",
            "Thirty closed paper trades is the first honesty gate — still not live.",
            "PASS" if n >= PAPER_TRANSITION_N else "WAIT",
        ),
        _step(
            "report_card",
            "Autopilot report card",
            str(report_card.get("verdict_reason") or "No closed autopilot trades yet."),
            (
                "PASS" if verdict == "READY_CANDIDATE" else
                "BLOCK" if verdict == "NOT_READY" else
                "WAIT"
            ),
        ),
        _step(
            "paper_300",
            f"E4 paper sample {n}/{e4}",
            "Live needs ≥300 closed paper trades. Thirty is only TRANSITION.",
            "PASS" if n >= e4 else "WAIT",
        ),
        _step(
            "alpha",
            f"Strategy alpha {alpha.get('label') or 'E0'}",
            "promote() only — belief does not raise the level.",
            "PASS" if alpha_n >= 4 else "WAIT",
        ),
        _step(
            "live_lock",
            "Live money",
            live_why if not ok_live else "Every ladder gate cleared — still type ARM LIVE outside this desk.",
            "LOCKED" if not ok_live else "LOCKED",
            next_action="This desk never sends ARM LIVE. Telegram taps stay paper-only.",
        ),
    ]
    # Last step stays LOCKED even when gates clear: live arm is not exposed here.
    if ok_live:
        steps[-1]["status"] = "LOCKED"
        steps[-1]["detail"] = (
            "Gates are clear, but live still needs the typed phrase ARM LIVE "
            "on the Streamlit/CLI autopilot — not this browser desk."
        )

    classroom = _lab_classroom(
        diagnose=diagnose, autopilot=autopilot, report_card=report_card,
    )
    return {
        "schema_version": 2,
        "places_orders": False,
        "live_locked": True,
        "rung": rung,
        "paper_closed": n,
        "paper_e4_n": e4,
        "alpha": alpha,
        "live_unlocked": False,
        "ladder_live_unlocked": live_ok,
        "live_blockers": list(ladder.get("live_blockers") or []),
        "autopilot": {
            "armed": armed,
            "mode": mode,
            "allocation": allocation,
            "trades_today": autopilot.get("trades_today_count") or autopilot.get("trades_today"),
            "open_trades": len(autopilot.get("open_trades") or []) or classroom.get("open_n") or 0,
            "headline": diagnose.get("headline") or "",
            "blockers": list(diagnose.get("blockers") or []),
            "in_window": bool(diagnose.get("in_window")),
            "considered_today": int(diagnose.get("considered_today") or 0),
            "buy_setups": int(diagnose.get("buy_setups_in_last_scan") or 0),
        },
        "classroom": classroom,
        "report_card": {
            "verdict": verdict,
            "verdict_reason": report_card.get("verdict_reason") or "",
            "stats": {
                "n": stats.get("n") or 0,
                "win_rate": stats.get("win_rate"),
                "expectancy_r": stats.get("expectancy_r"),
                "profit_factor": stats.get("profit_factor"),
                "total_pnl": stats.get("total_pnl"),
                "paper_n": stats.get("paper_n"),
            },
        },
        "scaling": scaling,
        "steps": steps,
        "next_action": classroom.get("next_action") or "",
        "disclaimer": (
            "Paper is the production path. Live is earned by closed paper sample, "
            "E4 alpha, institutional certification, and a typed ARM LIVE — "
            "never by a green toggle on this page."
        ),
    }


def arm_paper_only(*, allocation: float | None = None) -> dict[str, Any]:
    """Force PAPER mode and arm. Refuses any live path."""
    from execution.autopilot import arm, disarm, get_status, set_config

    if allocation is not None:
        try:
            alloc = float(allocation)
        except (TypeError, ValueError):
            alloc = 0.0
        if alloc < 0:
            alloc = 0.0
        set_config(mode="PAPER", allocation=alloc)
    else:
        set_config(mode="PAPER")
    status = get_status()
    if str(status.get("mode") or "").upper() != "PAPER":
        disarm("trade-desk refused non-paper mode")
        return {
            "ok": False,
            "armed": False,
            "mode": "PAPER",
            "message": "Paper arm aborted — mode was not PAPER.",
            "places_orders": False,
            "live_locked": True,
        }
    ok, message = arm("")
    after = get_status()
    if str(after.get("mode") or "").upper() == "LIVE":
        disarm("trade-desk live abort")
        return {
            "ok": False,
            "armed": False,
            "mode": "PAPER",
            "message": "Paper arm aborted — LIVE is not allowed from this desk.",
            "places_orders": False,
            "live_locked": True,
        }
    feed = {"fed": 0, "message": ""}
    if after.get("armed"):
        try:
            feed = feed_paper_classroom()
        except Exception:
            feed = {"fed": 0, "message": "Armed, but last-scan feed failed — press Feed Ready tickets."}
        extra = str(feed.get("message") or "")
        if extra and extra not in message:
            message = f"{message}. {extra}"
    return {
        "ok": bool(ok),
        "armed": bool(after.get("armed")),
        "mode": "PAPER",
        "message": message,
        "allocation": after.get("allocation"),
        "fed": int(feed.get("fed") or 0),
        "trades_today": feed.get("trades_today"),
        "places_orders": False,
        "live_locked": True,
    }

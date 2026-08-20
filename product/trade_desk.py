"""Trade desk: ready queue, backtest lab, paper→live journey.

Research (Ideas) stays a monitor. This module is the production path the
user asked for:

  Ready  — names that clear money gates (Prime = high-evidence only)
  Lab    — walk-forward backtest as use cases, not a dump
  Journey — paper autopilot → live, earned, never toggled

Never invents EV, never arms LIVE, never hard-wires a ticker.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.production_ladder import (
    PAPER_E4_N,
    PAPER_TRANSITION_N,
    build_production_ladder,
    live_arm_allowed,
)

READY_EDGE_FLOOR = 0.05  # R; below this is thin, not a money claim


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _upside(entry: float | None, target: float | None) -> float | None:
    if entry is None or target is None or entry <= 0:
        return None
    return round((target / entry - 1.0) * 100.0, 1)


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


def _actionable(row: Mapping[str, Any]) -> tuple[bool, str]:
    verdict = str(row.get("verdict") or "").upper()
    status = str(row.get("status") or "")
    if verdict not in {"BUY", "STRONG BUY"} and status != "Ready to trade":
        return False, "Scanner did not mark BUY / Ready to trade"
    if row.get("chase_risk") or row.get("extended"):
        return False, "Chase / extension — late entry"
    if not _has_plan(row):
        return False, "No entry/stop — no risk plan"
    if _loser_edge(row):
        return False, "Measured edge or conservative EV is not positive"
    cats = set(row.get("categories") or [])
    if cats and not (cats & {"Momentum", "Breakout"}):
        return False, "Not a momentum or breakout ticket"
    return True, "Complete ticket; not a return promise"


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
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "lane": lane,
        "verdict": str(row.get("verdict") or ""),
        "status": str(row.get("status") or ""),
        "sector": str(row.get("sector") or "—"),
        "score": _f(row.get("score")),
        "edge_r": _f(row.get("edge_r")),
        "entry": entry,
        "stop": stop,
        "target": target,
        "cmp": _f(row.get("price") or row.get("cmp")),
        "upside_from_buy_pct": _upside(entry, target),
        "why": [str(x) for x in why if x],
        "honesty": (
            "Prime means every evidence gate passed — not a guarantee. "
            "Paper first. Live stays locked until the journey earns it."
            if lane == "prime"
            else "Ticket is complete. Measured sample is thin or below Prime — not a high-chance claim."
        ),
        **ev,
    }


def build_ready_queue(
    *,
    scan: Mapping[str, Any] | None = None,
    market: Mapping[str, Any] | None = None,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evidence-gated ready list. Empty is a valid, honest answer."""
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
        if is_prime and _has_plan(row) and not _loser_edge(row) and not row.get("chase_risk") and not row.get("extended"):
            prime.append(_card(row, lane="prime", why=prime_why))
            continue
        if ok_plan:
            actionable.append(_card(row, lane="actionable", why=[plan_why]))
            continue
        if str(row.get("verdict") or "").upper() in {"BUY", "STRONG BUY"} or str(row.get("status") or "") == "Ready to trade":
            rejected.append({"symbol": symbol, "reason": fail or plan_why})

    def _rank(card: Mapping[str, Any]) -> tuple:
        has_ev = 1 if card.get("ev_pct") is not None and int(card.get("ev_n") or 0) >= 30 else 0
        lb = _f(card.get("ev_lb_pct"))
        edge = _f(card.get("edge_r")) or 0.0
        score = _f(card.get("score")) or 0.0
        return (has_ev, lb if lb is not None else -999.0, edge, score)

    prime.sort(key=_rank, reverse=True)
    actionable.sort(key=_rank, reverse=True)

    empty_why: list[str] = []
    if not rows:
        empty_why.append("No scan is on file. Run a market scan first.")
    elif not prime and not actionable:
        empty_why.append(
            "No name clears the money gates today — BUY + stop + non-negative measured edge."
        )
        if breadth == "NARROW":
            empty_why.append("Breadth is NARROW — Prime is vetoed.")
        if rejected:
            empty_why.append(f"{len(rejected)} BUY name(s) failed a gate (chase, missing stop, or loser edge).")

    return {
        "schema_version": 1,
        "places_orders": False,
        "live_locked": True,
        "scanned_at": scan.get("scanned_at") or "",
        "universe_size": int(scan.get("universe_size") or len(rows)),
        "breadth": breadth or "—",
        "prime": prime[:12],
        "actionable": actionable[:12],
        "rejected_n": len(rejected),
        "rejected_sample": rejected[:8],
        "empty": not prime and not actionable,
        "empty_why": empty_why,
        "disclaimer": (
            "Ready is not a broker order and not a return promise. "
            "Prime uses the same evidence gates as Telegram 💎 (verdict, type, "
            "conviction, conservative EV, liquidity, breadth, regime). "
            "Missing numbers stay missing."
        ),
        "next": (
            "Open Lab to see which signals earned in this tape, then Journey to paper-trade them."
            if prime or actionable
            else "Fill the desk (scan + backtest) before expecting a Ready name."
        ),
    }


def _signal_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, raw in dict(report.get("signals") or {}).items():
        if not isinstance(raw, Mapping):
            continue
        out.append({
            "signal": str(key),
            "trades": int(raw.get("trades") or 0),
            "closed": int(raw.get("closed") or raw.get("trades") or 0),
            "win_rate": _f(raw.get("win_rate")),
            "expectancy_r": _f(raw.get("expectancy_r")),
            "verdict": str(raw.get("verdict") or "THIN"),
        })
    out.sort(key=lambda r: (r["verdict"] != "PROVEN", -(r["expectancy_r"] or -99.0)))
    return out


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
            "when": "When two Ready names compete and you need the measured one.",
            "how": "The playbook uses today's regime bucket, not a vibe ranking.",
            "status": "READY" if playbook.get("best") else ("MISSING" if not report else "THIN"),
            "result": playbook.get("regime") or "regime unknown",
            "best": list(playbook.get("best") or [])[:5],
        },
        {
            "id": "avoid_losers",
            "title": "Which signals should I skip?",
            "when": "Whenever a card still looks exciting but the combo is a proven loser.",
            "how": "Loser verdicts auto-demote the next scan. Do not override them by hand.",
            "status": "READY" if losers or playbook.get("avoid") else ("MISSING" if not report else "NONE"),
            "result": f"{len(losers)} loser signal(s)",
            "avoid": list(playbook.get("avoid") or [s["signal"] for s in losers])[:12],
        },
        {
            "id": "paper_loop",
            "title": "Did paper autopilot earn the next rupee?",
            "when": "Only after Lab is actionable and Ready has (or honestly has not) a name.",
            "how": "Journey reads autopilot's own closed paper trades — not a simulated backtest P&L.",
            "status": "OPEN",
            "result": "Open Trade → Journey. Live is earned there, not here.",
            "goto": "Live Journey",
        },
    ]

    return {
        "schema_version": 1,
        "places_orders": False,
        "live_locked": True,
        "running": running,
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
        "use_cases": use_cases,
        "disclaimer": (
            "This backtest never places paper or live orders. "
            "<30 trades on a signal = no claim. Truncated samples stay partial."
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

    return {
        "schema_version": 1,
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
            "open_trades": len(autopilot.get("open_trades") or []),
            "headline": diagnose.get("headline") or "",
            "blockers": list(diagnose.get("blockers") or []),
        },
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
    return {
        "ok": bool(ok),
        "armed": bool(after.get("armed")),
        "mode": "PAPER",
        "message": message,
        "allocation": after.get("allocation"),
        "places_orders": False,
        "live_locked": True,
    }

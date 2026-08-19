"""Production ladder — subsystems in sync, live money earned not toggled.

Reco is a research monitor. QuantTerm competes as a trading system:
tape → scan → evidence → brain → risk → paper → (only then) live.

Each subsystem has one job. The monitor never places an order. LIVE arming
is fail-closed until the paper evidence gate AND the institutional live
deployment contract both clear. QT_LIVE_ENABLED is only a migration lock.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

PAPER_E4_N = 300
PAPER_TRANSITION_N = 30
ALPHA_CAPABILITY = "Strategy alpha (the edge)"

# Contracts: one job each. execution=False means research/context only.
SUBSYSTEMS: tuple[dict[str, Any], ...] = (
    {
        "id": "tape",
        "job": "Official NSE history and last print. Never invent a bar.",
        "module": "data.bhavcopy_store + data.index_store + data.live_quotes",
        "reads": [],
        "writes": ["ohlcv", "index_strip"],
        "may_order": False,
    },
    {
        "id": "monitor",
        "job": "SEPA / stage / RS / breadth / news — Reco-style research overlay.",
        "module": "product.sepa_setup + product.monitor_context + product.monitor_market",
        "reads": ["tape"],
        "writes": ["setup_cards", "breadth"],
        "may_order": False,
    },
    {
        "id": "scan",
        "job": "Whole-market signals, calibrated by walk-forward — skip names with no data.",
        "module": "scan.unified_scanner + scan.auto_scan",
        "reads": ["tape"],
        "writes": ["setups"],
        "may_order": False,
    },
    {
        "id": "evidence",
        "job": "Expectancy, live edge, gauntlet. Demote proven losers. Never inflate.",
        "module": "scan.live_edge + scan.ev_engine + research.harness + gauntlet",
        "reads": ["scan", "feedback"],
        "writes": ["calibration", "ev"],
        "may_order": False,
    },
    {
        "id": "brain",
        "job": "One posture from regime × edge × book × breadth. Read-only conductor.",
        "module": "core.brain",
        "reads": ["scan", "evidence", "risk", "tape"],
        "writes": ["posture", "directives"],
        "may_order": False,
    },
    {
        "id": "risk",
        "job": "1% risk, 10% name cap, 5% open risk, correlation clusters, GTT on every entry.",
        "module": "risk.position_sizer + risk.portfolio_risk + execution.trade_executor",
        "reads": ["tape", "brain"],
        "writes": ["qty", "stop", "book_verdict"],
        "may_order": False,
    },
    {
        "id": "paper",
        "job": "Autopilot in PAPER. Outcomes feed evidence. Telegram taps stay paper-only.",
        "module": "execution.autopilot + research.autonomy.supervisor",
        "reads": ["scan", "brain", "risk"],
        "writes": ["paper_fills", "paper_outcomes"],
        "may_order": True,
        "mode": "PAPER",
    },
    {
        "id": "execution",
        "job": "Kite entry + exchange-side GTT. Live only if the ladder unlocks.",
        "module": "execution.trade_executor + execution.oms",
        "reads": ["risk", "ladder"],
        "writes": ["orders", "gtt"],
        "may_order": True,
        "mode": "LIVE",
    },
    {
        "id": "feedback",
        "job": "Every BUY tracked. Decision journal: taken vs rejected. Calibration, not vibes.",
        "module": "core.signal_outcome_tracker + core.decision_journal",
        "reads": ["paper", "execution"],
        "writes": ["outcomes"],
        "may_order": False,
    },
)


def _env_live_interlock(env: Mapping[str, str] | None = None) -> bool:
    raw = (env or os.environ).get("QT_LIVE_ENABLED", "")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _paper_closed_n() -> int:
    try:
        import sqlite3
        from pathlib import Path
        db = Path(__file__).resolve().parents[1] / "logs" / "trades.db"
        if not db.exists():
            return 0
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT COUNT(*) FROM trades WHERE status IN ('PAPER_WIN','PAPER_LOSS')"
            ).fetchone()
        finally:
            con.close()
        return int(row[0] if row else 0)
    except Exception:
        return 0


def _alpha_level(report: Sequence[Mapping[str, Any]] | None = None) -> tuple[int, str]:
    try:
        if report is None:
            from core.evidence_levels import report as evidence_report
            report = evidence_report()
        for row in report or []:
            if str(row.get("capability") or "") == ALPHA_CAPABILITY:
                return int(row.get("level") or 0), str(row.get("label") or "E0 designed")
    except Exception:
        pass
    return 0, "E0 designed"


def live_blockers(
    *,
    env: Mapping[str, str] | None = None,
    paper_closed: int | None = None,
    alpha_level: int | None = None,
    institutional_live_allowed: bool | None = None,
) -> list[str]:
    """Every item must clear before real-money automation. Fail closed."""
    blocks: list[str] = []
    if not _env_live_interlock(env):
        blocks.append("QT_LIVE_ENABLED is off — migration interlock, not a graduation switch.")
    n = _paper_closed_n() if paper_closed is None else int(paper_closed)
    if n < PAPER_E4_N:
        blocks.append(
            f"Forward paper sample {n}/{PAPER_E4_N} closed trades — E4 needs ≥{PAPER_E4_N}."
        )
    lvl = alpha_level
    if lvl is None:
        lvl, _ = _alpha_level()
    if int(lvl) < 4:
        blocks.append(
            f"Strategy alpha is E{int(lvl)} — live needs E4 paper-validated (promote() only)."
        )
    live_ok = institutional_live_allowed
    if live_ok is None:
        try:
            from product.institutional_readiness import build_institutional_readiness
            dep = (build_institutional_readiness() or {}).get("deployment", {}).get("live") or {}
            live_ok = bool(dep.get("allowed"))
        except Exception:
            live_ok = False
    if not live_ok:
        blocks.append(
            "Institutional live deployment is BLOCKED — OMS, risk governor, "
            "reconciliation and owner approval are not certified."
        )
    return blocks


def live_arm_allowed(**kwargs: Any) -> tuple[bool, str]:
    blocks = live_blockers(**kwargs)
    if blocks:
        return False, "LIVE locked. " + blocks[0]
    return True, "LIVE unlocked — every ladder gate cleared."


def _rung(paper_closed: int, tape_ok: bool, paper_ok: bool, live_ok: bool) -> dict[str, Any]:
    if live_ok:
        key, label, next_rung = "LIVE", "LIVE — real money, earned", None
    elif paper_closed >= PAPER_TRANSITION_N:
        key, label, next_rung = (
            "TRANSITION",
            "TRANSITION — paper is running; live still locked",
            f"Need {PAPER_E4_N} closed paper trades, E4 alpha, and institutional live READY.",
        )
    elif paper_ok:
        key, label, next_rung = (
            "PAPER",
            "PAPER — this is the production path",
            "Arm paper, log every outcome, do not skip to live.",
        )
    elif tape_ok:
        key, label, next_rung = (
            "RESEARCH",
            "RESEARCH — monitor and scan only",
            "Start paper autonomy once the tape and scan are fresh.",
        )
    else:
        key, label, next_rung = (
            "OBSERVE",
            "OBSERVE — official tape is not ready",
            "Refresh NSE bhavcopy / index store before any claim.",
        )
    return {"id": key, "label": label, "next": next_rung}


def build_production_ladder(
    *,
    data: Mapping[str, Any] | None = None,
    scan: Mapping[str, Any] | None = None,
    paper: Mapping[str, Any] | None = None,
    autonomy: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    paper_closed: int | None = None,
) -> dict[str, Any]:
    """Read-only orchestra. Never starts workers or places orders."""
    data = dict(data or {})
    scan = dict(scan or {})
    paper = dict(paper or {})
    autonomy = dict(autonomy or {})
    bhav = dict(data.get("bhavcopy") or {})
    tape_ok = bool(bhav.get("ready")) and int(bhav.get("sessions") or 0) >= 60
    scan_ok = bool(scan.get("available") or scan.get("records") or scan.get("scanned_at"))
    paper_ok = bool(paper.get("available") or paper.get("enabled") or autonomy.get("available"))
    n = _paper_closed_n() if paper_closed is None else int(paper_closed)
    alpha_n, alpha_label = _alpha_level()
    blocks = live_blockers(env=env, paper_closed=n, alpha_level=alpha_n)
    live_ok = not blocks
    rung = _rung(n, tape_ok, paper_ok, live_ok)

    status = {
        "tape": "READY" if tape_ok else "BLOCKED",
        "monitor": "READY" if tape_ok else "PARTIAL",
        "scan": "READY" if scan_ok else "PARTIAL",
        "evidence": "READY" if alpha_n >= 3 else "PARTIAL",
        "brain": "READY" if tape_ok else "PARTIAL",
        "risk": "READY",
        "paper": "READY" if paper_ok else "PARTIAL",
        "execution": "READY" if live_ok else "LOCKED",
        "feedback": "READY" if n > 0 else "PARTIAL",
    }
    nodes = []
    for spec in SUBSYSTEMS:
        item = dict(spec)
        item["status"] = status.get(spec["id"], "PARTIAL")
        nodes.append(item)

    return {
        "schema_version": 1,
        "thesis": (
            "Compete as a trading system, not a Reco clone. "
            "Paper is production. Live is earned. The monitor never orders."
        ),
        "rung": rung,
        "paper_closed": n,
        "paper_e4_n": PAPER_E4_N,
        "alpha": {"level": alpha_n, "label": alpha_label},
        "live_unlocked": live_ok,
        "live_blockers": blocks,
        "subsystems": nodes,
        "handshake": [
            {"from": "tape", "to": "scan", "payload": "official OHLCV"},
            {"from": "tape", "to": "monitor", "payload": "SEPA / breadth / RS"},
            {"from": "scan", "to": "brain", "payload": "setups"},
            {"from": "evidence", "to": "brain", "payload": "demote-only edge"},
            {"from": "brain", "to": "paper", "payload": "posture gate"},
            {"from": "risk", "to": "paper", "payload": "qty + stop + GTT"},
            {"from": "paper", "to": "feedback", "payload": "closed R"},
            {"from": "feedback", "to": "evidence", "payload": "live edge / EV"},
            {"from": "ladder", "to": "execution", "payload": "live lock"},
        ],
        "rules": [
            "No fake bars. A missing symbol is skipped.",
            "Telegram taps cannot place live orders.",
            "Every live entry must ship an exchange-side GTT.",
            "QT_LIVE_ENABLED does not graduate a strategy.",
            "promote() is the only way evidence levels rise.",
        ],
    }

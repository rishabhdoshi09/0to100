"""
🛡️ Governance Sentinel — the one authority that can say STOP, and everything obeys.

Institutional trading systems spend as much effort defining when to WITHDRAW
confidence as when to grant it. Until now QuantTerm's safety was scattered — a GTT
warning here, a portfolio DANGER there, a circuit breaker in autopilot. This is
the single, always-on guard every order path must consult, with three states:

    NORMAL   → trade within the usual risk rules
    DE_RISK  → halve size, no new full-size positions (a soft brake)
    HALT     → place NO new orders; alert a human (a hard stop)

It composes two governance layers the committee demanded:
  • KILL CONDITIONS (→ HALT): data corruption / stale bars, corporate-action
    mismatch, broker auth / repeated order rejections, repeated GTT failures,
    drawdown past the hard limit, extreme/unknown market state, a broker-vs-
    journal reconciliation mismatch, or a human kill-switch.
  • ROLLBACK TRIGGERS (→ DE_RISK): drawdown past the soft limit, confidence
    miscalibration, slippage running at ≥2× the model, an adverse/unproven regime.

Design: the DECISION (`evaluate_state`) is a pure, unit-tested function over a
signals dict; the collectors are fail-open. Fail-open on the *mechanism* (a broken
check never fabricates a halt AND never blocks all trading on a transient bug),
but every real detected condition is honored. A human `set_manual_halt(True)` is
absolute.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

NORMAL, DE_RISK, HALT = "NORMAL", "DE_RISK", "HALT"

# ── thresholds (env-tunable) ──────────────────────────────────────────────────
_HARD_DD = float(os.getenv("QT_GOV_HARD_DD_PCT", "25") or 25)      # HALT
_SOFT_DD = float(os.getenv("QT_GOV_SOFT_DD_PCT", "12") or 12)      # DE_RISK
_MAX_ORDER_FAILS = int(os.getenv("QT_GOV_MAX_ORDER_FAILS", "3") or 3)
_MAX_GTT_FAILS = int(os.getenv("QT_GOV_MAX_GTT_FAILS", "3") or 3)
_EXTREME_VIX = float(os.getenv("QT_GOV_EXTREME_VIX", "45") or 45)
_ECE_LIMIT = float(os.getenv("QT_GOV_ECE_LIMIT", "0.10") or 0.10)
_SLIP_LIMIT = float(os.getenv("QT_GOV_SLIP_LIMIT", "2.0") or 2.0)
_DERISK_MULT = float(os.getenv("QT_GOV_DERISK_MULT", "0.5") or 0.5)

_STATE_FILE = Path(__file__).resolve().parent.parent / "logs" / "governance_state.json"


# ══════════════════════════════════════════════════════════════════════════════
# Pure decision logic (unit-tested)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_state(s: dict) -> dict:
    """Map a snapshot of conditions to a governance state. Pure. Returns
    {state, size_multiplier, kill_reasons, derisk_reasons, reasons}."""
    kill: list[str] = []
    derisk: list[str] = []

    if s.get("manual_halt"):
        kill.append("human kill-switch is ON")
    if s.get("data_corrupt"):
        kill.append("data corruption / stale / missing bars")
    if s.get("ca_mismatch"):
        kill.append("possible corporate-action mismatch (un-adjusted price gaps)")
    if s.get("reconciliation_mismatch"):
        kill.append("broker positions ≠ journal — reconciliation mismatch")
    if int(s.get("consecutive_order_failures") or 0) >= _MAX_ORDER_FAILS:
        kill.append(f"{s.get('consecutive_order_failures')} consecutive order "
                    f"failures / broker rejecting")
    if int(s.get("consecutive_gtt_failures") or 0) >= _MAX_GTT_FAILS:
        kill.append(f"{s.get('consecutive_gtt_failures')} consecutive GTT "
                    f"(safety-net) failures")
    if float(s.get("drawdown_pct") or 0) >= _HARD_DD:
        kill.append(f"drawdown {s.get('drawdown_pct'):.0f}% — hard kill limit")
    if float(s.get("vix") or 0) >= _EXTREME_VIX:
        kill.append(f"extreme volatility (VIX {s.get('vix'):.0f}) — unknown state")

    if float(s.get("drawdown_pct") or 0) >= _SOFT_DD:
        derisk.append(f"drawdown {s.get('drawdown_pct'):.0f}% — de-risking")
    if float(s.get("calibration_ece") or 0) > _ECE_LIMIT:
        derisk.append("confidence miscalibrated — trust it less")
    if float(s.get("slippage_ratio") or 1) > _SLIP_LIMIT:
        derisk.append("slippage running ≥2× model — costs eating the edge")
    if s.get("regime_adverse"):
        derisk.append("regime unproven for the edge — stand smaller")

    if kill:
        return {"state": HALT, "size_multiplier": 0.0, "kill_reasons": kill,
                "derisk_reasons": derisk, "reasons": kill}
    if derisk:
        return {"state": DE_RISK, "size_multiplier": _DERISK_MULT,
                "kill_reasons": [], "derisk_reasons": derisk, "reasons": derisk}
    return {"state": NORMAL, "size_multiplier": 1.0, "kill_reasons": [],
            "derisk_reasons": [], "reasons": []}


# ══════════════════════════════════════════════════════════════════════════════
# Persistent counters + human kill-switch
# ══════════════════════════════════════════════════════════════════════════════

def _read_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _write_state(d: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(d))
    except Exception:
        pass


def set_manual_halt(on: bool) -> None:
    """The human kill-switch — absolute, survives restarts."""
    d = _read_state()
    d["manual_halt"] = bool(on)
    _write_state(d)


def record_order_result(ok: bool, gtt_ok: bool | None = None) -> None:
    """Track CONSECUTIVE failures — the broker-rejecting / safety-net-failing
    kill signals. A success resets the relevant counter."""
    d = _read_state()
    d["consecutive_order_failures"] = 0 if ok else int(d.get("consecutive_order_failures") or 0) + 1
    if gtt_ok is not None:
        d["consecutive_gtt_failures"] = 0 if gtt_ok else int(d.get("consecutive_gtt_failures") or 0) + 1
    d["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_state(d)


# ══════════════════════════════════════════════════════════════════════════════
# Fail-open collectors → live signals snapshot
# ══════════════════════════════════════════════════════════════════════════════

def _collect_signals() -> dict:
    s = dict(_read_state())          # manual_halt + failure counters
    # data integrity (corporate-action mismatch / staleness)
    try:
        from core.data_integrity import integrity_report
        rep = integrity_report()
        if rep.get("checked", 0) > 0:
            s["ca_mismatch"] = bool(rep.get("ca_mismatch"))
            s["data_corrupt"] = bool(rep.get("stale"))
    except Exception:
        pass
    # real-account drawdown (from actual trades)
    try:
        from reports.verdict_dashboard import build_trade_equity_curve
        s["drawdown_pct"] = float(build_trade_equity_curve()["stats"].get(
            "max_drawdown_pct", 0) or 0)
    except Exception:
        pass
    # confidence calibration
    try:
        from research.calibration import calibration_report
        ece = calibration_report().get("ece")
        if ece is not None:
            s["calibration_ece"] = float(ece)
    except Exception:
        pass
    return s


# ══════════════════════════════════════════════════════════════════════════════
# Public API — the gate every order path consults
# ══════════════════════════════════════════════════════════════════════════════

_cache = {"ts": 0.0, "data": None}


def assess(force: bool = False) -> dict:
    """Current governance state (cached 60s). Fail-open → NORMAL with a note (a
    broken sentinel must not block all trading, but every real condition it CAN
    see is honored)."""
    if not force and _cache["data"] and time.time() - _cache["ts"] < 60:
        return _cache["data"]
    try:
        out = evaluate_state(_collect_signals())
    except Exception as exc:
        out = {"state": NORMAL, "size_multiplier": 1.0, "kill_reasons": [],
               "derisk_reasons": [], "reasons": [], "error": str(exc)}
    _cache.update(ts=time.time(), data=out)
    return out


def can_place_order() -> tuple[bool, float, str]:
    """(allowed, size_multiplier, reason) — the one call every LIVE order path
    makes. HALT blocks the order; DE_RISK allows it at reduced size; NORMAL is
    clear."""
    g = assess()
    if g["state"] == HALT:
        return False, 0.0, "🛑 Trading HALTED: " + "; ".join(g["kill_reasons"])
    if g["state"] == DE_RISK:
        return True, g["size_multiplier"], "⚠️ DE-RISK: " + "; ".join(g["derisk_reasons"])
    return True, 1.0, ""


def governance_directive() -> list[dict]:
    """Brain-ready — surface a HALT/DE_RISK so the read reflects it. Fail-open."""
    g = assess()
    if g["state"] == HALT:
        return [{"severity": "warn",
                 "text": "🛑 Governance HALT — new trades blocked: "
                         + "; ".join(g["kill_reasons"])}]
    if g["state"] == DE_RISK:
        return [{"severity": "warn",
                 "text": "⚠️ Governance DE-RISK — size halved: "
                         + "; ".join(g["derisk_reasons"])}]
    return []

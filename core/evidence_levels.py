"""
🎚️ Evidence Levels — maturity that can only rise through PROOF, never belief.

Institutional maturity is communicated by what evidence supports a capability, not
by how good the code looks. Every capability is rated by its highest DEFENSIBLE
evidence about the real market:

    E0 designed · E1 code-reviewed · E2 unit-tested · E3 historically validated
    (real clean data) · E4 forward-paper validated · E5 live-capital validated ·
    E6 stable across regimes

The committee's rule, enforced here in code: a level rises ONE step at a time and
ONLY when an OBJECTIVE evidence artifact clears that step's gate — the same
pre-registered thresholds as the Governance Charter. There is deliberately NO way
to set a level by developer judgment; `promote()` refuses anything that doesn't
meet the numeric gate, and records the artifact to an audit trail. Withdrawing
trust is the safe direction, so `demote()` is always allowed and audited.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

E0, E1, E2, E3, E4, E5, E6 = range(7)
_LABEL = {E0: "E0 designed", E1: "E1 code-reviewed", E2: "E2 unit-tested",
          E3: "E3 historically validated", E4: "E4 paper-validated",
          E5: "E5 live-validated", E6: "E6 stable across regimes"}

_AUDIT = Path(__file__).resolve().parent.parent / "logs" / "evidence_audit.jsonl"

# capability → (current level, one-line honest basis). Conservative on purpose.
_LEVELS: dict[str, tuple[int, str]] = {
    "Execution engine": (E2, "correct order sequence; never validated on real fills"),
    "Outcome resolver (first-touch)": (E2, "10/10 on ground truth; zero real outcomes"),
    "Cost model (mechanism)": (E2, "applied consistently, tested"),
    "Cost model (values)": (E0, "0.32%/slippage are assumed, not measured"),
    "Backtest engine": (E2, "walk-forward, no-lookahead — never run on real data"),
    "Anti-overfit harness": (E2, "coded + tested; zero experiments on real data"),
    "Strategy alpha (the edge)": (E0, "designed only — no historical/paper/live evidence"),
    "Signal scoring calibration": (E0, "defaults to 1.0 with an empty store"),
    "Confidence calibration": (E2, "math tested; no real forecasts scored → E0 evidence"),
    "Drift / beliefs / EV": (E2, "code tested; dormant until ≥30 real outcomes → E0 evidence"),
    "Feature platform": (E2, "code tested; thin corpus, no real observations"),
    "Long-term screener": (E2, "code tested; never validated on real forward returns"),
    "Portfolio risk / sizing": (E2, "rules enforced & tested"),
    "Data integrity guard": (E2, "phantom-gap scan tested; CA-adjustment not yet built"),
    "Governance sentinel": (E2, "kill/rollback logic tested; unexercised in live"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Objective promotion gates — the ONLY way a level rises (pre-registered)
# ══════════════════════════════════════════════════════════════════════════════

def _missing(artifact: dict, checks: list[tuple[str, bool, str]]) -> list[str]:
    """checks = [(label, passed_bool, ...)]; returns the labels that failed."""
    return [label for label, ok, *_ in checks if not ok]


def _gate_e1(a: dict) -> tuple[bool, list[str]]:
    fails = _missing(a, [("code reviewed", bool(a.get("reviewed")))])
    return not fails, fails


def _gate_e2(a: dict) -> tuple[bool, list[str]]:
    fails = _missing(a, [
        ("tests present", int(a.get("tests_passed") or 0) >= 1),
        ("all tests passing", bool(a.get("all_pass")))])
    return not fails, fails


def _gate_e3(a: dict) -> tuple[bool, list[str]]:
    # HISTORICAL validation on clean, survivorship-complete data (Charter Phase 1)
    fails = _missing(a, [
        ("data integrity clean (no CA mismatch)", bool(a.get("data_clean"))),
        ("Deflated Sharpe > 0.95", float(a.get("dsr") or 0) > 0.95),
        ("Reality-Check p < 0.05", float(a.get("reality_check_p") or 1) < 0.05),
        ("FDR corrected", bool(a.get("fdr_corrected"))),
        ("power ≥ 0.8", float(a.get("power") or 0) >= 0.8),
        ("net expectancy ≥ +0.15R after costs", float(a.get("net_expectancy_r") or -9) >= 0.15),
        ("profit factor ≥ 1.3", float(a.get("profit_factor") or 0) >= 1.3),
        ("positive in ≥2 regimes", int(a.get("regimes_positive") or 0) >= 2)])
    return not fails, fails


def _gate_e4(a: dict) -> tuple[bool, list[str]]:
    # FORWARD PAPER validation (Charter Phase 2)
    fails = _missing(a, [
        ("≥300 forward paper trades", int(a.get("paper_trades") or 0) >= 300),
        ("expectancy 95% CI lower bound > 0", float(a.get("expectancy_ci_lower") or -9) > 0),
        ("calibration ECE < 0.05", float(a.get("ece") or 1) < 0.05),
        ("Brier skill > 0", float(a.get("brier_skill") or -1) > 0),
        ("slippage ≤ 1.5× model", float(a.get("slippage_ratio") or 9) <= 1.5),
        ("max drawdown ≤ 20%", float(a.get("max_drawdown_pct") or 99) <= 20),
        ("positive in ≥2 regimes", int(a.get("regimes_positive") or 0) >= 2)])
    return not fails, fails


def _gate_e5(a: dict) -> tuple[bool, list[str]]:
    # LIVE validation + reconciliation (Charter Phase 3)
    fails = _missing(a, [
        ("≥50 live trades", int(a.get("live_trades") or 0) >= 50),
        ("live within 1 SE of paper", bool(a.get("live_within_se"))),
        ("slippage ≤ 2× model", float(a.get("slippage_ratio") or 9) <= 2.0),
        ("reconciliation clean", bool(a.get("reconciliation_ok")))])
    return not fails, fails


def _gate_e6(a: dict) -> tuple[bool, list[str]]:
    fails = _missing(a, [
        ("≥3 regimes covered", int(a.get("regimes_covered") or 0) >= 3),
        ("sample ≥ 500", int(a.get("sample") or 0) >= 500),
        ("stable (no confirmed drift)", bool(a.get("stable")))])
    return not fails, fails


_GATES = {E1: _gate_e1, E2: _gate_e2, E3: _gate_e3, E4: _gate_e4,
          E5: _gate_e5, E6: _gate_e6}


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

def level_of(capability: str) -> tuple[int, str]:
    return _LEVELS.get(capability, (E0, "unclassified"))


def next_gate(capability: str) -> dict:
    """What OBJECTIVE evidence is required to advance one level — transparency so
    nobody has to guess (or fudge) the bar."""
    cur = level_of(capability)[0]
    if cur >= E6:
        return {"capability": capability, "at": _LABEL[E6], "next": None}
    nxt = cur + 1
    # probe the gate with an empty artifact → the full list of requirements
    _, fails = _GATES[nxt]({})
    return {"capability": capability, "at": _LABEL[cur], "next": _LABEL[nxt],
            "requires": fails}


def _audit(entry: dict) -> None:
    try:
        _AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def promote(capability: str, artifact: dict) -> dict:
    """Raise a capability's level by ONE step — ONLY if `artifact` clears that
    step's objective gate. No level-skipping, no developer-judgment override, no
    free-text basis. Refuses (with the exact unmet criteria) otherwise, and audits
    every promotion. `artifact` must carry the measured metrics from a validation
    workflow (harness report, paper report, live reconciliation)."""
    cur = level_of(capability)[0]
    if cur >= E6:
        return {"promoted": False, "reason": "already at E6 (max)"}
    nxt = cur + 1
    ok, fails = _GATES[nxt](artifact or {})
    if not ok:
        return {"promoted": False, "from": _LABEL[cur], "to": _LABEL[nxt],
                "reason": "gate not met", "unmet": fails}
    basis = f"{_LABEL[nxt]} via {artifact.get('source', 'validation artifact')}"
    _LEVELS[capability] = (nxt, basis)
    _audit({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "capability": capability,
            "from": _LABEL[cur], "to": _LABEL[nxt], "action": "PROMOTE",
            "artifact": artifact})
    return {"promoted": True, "from": _LABEL[cur], "to": _LABEL[nxt], "basis": basis}


def demote(capability: str, to_level: int, reason: str) -> dict:
    """Withdraw evidence — always allowed (the SAFE direction). Used when a
    validated edge decays or a gate later fails. Audited."""
    cur = level_of(capability)[0]
    to_level = max(E0, min(int(to_level), cur))
    _LEVELS[capability] = (to_level, f"demoted: {reason}")
    _audit({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "capability": capability,
            "from": _LABEL[cur], "to": _LABEL[to_level], "action": "DEMOTE",
            "reason": reason})
    return {"demoted": True, "from": _LABEL[cur], "to": _LABEL[to_level]}


def report() -> list[dict]:
    rows = [{"capability": k, "level": v[0], "label": _LABEL[v[0]], "basis": v[1]}
            for k, v in _LEVELS.items()]
    return sorted(rows, key=lambda r: r["level"])


def headline() -> str:
    infra = [v[0] for k, v in _LEVELS.items()
             if any(t in k for t in ("Execution", "Outcome", "risk", "platform",
                                     "Governance", "integrity"))]
    alpha = _LEVELS["Strategy alpha (the edge)"][0]
    infra_min = min(infra) if infra else E0
    return (f"Infrastructure ≥ {_LABEL[infra_min]}; the ALPHA is {_LABEL[alpha]} "
            f"— i.e. the edge is unproven. Not investable until it reaches E3+.")

"""
🎚️ Evidence Levels — how much the market has actually PROVEN about each capability.

Institutional maturity is communicated not by "it's built" but by "what evidence
supports it." Every capability is rated by its highest DEFENSIBLE evidence about
the real market — never by how good the code looks. This is the system telling the
honest truth about itself, so no screen can imply more certainty than exists.

    E0 designed · E1 code-reviewed · E2 unit-tested · E3 historically validated
    (real clean data) · E4 forward-paper validated · E5 live-capital validated ·
    E6 stable across regimes

The classification below is deliberately conservative: infrastructure that is
tested but never run on real data is E2, and ALL alpha/learning is E0 until real
outcomes exist. `promote()` is the ONLY way a level rises, and only the evidence
milestones in the Governance Charter justify a call to it.
"""
from __future__ import annotations

E0, E1, E2, E3, E4, E5, E6 = range(7)

_LABEL = {E0: "E0 designed", E1: "E1 code-reviewed", E2: "E2 unit-tested",
          E3: "E3 historically validated", E4: "E4 paper-validated",
          E5: "E5 live-validated", E6: "E6 stable across regimes"}

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


def level_of(capability: str) -> tuple[int, str]:
    return _LEVELS.get(capability, (E0, "unclassified"))


def promote(capability: str, level: int, basis: str) -> None:
    """Raise a capability's evidence level. Call ONLY when a Governance-Charter
    milestone is met (e.g., a signal survives the harness on real data → E3)."""
    _LEVELS[capability] = (int(level), basis)


def report() -> list[dict]:
    """The honest maturity scorecard, lowest evidence first (what to trust least)."""
    rows = [{"capability": k, "level": v[0], "label": _LABEL[v[0]], "basis": v[1]}
            for k, v in _LEVELS.items()]
    return sorted(rows, key=lambda r: r["level"])


def headline() -> str:
    """One honest line: infra vs alpha maturity."""
    infra = [v[0] for k, v in _LEVELS.items()
             if any(t in k for t in ("Execution", "Outcome", "risk", "platform",
                                     "Governance", "integrity"))]
    alpha = _LEVELS["Strategy alpha (the edge)"][0]
    infra_min = min(infra) if infra else E0
    return (f"Infrastructure ≥ {_LABEL[infra_min]}; the ALPHA is {_LABEL[alpha]} "
            f"— i.e. the edge is unproven. Not investable until it reaches E3+.")

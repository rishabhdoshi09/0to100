"""Scientific-memory bridge — source-labelled forward evidence, no online rule mutation."""
from __future__ import annotations

from research.forward_evidence.reporting import policy_report
from research.forward_evidence.sources import PAPER_FORWARD


def remember_forward_policy(policy_id: str, *, hypothesis_id: str | None = None) -> str | None:
    """Record a WATCH belief from PAPER_FORWARD stats. Never promotes live authority."""
    try:
        from research import scientific_memory as SM
    except Exception:
        return None
    rep = policy_report(policy_id)
    n = int(rep["paper_forward"]["n"])
    if n <= 0:
        return None
    exp = rep["paper_forward"]["expectancy_r"]
    statement = (
        f"{policy_id} PAPER_FORWARD n={n} expectancy_R={exp}; "
        f"conclusion={rep['scientific_conclusion']} "
        f"[source={PAPER_FORWARD}; live_authorized=False]"
    )
    return SM.record_belief(
        statement,
        signal=f"forward_paper:{policy_id}",
        status=SM.WATCH,
        evidence_n=n,
        confidence="LOW" if n < 30 else "MEDIUM",
        ev_r=exp,
        hypothesis_id=hypothesis_id,
        notes="Forward paper observation only. Does not change frozen strategy rules.",
    )

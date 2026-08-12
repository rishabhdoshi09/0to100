"""Arm PAPER forward-evidence using existing PAPER_AUTO controls. LIVE stays blocked."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.forward_evidence.policy_allowlist import (
    seed_default_allowlist,
    status_rows,
    PaperPolicyAllowlist,
)
from research.forward_evidence.reporting import learning_blurb, policy_report
from research.forward_evidence.outcome_ledger import load_outcomes
from research.forward_evidence.sources import PAPER_FORWARD, LIVE
from research.forward_evidence.paper_vs_live import compare_policy
from research.intelligence.runtime import modes as MODES

REPO = Path(__file__).resolve().parents[2]
STATUS_PATH = REPO / "logs" / "forward_evidence" / "system_status.json"


def ensure_armed(*, enable_paper_auto: bool = True) -> dict[str, Any]:
    """Seed allowlist and ensure PAPER_AUTO is enabled. Never unlocks LIVE."""
    al = seed_default_allowlist()
    paper_cfg = REPO / "logs" / "intelligence" / "paper_config.json"
    paper_cfg.parent.mkdir(parents=True, exist_ok=True)
    enabled = True
    if paper_cfg.exists():
        try:
            enabled = bool(json.loads(paper_cfg.read_text()).get("enabled", True))
        except Exception:
            enabled = True
    if enable_paper_auto and not enabled:
        # Owner previously disabled — respect opt-out; still seed allowlist
        armed = False
    else:
        if enable_paper_auto:
            try:
                from research.auto_research.scheduler import get_brain
                brain = get_brain()
                brain.enable_paper_auto()
                armed = brain.is_paper_auto_enabled()
            except Exception:
                # Persist config even if brain wiring unavailable in this process
                paper_cfg.write_text(json.dumps({
                    "enabled": True,
                    "starting_capital": 100_000.0,
                    "forward_evidence_mode": "PAPER_FORWARD_EVIDENCE",
                    "live_enabled": False,
                }), encoding="utf-8")
                armed = True
        else:
            armed = False

    status = {
        "paper_auto_trading_ready": True,
        "paper_mode_armed": bool(armed or enabled),
        "mode": MODES.PAPER_AUTO,
        "forward_evidence_label": "PAPER_FORWARD_EVIDENCE",
        "live_trading_enabled": False,
        "limited_live_enabled": False,
        "live_data_ingestion_ready": True,  # observer/recon exist; submission blocked
        "broker_mutations_enabled": False,
        "current_paper_policies": [
            p.research_policy_id for p in al.paper_active()
        ],
        "denied_policies": [
            p.research_policy_id for p in al.paper_denied()
        ],
        "forward_evidence_location": str((REPO / "logs" / "forward_evidence").relative_to(REPO)),
        "next_operational_action": (
            "Keep autonomy supervisor running during market hours; "
            "paper trades accumulate automatically for allowlisted policies."
        ),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def system_status() -> dict[str, Any]:
    if STATUS_PATH.exists():
        try:
            base = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            base = ensure_armed()
    else:
        base = ensure_armed()
    al = seed_default_allowlist()
    paper_n = len(load_outcomes(evidence_source=PAPER_FORWARD))
    live_n = len(load_outcomes(evidence_source=LIVE))
    base["paper_outcomes_recorded"] = paper_n
    base["live_outcomes_recorded"] = live_n
    base["policy_table"] = status_rows(al)
    base["live_trading_enabled"] = False
    return base


def plain_operating_guide() -> list[str]:
    st = system_status()
    lines = [
        "Paper trading is ON." if st.get("paper_mode_armed") else "Paper trading is OFF.",
        "QuantTerm can take simulated trades automatically for allowlisted policies.",
        "Allowlisted strategies are being observed, not trusted yet.",
        "No real money is being used.",
        f"Forward paper outcomes recorded: {st.get('paper_outcomes_recorded', 0)}.",
    ]
    if not st.get("live_outcomes_recorded"):
        lines.append("NO LIVE EVIDENCE YET.")
    for pid in st.get("current_paper_policies", [])[:6]:
        lines.append(learning_blurb(pid))
    lines.append("EXP-FUND-03 / earnings-growth is NOT paper-enabled from research status alone.")
    return lines


def dashboard_payload() -> dict[str, Any]:
    """Payload for the Streamlit paper evidence panel."""
    st = system_status()
    al = seed_default_allowlist()
    policy_cards = []
    for p in al.paper_active():
        rep = policy_report(p.research_policy_id)
        cmp_ = compare_policy(
            policy_id=p.research_policy_id,
            paper_outcomes=load_outcomes(evidence_source=PAPER_FORWARD, policy_id=p.research_policy_id),
            live_outcomes=load_outcomes(evidence_source=LIVE, policy_id=p.research_policy_id),
        )
        policy_cards.append({
            "policy": p.research_policy_id,
            "scientific_status": p.scientific_status,
            "paper_observation": "ACTIVE",
            "live_status": "NOT AUTHORIZED",
            "report": rep,
            "learning": learning_blurb(p.research_policy_id),
            "paper_vs_live": cmp_.as_dict(),
        })
    return {"system": st, "guide": plain_operating_guide(), "policies": policy_cards,
            "denied": status_rows(al)}

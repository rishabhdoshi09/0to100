"""PAPER policy allowlist — paper_enabled is independent of live_enabled.

Scientific PASS/CONFIRMED/INCONCLUSIVE never auto-infers paper observation
authority. EXP-FUND-03 (INCONCLUSIVE_FOLLOWUP) is explicitly denied.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "logs" / "forward_evidence" / "paper_policy_allowlist.json"

# Explicitly never auto-enable from research status alone
_DENY_POLICY_IDS = frozenset({
    "EXP-FUND-01", "EXP-FUND-02", "EXP-FUND-03", "EXP-FUND-04",
    "EXP-FUND-03-FOLLOWUP",
    "earnings_growth", "pead", "quality_profitability", "value_pe",
    "low_vol", "low-volatility",
})

# Observational seed: existing bar-by-bar PAPER_AUTO families (not research fund cycle)
_SEED_FAMILIES = (
    "cross_sectional_momentum",
    "breakout",
    "pullback",
    "trend_following",
    "relative_strength",
    "sector_rotation",
)


@dataclass
class PaperPolicy:
    research_policy_id: str
    version: str = "1"
    family: str = ""
    paper_enabled: bool = False
    live_enabled: bool = False          # ALWAYS False unless separate owner LIVE unlock
    frozen_config_hash: str = ""
    approved_at: str = ""
    approval_reason: str = ""
    evidence_source: str = "PAPER_FORWARD"
    scientific_status: str = "UNPROVEN"
    paper_observation_status: str = "INACTIVE"  # ACTIVE / INACTIVE / DENIED
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class PaperPolicyAllowlist:
    """Durable allowlist. paper_enabled != live_enabled is the normal case."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self._policies: dict[str, PaperPolicy] = {}
        self.load()

    def load(self) -> None:
        self._policies = {}
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in raw.get("policies", []):
            try:
                p = PaperPolicy(**{k: v for k, v in row.items() if k in PaperPolicy.__dataclass_fields__})
                # Hard safety: never persist live_enabled True from this module's seed path
                if p.research_policy_id in _DENY_POLICY_IDS:
                    p.paper_enabled = False
                    p.live_enabled = False
                    p.paper_observation_status = "DENIED"
                self._policies[p.research_policy_id] = p
            except Exception:
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "live_trading_enabled": False,
            "note": "paper_enabled does not grant live_enabled. LIVE stays blocked.",
            "policies": [p.as_dict() for p in sorted(self._policies.values(),
                                                     key=lambda x: x.research_policy_id)],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def upsert(self, policy: PaperPolicy) -> PaperPolicy:
        if policy.research_policy_id in _DENY_POLICY_IDS:
            policy.paper_enabled = False
            policy.live_enabled = False
            policy.paper_observation_status = "DENIED"
            policy.approval_reason = (
                policy.approval_reason
                or "Explicitly denied — research status does not grant paper observation"
            )
        # This module never turns live on
        if policy.live_enabled:
            policy.live_enabled = False
        self._policies[policy.research_policy_id] = policy
        self.save()
        return policy

    def get(self, policy_id: str) -> PaperPolicy | None:
        return self._policies.get(policy_id)

    def may_paper_trade(self, policy_id: str, *, family: str = "") -> bool:
        if policy_id in _DENY_POLICY_IDS or family in _DENY_POLICY_IDS:
            return False
        p = self._policies.get(policy_id)
        if p is None and family:
            p = self._policies.get(family)
        if p is None and family:
            for cand in self._policies.values():
                if cand.family == family and cand.paper_enabled:
                    p = cand
                    break
        if p is None:
            return False
        return bool(p.paper_enabled) and p.paper_observation_status == "ACTIVE" and not p.live_enabled

    def may_live_trade(self, policy_id: str) -> bool:
        """Always False in this milestone — live requires separate owner unlock."""
        return False

    def list_policies(self) -> list[PaperPolicy]:
        return sorted(self._policies.values(), key=lambda p: p.research_policy_id)

    def paper_active(self) -> list[PaperPolicy]:
        return [p for p in self.list_policies() if p.paper_enabled and p.paper_observation_status == "ACTIVE"]

    def paper_denied(self) -> list[PaperPolicy]:
        return [p for p in self.list_policies() if p.paper_observation_status == "DENIED"]


def seed_default_allowlist(path: Path | None = None, *, force: bool = False) -> PaperPolicyAllowlist:
    """Seed observational PAPER policies for existing runtime families.

    Does NOT enable EXP-FUND-03 or other fund-cycle experiments.
    """
    al = PaperPolicyAllowlist(path)
    if al._policies and not force:
        # Still ensure deny entries exist
        _ensure_denies(al)
        return al

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for fam in _SEED_FAMILIES:
        al.upsert(PaperPolicy(
            research_policy_id=fam,
            version="1",
            family=fam,
            paper_enabled=True,
            live_enabled=False,
            frozen_config_hash="",
            approved_at=now,
            approval_reason="FORWARD_PAPER_OBSERVATION — existing PAPER_AUTO runtime family",
            evidence_source="PAPER_FORWARD",
            scientific_status="UNPROVEN",
            paper_observation_status="ACTIVE",
            notes="Observed in paper only. Not trusted for real money.",
        ))
    _ensure_denies(al)
    return al


def _ensure_denies(al: PaperPolicyAllowlist) -> None:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    denies = [
        ("EXP-FUND-03", "INCONCLUSIVE_FOLLOWUP",
         "Follow-up inconclusive; RECORD_EVIDENCE_NO_TUNING — not paper-enabled from status"),
        ("EXP-FUND-03-FOLLOWUP", "INCONCLUSIVE_FOLLOWUP", "Follow-up itself is not a trade policy"),
        ("EXP-FUND-01", "INCONCLUSIVE", "HOLD_NO_TUNING"),
        ("EXP-FUND-02", "INCONCLUSIVE", "HOLD_NO_TUNING"),
        ("EXP-FUND-04", "INCONCLUSIVE", "HOLD_NO_TUNING"),
        ("earnings_growth", "INCONCLUSIVE_FOLLOWUP", "Alias deny for earnings-growth research"),
        ("low_vol", "FAIL", "Closed branch — do not reopen as paper edge claim"),
    ]
    for pid, sci, reason in denies:
        existing = al.get(pid)
        if existing and existing.paper_observation_status == "DENIED" and not force_update(existing):
            continue
        al.upsert(PaperPolicy(
            research_policy_id=pid,
            family=pid,
            paper_enabled=False,
            live_enabled=False,
            approved_at=now,
            approval_reason=reason,
            scientific_status=sci,
            paper_observation_status="DENIED",
            notes="Scientific status ≠ paper observation authority.",
        ))


def force_update(_p: PaperPolicy) -> bool:
    return False


def status_rows(al: PaperPolicyAllowlist | None = None) -> list[dict[str, Any]]:
    al = al or seed_default_allowlist()
    rows = []
    for p in al.list_policies():
        rows.append({
            "POLICY": p.research_policy_id,
            "SCIENTIFIC STATUS": p.scientific_status,
            "PAPER OBSERVATION STATUS": (
                "ACTIVE" if p.paper_enabled and p.paper_observation_status == "ACTIVE"
                else p.paper_observation_status
            ),
            "LIVE STATUS": "NOT AUTHORIZED",
            "family": p.family,
            "approval_reason": p.approval_reason,
        })
    return rows

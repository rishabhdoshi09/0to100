"""
🚦 Live preflight + readiness states (Phases 13, 23).

`readiness_state(...)` reports ONE explicit state — never blurred — from NOT_READY up to
USER_ACTIVATED. `live_preflight(...)` runs the hard gates that must all pass before LIMITED_LIVE
may create new risk; a failed critical check blocks new risk (existing positions still managed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ems import schemas as SC


@dataclass
class PreflightReport:
    ok: bool = True
    failed: list = field(default_factory=list)
    passed: list = field(default_factory=list)

    def chk(self, name, cond, reason=""):
        if cond:
            self.passed.append(name)
        else:
            self.failed.append((name, reason)); self.ok = False

    def as_dict(self):
        return {"ok": self.ok, "failed": self.failed, "passed": self.passed}


def readiness_state(*, architecture_ok: bool, simulator_certified: bool,
                    broker_connected: bool, user_activated: bool) -> str:
    if user_activated and broker_connected:
        return SC.USER_ACTIVATED
    if broker_connected:
        return SC.BROKER_CONNECTED
    if simulator_certified:
        return SC.SIMULATOR_CERTIFIED
    if architecture_ok:
        return SC.ARCHITECTURE_READY
    return SC.NOT_READY


def live_preflight(*, mode: str, envelope, snapshot_verified: bool, forward_eligible: bool,
                   data_fresh: bool, registry_valid: bool, broker_authenticated: bool,
                   reconciled: bool, governor_healthy: bool, ledger_writable: bool,
                   protection_healthy: bool, has_critical_incident: bool,
                   daily_loss_ok: bool) -> PreflightReport:
    r = PreflightReport()
    r.chk("mode_is_live", SC.is_live_mode(mode), f"mode {mode} is not live")
    r.chk("envelope_user_approved", envelope is not None and envelope.is_user_approved(),
          "operating envelope not user-approved")
    r.chk("snapshot_verified", snapshot_verified, "active snapshot not verified")
    r.chk("forward_eligible", forward_eligible, "data not forward-eligible")
    r.chk("data_fresh", data_fresh, "data not fresh per trading calendar")
    r.chk("registry_valid", registry_valid, "strategy registry invalid")
    r.chk("broker_authenticated", broker_authenticated, "broker not authenticated")
    r.chk("reconciled", reconciled, "broker/local state not reconciled")
    r.chk("governor_healthy", governor_healthy, "risk governor unhealthy")
    r.chk("ledger_writable", ledger_writable, "execution ledger not writable")
    r.chk("protection_healthy", protection_healthy, "protection service unhealthy")
    r.chk("no_critical_incident", not has_critical_incident, "unresolved critical incident")
    r.chk("daily_loss_permits", daily_loss_ok, "daily-loss state blocks trading")
    return r

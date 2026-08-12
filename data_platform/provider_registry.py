"""Provider capability registry with source health and priority."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from data_platform.contracts import DataCapability, QualityStatus


@dataclass
class ProviderSpec:
    name: str
    capabilities: frozenset[DataCapability]
    priority: int
    authentication_status: str = "unknown"
    last_success: str = ""
    last_failure: str = ""
    coverage_note: str = ""
    freshness_note: str = ""


_REGISTRY: list[ProviderSpec] = [
    ProviderSpec(
        name="nse_bhavcopy",
        capabilities=frozenset({DataCapability.DAILY_PRICES, DataCapability.CORPORATE_ACTIONS}),
        priority=1,
        authentication_status="public_download",
        coverage_note="Official NSE EOD OHLCV via logs/bhav",
    ),
    ProviderSpec(
        name="kite",
        capabilities=frozenset({DataCapability.LIVE_QUOTES, DataCapability.SECURITY_MASTER}),
        priority=2,
        authentication_status="token_required",
        coverage_note="Zerodha Kite when access token present",
    ),
    ProviderSpec(
        name="nse_snapshot",
        capabilities=frozenset({DataCapability.LIVE_QUOTES}),
        priority=3,
        authentication_status="public",
        coverage_note="NSE index-API snapshot overlay",
    ),
    ProviderSpec(
        name="screener_deep",
        capabilities=frozenset({DataCapability.FUNDAMENTALS, DataCapability.OWNERSHIP}),
        priority=4,
        authentication_status="scrape",
        coverage_note="Screener.in via fundamentals resolver (primary)",
    ),
    ProviderSpec(
        name="yahoo_finance",
        capabilities=frozenset({DataCapability.FUNDAMENTALS, DataCapability.DAILY_PRICES}),
        priority=6,
        authentication_status="public",
        coverage_note="Reputed fallback when Screener.in is blocked or thin",
    ),
    ProviderSpec(
        name="google_finance",
        capabilities=frozenset({DataCapability.LIVE_QUOTES}),
        priority=99,
        authentication_status="fragile_scrape",
        coverage_note="Fallback only — never primary",
    ),
    ProviderSpec(
        name="user_import",
        capabilities=frozenset({
            DataCapability.DAILY_PRICES,
            DataCapability.FUNDAMENTALS,
            DataCapability.OWNERSHIP,
            DataCapability.CORPORATE_ACTIONS,
        }),
        priority=5,
        authentication_status="user_supplied",
        coverage_note="Validated CSV/ZIP import pipeline + Research Data uploads",
    ),
]

def providers_for(capability: DataCapability) -> list[ProviderSpec]:
    ranked = [p for p in _REGISTRY if capability in p.capabilities]
    return sorted(ranked, key=lambda p: p.priority)


def registry_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    bhav_ready = False
    bhav_sessions = 0
    try:
        from data.bhavcopy_runtime import status
        snap = status(load_cache=False)
        bhav_ready = bool(snap.get("ready"))
        bhav_sessions = int(snap.get("sessions") or 0)
    except Exception:
        pass
    fund_cache = False
    try:
        from pathlib import Path
        fund_cache = Path("data/fundamentals_cache.db").exists()
    except Exception:
        pass
    rows = []
    for spec in _REGISTRY:
        health = QualityStatus.MISSING
        note = spec.coverage_note
        if spec.name == "nse_bhavcopy" and bhav_ready:
            health = QualityStatus.FRESH
            note = f"{bhav_sessions} sessions loaded"
        elif spec.name == "screener_deep" and fund_cache:
            health = QualityStatus.PARTIAL
        elif spec.name == "user_import":
            health = QualityStatus.NOT_APPLICABLE
        rows.append({
            "name": spec.name,
            "capabilities": [c.value for c in spec.capabilities],
            "priority": spec.priority,
            "authentication_status": spec.authentication_status,
            "health": health.value,
            "coverage_note": note,
            "last_checked": now,
        })
    return {"generated_at": now, "providers": rows}


def pick_provider(capability: DataCapability) -> ProviderSpec | None:
    ranked = providers_for(capability)
    return ranked[0] if ranked else None

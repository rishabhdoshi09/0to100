"""Field-level quality tokens. Future experiments fail closed on these."""

FUNDAMENTAL_PIT_STRONG = "FUNDAMENTAL_PIT_STRONG"
FUNDAMENTAL_PIT_DEGRADED = "FUNDAMENTAL_PIT_DEGRADED"
FUNDAMENTAL_DESCRIPTIVE = "FUNDAMENTAL_DESCRIPTIVE"
FUNDAMENTAL_MISSING = "FUNDAMENTAL_MISSING"

EVENT_TIMESTAMP_STRONG = "EVENT_TIMESTAMP_STRONG"
EVENT_DATE_ONLY = "EVENT_DATE_ONLY"
EVENT_STATIC_OR_UNVERIFIED = "EVENT_STATIC_OR_UNVERIFIED"
EVENT_MISSING = "EVENT_MISSING"


def fundamental_quality(row: dict | None) -> str:
    if not row:
        return FUNDAMENTAL_MISSING
    if not row.get("available_at"):
        return FUNDAMENTAL_DESCRIPTIVE
    src = str(row.get("source") or "")
    if src.startswith("nse") and row.get("raw_hash"):
        if row.get("quarterly_usable") is False and str(row.get("period") or "").lower() == "quarterly":
            return FUNDAMENTAL_PIT_DEGRADED
        return FUNDAMENTAL_PIT_STRONG
    if row.get("available_at"):
        return FUNDAMENTAL_PIT_DEGRADED
    return FUNDAMENTAL_DESCRIPTIVE


def event_quality(row: dict | None) -> str:
    if not row:
        return EVENT_MISSING
    tq = row.get("time_quality")
    if tq in {EVENT_TIMESTAMP_STRONG, EVENT_DATE_ONLY, EVENT_STATIC_OR_UNVERIFIED, EVENT_MISSING}:
        return tq
    if row.get("timestamp") or row.get("available_at_ts"):
        return EVENT_TIMESTAMP_STRONG
    if row.get("announced_date") or row.get("available_at"):
        return EVENT_DATE_ONLY
    return EVENT_MISSING

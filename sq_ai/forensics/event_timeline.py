"""
Corporate Event Timeline.

Merges discrete corporate events from every intelligence source into one
chronological narrative:

    auditor changes / resignations / qualifications
    promoter stake & pledging changes (quarterly snapshots)
    guidance misses (Management Credibility Tracker)
    DAL announcements (acquisitions, buybacks, equity raises, ...)
    dated red flags (fiscal-period forensic findings)

A flat flag list answers "what is wrong?"; the timeline answers "in what
order did it go wrong?" — which is how blow-ups are actually recognised
in hindsight (pledge ↑ → auditor exit → receivables spike → cash collapse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from forensics.models import RedFlag, Severity


@dataclass
class TimelineEvent:
    date: str           # ISO date or fiscal label, used for sorting
    category: str       # Auditor / Ownership / Guidance / Forensic / Corporate
    description: str
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW / INFO


def _auditor_events(layers: Dict) -> List[TimelineEvent]:
    aud = layers.get("auditor")
    if aud is None:
        return []
    events: List[TimelineEvent] = []
    # Historical year-by-year record if curated data exists
    for fl in aud.flags:
        events.append(TimelineEvent(
            date=fl.period or "recent",
            category="Auditor",
            description=fl.title + " — " + fl.evidence,
            severity=fl.severity.value,
        ))
    return events


def _ownership_events(symbol: str) -> List[TimelineEvent]:
    """Quarter-over-quarter promoter / pledging moves from stored snapshots."""
    try:
        from forensics.promoter_intelligence import _load_snapshots
        snaps = _load_snapshots(symbol, 8)
    except Exception:
        return []
    events: List[TimelineEvent] = []
    # snaps are newest-first; walk pairs to detect QoQ moves
    for newer, older in zip(snaps, snaps[1:]):
        q = newer["quarter"]
        if newer["promoter_pct"] is not None and older["promoter_pct"] is not None:
            delta = newer["promoter_pct"] - older["promoter_pct"]
            if abs(delta) >= 1.0:
                sev = "HIGH" if delta <= -2.0 else "MEDIUM" if delta < 0 else "INFO"
                verb = "reduced" if delta < 0 else "increased"
                events.append(TimelineEvent(
                    date=q, category="Ownership",
                    description=f"Promoter stake {verb} {delta:+.1f}pp "
                                f"to {newer['promoter_pct']:.1f}%",
                    severity=sev))
        if newer["pledged_pct"] is not None and older["pledged_pct"] is not None:
            dpl = newer["pledged_pct"] - older["pledged_pct"]
            if abs(dpl) >= 2.0:
                sev = "HIGH" if dpl > 0 else "LOW"
                verb = "rose" if dpl > 0 else "fell"
                events.append(TimelineEvent(
                    date=q, category="Ownership",
                    description=f"Promoter pledging {verb} {dpl:+.1f}pp "
                                f"to {newer['pledged_pct']:.1f}%",
                    severity=sev))
    return events


def _guidance_events(symbol: str) -> List[TimelineEvent]:
    try:
        from forensics.management_credibility import _load_guidance
        items = _load_guidance(symbol)
    except Exception:
        return []
    events: List[TimelineEvent] = []
    for g in items:
        if g.met is False:
            events.append(TimelineEvent(
                date=g.fy, category="Guidance",
                description=f"Guidance miss — {g.metric}: guided "
                            f"{g.guided:.0%}, delivered {g.actual:.0%}",
                severity="MEDIUM" if (g.miss_magnitude or 0) < 0.4 else "HIGH"))
    return events


def _announcement_events(symbol: str) -> List[TimelineEvent]:
    """Corporate announcements stored in the DAL (acquisitions, buybacks...)."""
    try:
        from forensics import dal
        recs = dal.history("announcements", symbol)
    except Exception:
        return []
    events: List[TimelineEvent] = []
    for r in recs:
        payload = r.payload if isinstance(r.payload, dict) else {}
        events.append(TimelineEvent(
            date=r.period or r.fetched_at[:10],
            category="Corporate",
            description=str(payload.get("headline", payload or "Announcement")),
            severity=str(payload.get("severity", "INFO")).upper(),
        ))
    return events


def _forensic_events(flags: List[RedFlag]) -> List[TimelineEvent]:
    """Dated forensic findings (only flags carrying a fiscal period)."""
    return [TimelineEvent(
        date=fl.period, category="Forensic",
        description=fl.title, severity=fl.severity.value)
        for fl in flags
        if fl.period and fl.severity in (Severity.CRITICAL, Severity.HIGH)]


def build(symbol: str, layers: Dict, flags: List[RedFlag]) -> List[TimelineEvent]:
    """Merged corporate event timeline, oldest first."""
    events = (
        _auditor_events(layers)
        + _ownership_events(symbol)
        + _guidance_events(symbol)
        + _announcement_events(symbol)
        + _forensic_events(flags)
    )
    return sorted(events, key=lambda e: e.date)

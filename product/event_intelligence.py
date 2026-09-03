"""Official-event intelligence. Catalyst notes, not family confirmation.

Uses warehouse-dated announcements. Does not scrape historical media.
A classified event is evidence that something was disclosed at T — not
a BUY chip and not SECTOR_CONTEXT confirmation.
"""
from __future__ import annotations

from typing import Any

from product.pit_events import RESULTS, get_events


def catalyst_notes(symbol: str, *, as_of: str, path=None) -> dict[str, Any]:
    events = get_events(symbol, as_of=as_of, path=path, limit=20)
    classes = [e.get("event_class") for e in events]
    recent_results = [e for e in events if e.get("event_class") == RESULTS]
    return {
        "symbol": str(symbol).upper(),
        "as_of": str(as_of)[:10],
        "n_events": len(events),
        "classes": sorted({str(c) for c in classes if c}),
        "recent_result": recent_results[0] if recent_results else None,
        "usable_as_family_confirm": False,
        "note": (
            "Official announcements available at T. "
            "Not independent business-quality confirmation."
        ),
        "events": events[:12],
    }

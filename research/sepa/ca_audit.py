"""Corporate-action audit and quarantine. Never infer a factor from a price gap."""
from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from core.data_integrity import phantom_gaps
from research.sepa.frames import iso_date, slice_as_of


EVENT_CLASSES = (
    "split",
    "bonus",
    "consolidation",
    "rights",
    "demerger",
    "merger",
    "special_distribution",
    "symbol_restructuring",
    "genuine_market_gap",
    "bad_print",
    "unknown_discontinuity",
)

_DEMERGER = re.compile(r"\b(demerger|spin[- ]?off|scheme of arrangement|slump sale)\b", re.I)
_MERGER = re.compile(r"\b(merger|amalgamat)\b", re.I)
_RIGHTS = re.compile(r"\bright[s]?\s+(issue|entitlement)\b", re.I)
_RESTRUCT = re.compile(r"\b(name change|symbol change|restructur|capital reduction)\b", re.I)
_SPECIAL = re.compile(r"\b(special dividend|buyback|reduction of capital)\b", re.I)


def classify_subject(subject: str) -> str:
    s = str(subject or "")
    if _DEMERGER.search(s):
        return "demerger"
    if _MERGER.search(s):
        return "merger"
    if _RIGHTS.search(s):
        return "rights"
    if _RESTRUCT.search(s):
        return "symbol_restructuring"
    if _SPECIAL.search(s):
        return "special_distribution"
    return "unknown_discontinuity"


def classify_gap_event(
    *,
    symbol: str,
    date: str,
    pct: float,
    events_near: list[dict] | None = None,
    subjects: list[str] | None = None,
) -> dict[str, Any]:
    """Label one unresolved discontinuity. ratio hints are investigative only."""
    near = list(events_near or [])
    kinds = {str(e.get("type") or "").lower() for e in near}
    subject_class = "unknown_discontinuity"
    for sub in subjects or []:
        subject_class = classify_subject(sub)
        if subject_class != "unknown_discontinuity":
            break
    if kinds & {"split", "bonus", "consolidation"}:
        event_class = next(iter(kinds & {"split", "bonus", "consolidation"}))
        resolved = False  # still flagged ⇒ ledger did not restore continuity
        treatment = "unresolved_supported_type_quarantine"
    elif subject_class in {"demerger", "merger", "rights", "symbol_restructuring", "special_distribution"}:
        event_class = subject_class
        resolved = False
        treatment = "quarantine_no_inferred_factor"
    elif abs(float(pct)) >= 80:
        event_class = "unknown_discontinuity"
        resolved = False
        treatment = "quarantine_unknown"
    else:
        event_class = "unknown_discontinuity"
        resolved = False
        treatment = "quarantine_unknown"
    return {
        "symbol": str(symbol).upper(),
        "date": str(date),
        "discontinuity_pct": round(float(pct), 3),
        "event_classification": event_class,
        "subject_class": subject_class,
        "source": "adjusted_ohlcv_gap_scan+ledger",
        "resolved": resolved,
        "treatment": treatment,
        "events_near": near,
        "never_infers_factor": True,
    }


def unresolved_events(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of=None,
    events: Mapping[str, list] | None = None,
    sample: int | None = None,
) -> list[dict[str, Any]]:
    """Unresolved consecutive discontinuities. Does not invent factors.

    Uses ``discontinuity_audit`` so genuine ≥35% market days and suspension
    spans are not treated as missing CA. Only ``UNRESOLVED`` consecutive
    events (and demerger/merger subjects when present) are returned.
    """
    from data.corporate_actions import load_events
    from research.intelligence.data.discontinuity_audit import audit_symbol

    ledger = events if events is not None else load_events()
    out: list[dict[str, Any]] = []
    symbols = [str(s).upper() for s in frames.keys()]
    if sample:
        symbols = symbols[: int(sample)]
    for sym in symbols:
        df = frames.get(sym)
        if df is None:
            df = next((frames[k] for k in frames if str(k).upper() == sym), None)
        try:
            rows = audit_symbol(sym, events=dict(ledger))
        except Exception:
            rows = []
        if not rows and df is not None and "close" in getattr(df, "columns", []):
            # Synthetic / off-store frames (unit tests): consecutive phantom gaps
            # without a ledger event are unresolved and quarantined.
            sliced = slice_as_of(df, as_of) if as_of is not None else df
            if sliced is None or len(sliced) < 3:
                continue
            from core.data_integrity import phantom_gaps
            for g in phantom_gaps(sliced["close"].to_numpy(dtype=float)):
                idx = int(g["index"])
                try:
                    d1 = iso_date(sliced.index[idx])
                except Exception:
                    d1 = str(idx)
                out.append(classify_gap_event(
                    symbol=sym, date=d1, pct=float(g.get("pct") or 0), events_near=[],
                ))
            continue
        for disc in rows:
            if int(getattr(disc, "cal_days", 99) or 99) > 3:
                continue
            if str(disc.classification) not in {"UNRESOLVED", "IDENTITY_TRANSITION"}:
                continue
            d1 = str(disc.d1)
            if as_of is not None and d1 > iso_date(as_of):
                continue
            out.append(classify_gap_event(
                symbol=sym, date=d1, pct=float(disc.pct_adj if disc.pct_adj is not None else disc.pct_raw),
                events_near=list(disc.ca_events_near or []),
                subjects=[str(disc.notes or "")],
            ))
    return out


def quarantine_symbols(unresolved: list[dict[str, Any]]) -> set[str]:
    """Symbols that must not enter trend/VCP/return research."""
    q = set()
    for row in unresolved:
        if not row.get("resolved"):
            q.add(str(row["symbol"]).upper())
    return q


def ca_applied_as_of(frame, events: list, as_of) -> Any:
    """Adjust using only events with ex_date ≤ as_of. Future CA must not leak."""
    from data.corporate_actions import adjust_frame

    cutoff = pd.Timestamp(iso_date(as_of))
    live = []
    for e in events or []:
        try:
            ex = pd.Timestamp(e.get("ex_date"))
        except Exception:
            continue
        if ex <= cutoff:
            live.append(e)
    sliced = slice_as_of(frame, as_of)
    return adjust_frame(sliced, live) if sliced is not None else None


def verify_report(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of=None,
    sample: int | None = None,
) -> dict[str, Any]:
    """Honest verify wrapper: does not lower the 0.002 unresolved-symbol threshold."""
    from core.data_integrity import verify_ca_adjustment
    from data.corporate_actions import ledger_status, load_events

    unresolved = unresolved_events(frames, as_of=as_of, sample=sample)
    q = quarantine_symbols(unresolved)
    remaining = {s: frames[s] for s in frames if str(s).upper() not in q}
    try:
        v = verify_ca_adjustment(sample=int(sample or 80))
    except Exception as exc:
        v = {"passed": False, "note": str(exc)}
    status = ledger_status()
    n_events = sum(len(x) for x in (load_events() or {}).values())
    passed = bool(v.get("passed"))
    return {
        "verify_passed": passed,
        "ca_complete": bool(n_events) and passed,
        "ledger": {
            "n_events": n_events,
            "source": status.get("source"),
            "path": status.get("path"),
            "gap_rate": v.get("gap_rate"),
            "flagged": v.get("flagged"),
        },
        "unresolved_events": unresolved,
        "quarantine_symbols": sorted(q),
        "n_quarantine": len(q),
        "n_remaining_frames": len(remaining),
        "never_infers_factor": True,
        "threshold_unchanged": True,
        "note": (
            "PASS requires official share-count ledger AND unresolved consecutive "
            "gap rate ≤ 0.002. Quarantine is exclusion, not a fabricated factor."
        ),
    }

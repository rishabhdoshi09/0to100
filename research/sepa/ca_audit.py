"""Corporate-action audit and quarantine. Never infer a factor from a price gap."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
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

    Scans the **passed** frames in memory. Do not call ``audit_symbol`` here:
    that helper re-walks the entire raw store per name (O(n²)) and is not
    an exhaustive research audit.

    Remaining ≥35% adjacent gaps on the research series (already CA-adjusted
    on official read) are classified with the same discontinuity rules:
    genuine restored continuity is not quarantined; unresolved consecutive
    events and identity transitions are.
    """
    from core.data_integrity import phantom_gaps
    from data.corporate_actions import load_events
    from research.intelligence.data.discontinuity_audit import (
        _CONSEC_CAL_DAYS,
        _ca_near,
        classify_discontinuity,
    )

    ledger = events if events is not None else load_events()
    try:
        from data.security_identity import load_identity_ledger
        id_ledger = load_identity_ledger() or {}
    except Exception:
        id_ledger = {}
    changes_all = list((id_ledger or {}).get("symbol_changes") or [])

    out: list[dict[str, Any]] = []
    symbols = [str(s).upper() for s in frames.keys()]
    if sample:
        symbols = symbols[: int(sample)]
    n = len(symbols)
    for i, sym in enumerate(symbols):
        if i and i % 500 == 0:
            print(f"CA unresolved audit {i}/{n}", flush=True)
        df = frames.get(sym)
        if df is None:
            df = next((frames[k] for k in frames if str(k).upper() == sym), None)
        if df is None or "close" not in getattr(df, "columns", []):
            continue
        sliced = slice_as_of(df, as_of) if as_of is not None else df
        if sliced is None or len(sliced) < 3:
            continue
        close = sliced["close"].to_numpy(dtype=float)
        ev = list(ledger.get(sym, []) or [])
        changes = [
            ch for ch in changes_all
            if str(ch.get("old_symbol") or "").upper() == sym
            or str(ch.get("new_symbol") or "").upper() == sym
        ]
        for g in phantom_gaps(close):
            idx = int(g["index"])
            if idx < 1 or idx >= len(sliced):
                continue
            d0 = sliced.index[idx - 1]
            d1 = sliced.index[idx]
            try:
                cal = int((pd.Timestamp(d1) - pd.Timestamp(d0)).days)
            except Exception:
                cal = 1
            if cal > int(_CONSEC_CAL_DAYS):
                continue
            d1s = iso_date(d1)
            if as_of is not None and d1s > iso_date(as_of):
                continue
            pre = float(sliced["close"].iloc[idx - 1])
            post = float(sliced["close"].iloc[idx])
            pct = float(g.get("pct") or 0)
            ca_near = _ca_near(ev, d0, d1)
            identity_hit = False
            for ch in changes:
                try:
                    when = pd.Timestamp(ch.get("effective_date"))
                    if min(abs((when - pd.Timestamp(d0)).days), abs((when - pd.Timestamp(d1)).days)) <= 5:
                        identity_hit = True
                        break
                except Exception:
                    continue
            disc = classify_discontinuity(
                symbol=sym, d0=d0, d1=d1, cal_days=cal,
                pct_raw=pct, pct_adj=pct,
                pre_raw=pre, post_raw=post, pre_adj=pre, post_adj=post,
                ca_near=ca_near, identity_hit=identity_hit,
            )
            if str(disc.classification) not in {"UNRESOLVED", "IDENTITY_TRANSITION"}:
                continue
            out.append(classify_gap_event(
                symbol=sym, date=d1s, pct=pct,
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
            "gap rate ≤ 0.002. Quarantine is exclusion, not a fabricated factor. "
            "Static quarantine_symbols is an event catalogue, not a historical "
            "as-of membership filter — use CATimeline for causal segments."
        ),
    }


# Longest canonical SEPA lookback (Stage-2 / RS 252) plus one prior close.
CANONICAL_FEATURE_SESSIONS = 252


def _date_ns(value) -> int:
    return int(np.datetime64(iso_date(value), "ns").astype(np.int64))


class CATimeline:
    """Date/segment-aware unresolved CA map. Never a static symbol blacklist.

    For each unresolved event D:
    - observations strictly before D remain valid if lookback and forward
      outcome do not cross D
    - a forward path that includes D is CA_CENSORED_OUTCOME
    - at/after D, indicators may only use bars strictly after D, and the
      name re-enters only after CANONICAL_FEATURE_SESSIONS of clean history
    """

    def __init__(self, events: Sequence[Mapping[str, Any]] | None = None):
        self.rows: list[dict[str, Any]] = []
        self.by_symbol: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for raw in events or []:
            if raw.get("resolved"):
                continue
            sym = str(raw.get("symbol") or "").upper()
            d = str(raw.get("date") or "")
            if not sym or not d:
                continue
            if d in seen.setdefault(sym, set()):
                continue
            seen[sym].add(d)
            row = {
                "symbol": sym,
                "event_date": d,
                "event_classification": raw.get("event_classification") or raw.get("classification") or "unknown_discontinuity",
                "treatment": raw.get("treatment") or "quarantine_no_inferred_factor",
                "discontinuity_pct": raw.get("discontinuity_pct"),
                "never_infers_factor": True,
                "clean_pre_end": None,
                "clean_post_start": None,
            }
            self.rows.append(row)
            self.by_symbol.setdefault(sym, []).append(d)
        for sym in self.by_symbol:
            self.by_symbol[sym] = sorted(set(self.by_symbol[sym]))

    def annotate_calendar(self, calendar: Sequence) -> "CATimeline":
        """Fill clean_pre_end / clean_post_start from an exchange session list."""
        cal = [iso_date(t) for t in calendar]
        loc = {d: i for i, d in enumerate(cal)}
        for row in self.rows:
            d = row["event_date"]
            i = loc.get(d)
            if i is None:
                # Event date may fall on a non-session; bind to first session ≥ D.
                later = [j for j, s in enumerate(cal) if s >= d]
                i = later[0] if later else None
            if i is None:
                continue
            row["clean_pre_end"] = cal[i - 1] if i > 0 else None
            row["clean_post_start"] = cal[i + 1] if i + 1 < len(cal) else None
        return self

    def event_dates(self, symbol: str) -> list[str]:
        return list(self.by_symbol.get(str(symbol).upper(), []))

    def last_event_on_or_before(self, symbol: str, as_of) -> str | None:
        as_s = iso_date(as_of)
        last = None
        for d in self.event_dates(symbol):
            if d <= as_s:
                last = d
            else:
                break
        return last

    def clean_start_index(self, symbol: str, dates_ns: np.ndarray, as_of_ns: int) -> int:
        """First index that may be used for indicators at as_of (0 = full history)."""
        last = None
        for d in self.event_dates(symbol):
            dns = _date_ns(d)
            if dns <= as_of_ns:
                last = dns
            else:
                break
        if last is None:
            return 0
        return int(np.searchsorted(dates_ns, last, side="right"))

    def n_clean_sessions(self, symbol: str, dates_ns: np.ndarray, as_of_ns: int, j: int) -> int:
        start = self.clean_start_index(symbol, dates_ns, as_of_ns)
        return int(j - start + 1) if j >= start else 0

    def lookback_contaminated(
        self, symbol: str, dates_ns: np.ndarray, as_of_ns: int, j: int, lookback: int,
    ) -> bool:
        return self.n_clean_sessions(symbol, dates_ns, as_of_ns, j) < int(lookback)

    def horizon_crosses(self, symbol: str, start_exclusive, end_inclusive) -> bool:
        """True if an unresolved D satisfies start < D ≤ end."""
        lo = iso_date(start_exclusive)
        hi = iso_date(end_inclusive)
        for d in self.event_dates(symbol):
            if lo < d <= hi:
                return True
        return False

    def would_static_quarantine(self, symbol: str) -> bool:
        """Old (incorrect) behaviour: any future unresolved event banned the name forever."""
        return str(symbol).upper() in self.by_symbol

    def to_audit(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows]


def build_timeline(
    unresolved: Sequence[Mapping[str, Any]] | None,
    calendar: Sequence | None = None,
) -> CATimeline:
    tl = CATimeline(unresolved)
    if calendar is not None:
        tl.annotate_calendar(calendar)
    return tl


def ca_research_acceptability(
    *,
    unresolved: Sequence[Mapping[str, Any]],
    exhaustive: bool,
    inferred_factors: bool = False,
    unknown_path_crossings: int = 0,
    future_leak_removed_prior: int = 0,
    audit_persisted: bool,
    contaminated_uncensored: int = 0,
) -> dict[str, Any]:
    """Research-CA gate. Does **not** change global ``ca_complete``."""
    reasons: list[str] = []
    if not exhaustive:
        reasons.append("ca_audit_not_exhaustive")
    if inferred_factors:
        reasons.append("inferred_adjustment_factor")
    if int(unknown_path_crossings) > 0:
        reasons.append(f"unknown_discontinuity_on_eval_path={unknown_path_crossings}")
    if int(contaminated_uncensored) > 0:
        reasons.append(f"contaminated_uncensored={contaminated_uncensored}")
    if int(future_leak_removed_prior) > 0:
        reasons.append(f"future_ca_removed_prior_obs={future_leak_removed_prior}")
    if not audit_persisted:
        reasons.append("audit_not_persisted")
    return {
        "ca_research_acceptable": len(reasons) == 0,
        "reasons": reasons,
        "n_unresolved_enumerated": len(list(unresolved or [])),
        "exhaustive": bool(exhaustive),
        "inferred_factors": bool(inferred_factors),
        "unknown_path_crossings": int(unknown_path_crossings),
        "future_leak_removed_prior": int(future_leak_removed_prior),
        "contaminated_uncensored": int(contaminated_uncensored),
        "audit_persisted": bool(audit_persisted),
        "never_infers_factor": True,
        "does_not_set_ca_complete": True,
    }

"""SEPA-001R data-integrity report — never claim PIT-safe loosely."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from research.sepa.config import CA_POLICY_VERSION
from research.sepa.frames import ca_status, pit_universe


PIT_STRONG = "PIT_STRONG"
PIT_DEGRADED = "PIT_DEGRADED"
PIT_UNVERIFIED = "PIT_UNVERIFIED"

_INFERRED = frozenset({"", "bhav_inferred"})


def _worst(*classes: str) -> str:
    rank = {PIT_STRONG: 0, PIT_DEGRADED: 1, PIT_UNVERIFIED: 2}
    worst = PIT_STRONG
    for c in classes:
        if rank.get(c, 2) > rank.get(worst, 0):
            worst = c
    return worst


def classify_universe_pit(meta: Mapping[str, Any] | None) -> str:
    meta = dict(meta or {})
    source = str(meta.get("universe_source") or meta.get("source") or "")
    complete = bool(meta.get("universe_complete"))
    research_grade = bool(meta.get("research_grade"))
    if not complete:
        return PIT_UNVERIFIED
    src = source.strip()
    if src.startswith("bhav_") or src in _INFERRED:
        return PIT_DEGRADED
    if research_grade and src and src not in _INFERRED:
        return PIT_STRONG
    return PIT_DEGRADED


def classify_ca_pit(ca: Mapping[str, Any] | None) -> str:
    ca = dict(ca or {})
    if bool(ca.get("ca_complete")) and bool(ca.get("verified")):
        return PIT_STRONG
    if int(ca.get("n_events") or 0) > 0:
        return PIT_DEGRADED
    return PIT_UNVERIFIED


def classify_pit(*, universe_meta=None, ca=None) -> str:
    return _worst(classify_universe_pit(universe_meta), classify_ca_pit(ca))


def ca_ledger_fingerprint() -> dict[str, Any]:
    try:
        from data.corporate_actions import events_path, load_events
    except Exception as exc:
        return {"hash": "", "path": "", "error": str(exc)}
    try:
        p = events_path()
    except Exception:
        p = None
    digest = ""
    if p is not None and p.exists():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
    events = {}
    try:
        events = load_events() or {}
    except Exception:
        events = {}
    coverage = []
    for rows in events.values():
        for e in rows:
            try:
                coverage.append(str(e.get("ex_date")))
            except Exception:
                continue
    dates = sorted(d for d in coverage if d and d != "NaT")
    return {
        "hash": digest[:16] if digest else "",
        "sha256": digest,
        "path": str(p) if p is not None else "",
        "n_symbols": len(events),
        "n_events": sum(len(v) for v in events.values()),
        "coverage_start": dates[0] if dates else "",
        "coverage_end": dates[-1] if dates else "",
        "policy": CA_POLICY_VERSION,
    }


def unresolved_gap_symbols(frames: Mapping[str, Any] | None, *, sample: int | None = None) -> list[dict[str, Any]]:
    """Symbols whose RAW (or passed) close still has consecutive-session phantom gaps."""
    from core.data_integrity import phantom_gaps

    flagged: list[dict[str, Any]] = []
    if not frames:
        return flagged
    items = list(frames.items())
    if sample:
        items = items[: int(sample)]
    for sym, df in items:
        if df is None or getattr(df, "empty", True) or "close" not in df.columns:
            continue
        try:
            gaps = phantom_gaps(df["close"].to_numpy(dtype=float))
        except Exception:
            continue
        if gaps:
            flagged.append({
                "symbol": str(sym).upper(),
                "n_gaps": len(gaps),
                "first_pct": gaps[0].get("pct"),
            })
    return flagged


def research_integrity_report(
    *,
    frames: Mapping[str, Any] | None = None,
    as_of=None,
    verify: bool = False,
) -> dict[str, Any]:
    ca = ca_status()
    if verify:
        try:
            from data.corporate_actions import refresh_adjustment_verify
            v = refresh_adjustment_verify(sample=80)
            ca["verified"] = bool(v.get("passed"))
            ca["ca_complete"] = bool(ca.get("n_events")) and bool(v.get("passed"))
            ca["verify_result"] = v
        except Exception as exc:
            ca["verify_error"] = str(exc)
            ca["ca_complete"] = False
    fp = ca_ledger_fingerprint()
    u: dict[str, Any] = {}
    if as_of is not None:
        try:
            u = pit_universe(as_of)
        except Exception as exc:
            u = {"universe_complete": False, "note": str(exc), "symbols": []}
    gaps = unresolved_gap_symbols(frames)
    pit = classify_pit(universe_meta=u, ca=ca)
    if gaps and pit == PIT_STRONG:
        pit = PIT_DEGRADED
    return {
        "pit_class": pit,
        "price_integrity": {
            "source": "official_nse_bhavcopy_on_read",
            "unresolved_gap_symbols": [g["symbol"] for g in gaps],
            "n_unresolved": len(gaps),
            "note": (
                "Unresolved consecutive-session gaps remain — those names must "
                "be excluded from the research book."
                if gaps else
                "No consecutive-session phantom gaps in the passed frames "
                "(does not prove CA completeness)."
            ),
        },
        "ca_integrity": {
            **ca,
            **fp,
            "never_invents": True,
        },
        "universe_integrity": {
            "universe_complete": bool(u.get("universe_complete")),
            "research_grade": bool(u.get("research_grade")),
            "source": u.get("source") or "",
            "n_symbols": len(u.get("symbols") or []),
            "note": u.get("note") or "",
            "class": classify_universe_pit(u),
            "survivorship_claim": (
                "PIT membership is inferred or missing — do not set "
                "survivorship_complete=true as a research claim."
                if classify_universe_pit(u) != PIT_STRONG else
                "Official/operator membership archive in use."
            ),
        },
        "rs_integrity": {
            "formula": "rs_cs_v1 0.40*r63+0.20*r126+0.20*r189+0.20*r252",
            "cross_sectional": True,
            "fail_closed_on_missing_horizon": True,
        },
        "timestamp_integrity": {
            "as_of_slice": "frame.index <= as_of",
            "swing_confirmation": "causal zigzag — confirmed_index, never back-dated",
            "fill": "next session open vs buy-zone; no future-known stop",
        },
        "limitations": _limitations(pit, ca, u, gaps),
        "overall": pit,
    }


def _limitations(pit: str, ca: dict, u: dict, gaps: list) -> list[str]:
    out = []
    if not ca.get("ca_complete"):
        out.append("Corporate-action verification did not pass; prices may contain phantom gaps.")
    if classify_universe_pit(u) != PIT_STRONG:
        out.append("Universe membership is not an official listing archive.")
    if gaps:
        out.append(f"{len(gaps)} symbols still show unresolved consecutive-session discontinuities.")
    if pit != PIT_STRONG:
        out.append(f"Overall PIT class is {pit} — results must not be labelled PIT-safe.")
    return out

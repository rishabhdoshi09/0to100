"""Discontinuity audit — classify large price moves without inventing CA factors.

Root-cause finding (RESEARCH_GRADE remediation):
  ``gap_rate`` historically counted the share of sampled symbols with ANY
  adjacent-bar |return| ≥ 35% on the adjusted series. That conflates:

  • genuine unadjusted / mis-adjusted corporate actions (research defect)
  • sparse trading / suspension / relisting multi-day moves (not single-session CA)
  • identity breaks
  • true one-day market moves that happen to exceed the threshold

This module classifies each candidate and exposes an *unresolved consecutive-
session* rate suitable for RESEARCH_GRADE gating.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.data_integrity import _GAP_PCT, phantom_gaps
from product.plain_language import PlainCard, render_layers

# Adjacent symbol bars within this many calendar days are treated as a
# near-consecutive session pair (weekends/holidays). Larger spans are sparse /
# suspension-class and must NOT be scored as single-session CA failures.
_CONSEC_CAL_DAYS = 3

CLASSES = (
    "SUPPORTED_CA",
    "GENUINE_MARKET_MOVE",
    "SUSPENSION_OR_RELISTING",
    "IDENTITY_TRANSITION",
    "DATA_ERROR",
    "UNRESOLVED",
)

CA_STATUSES = ("VERIFIED", "PARTIAL", "CONFLICT", "MISSING_SOURCE", "UNRESOLVED")


@dataclass
class Discontinuity:
    symbol: str
    d0: str
    d1: str
    cal_days: int
    pct_raw: float
    pct_adj: float | None
    pre_raw: float
    post_raw: float
    pre_adj: float | None
    post_adj: float | None
    classification: str
    ca_status: str
    ca_events_near: list[dict] = field(default_factory=list)
    ratio_hint: float | None = None  # investigative only — never authoritative
    notes: str = ""
    impacts_phase_a5: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _plain_for_class(classification: str, ca_status: str = "") -> PlainCard:
    if classification == "SUPPORTED_CA" or ca_status == "VERIFIED":
        return PlainCard(
            label="Historical price adjustment",
            state="GOOD",
            explanation="A large past price change matches an official corporate action and adjusts cleanly.",
            implication="Safe to treat this stretch of history as continuous for research.",
            technical=f"classification={classification}; ca_status={ca_status}",
            internal_key="discontinuity_class",
            internal_value=classification,
        )
    if classification == "SUSPENSION_OR_RELISTING":
        return PlainCard(
            label="Trading pause in history",
            state="CAUTION",
            explanation=(
                "The stock did not trade for a stretch, so the next recorded price can jump "
                "without that meaning a bad corporate-action adjustment."
            ),
            implication="Not counted as a failed price-adjustment check by itself.",
            technical=f"classification={classification}",
            internal_key="discontinuity_class",
            internal_value=classification,
        )
    if classification == "UNRESOLVED" or ca_status in {"UNRESOLVED", "MISSING_SOURCE", "CONFLICT"}:
        return PlainCard(
            label="Historical price adjustment",
            state="NOT_READY",
            explanation=(
                "We found a large historical price change that may be related to a corporate "
                "action, but we do not yet have enough official evidence to adjust it safely."
            ),
            implication="Do not treat strategy tests on this name as fully trustworthy yet.",
            technical=f"classification=UNRESOLVED; ca_status={ca_status or 'UNRESOLVED'}",
            internal_key="UNRESOLVED_CA",
            internal_value=ca_status or "UNRESOLVED",
        )
    return PlainCard(
        label="Price history note",
        state="CAUTION",
        explanation=f"Classified as {classification.replace('_', ' ').title()}.",
        implication="Review technical details before relying on this stretch.",
        technical=f"classification={classification}; ca_status={ca_status}",
        internal_key="discontinuity_class",
        internal_value=classification,
    )


def phase_a5_symbols() -> set[str]:
    p = Path(__file__).resolve().parents[3] / "logs" / "phase_a5" / "sector_map.json"
    if not p.exists():
        return set()
    try:
        return {str(k).upper() for k in json.loads(p.read_text()).keys()}
    except Exception:
        return set()


def _raw_frame(symbol: str):
    from data import bhavcopy_store as BS
    for sym, df in BS.iter_raw_frames():
        if sym == symbol.upper():
            return df
    return None


def _adj_prices(df, events):
    from data.corporate_actions import adjust_frame
    if not events:
        return df
    return adjust_frame(df, events, copy=True)


def _ca_near(events: list, d0, d1, window: int = 5) -> list[dict]:
    out = []
    t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
    for e in events or []:
        ex = pd.Timestamp(e["ex_date"])
        if min(abs((ex - t0).days), abs((ex - t1).days)) <= window:
            out.append({
                "ex_date": str(ex.date()),
                "factor": float(e["factor"]),
                "type": str(e.get("type")),
            })
    return out


def classify_discontinuity(
    *,
    symbol: str,
    d0,
    d1,
    cal_days: int,
    pct_raw: float,
    pct_adj: float | None,
    pre_raw: float,
    post_raw: float,
    pre_adj: float | None,
    post_adj: float | None,
    ca_near: list[dict],
    identity_hit: bool = False,
    threshold: float = _GAP_PCT,
) -> Discontinuity:
    ratio_hint = round(pre_raw / post_raw, 4) if post_raw else None
    a5 = symbol.upper() in phase_a5_symbols()
    notes = []

    # Sparse / suspension: not a single-session CA fingerprint
    if cal_days > _CONSEC_CAL_DAYS:
        return Discontinuity(
            symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
            d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
            pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
            pre_adj=pre_adj, post_adj=post_adj,
            classification="SUSPENSION_OR_RELISTING",
            ca_status="UNRESOLVED" if ca_near else "MISSING_SOURCE",
            ca_events_near=ca_near, ratio_hint=ratio_hint,
            notes="Adjacent bars span >3 calendar days — sparse/suspension class.",
            impacts_phase_a5=a5,
        )

    if identity_hit:
        return Discontinuity(
            symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
            d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
            pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
            pre_adj=pre_adj, post_adj=post_adj,
            classification="IDENTITY_TRANSITION",
            ca_status="UNRESOLVED", ca_events_near=ca_near, ratio_hint=ratio_hint,
            notes="Symbol/identity transition evidence near this date.",
            impacts_phase_a5=a5,
        )

    adj_ok = pct_adj is not None and abs(pct_adj) < threshold
    if ca_near and adj_ok:
        return Discontinuity(
            symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
            d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
            pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
            pre_adj=pre_adj, post_adj=post_adj,
            classification="SUPPORTED_CA", ca_status="VERIFIED",
            ca_events_near=ca_near, ratio_hint=ratio_hint,
            notes="Official CA near ex-date; adjusted continuity restored.",
            impacts_phase_a5=a5,
        )

    if ca_near and not adj_ok:
        # Same-day multi-event under-adjustment is a common PARTIAL/CONFLICT case
        status = "PARTIAL" if len(ca_near) >= 1 else "CONFLICT"
        notes.append("Official CA present but adjusted series still discontinuous.")
        if ratio_hint and ca_near:
            prod = 1.0
            for e in ca_near:
                prod *= float(e["factor"])
            if abs(prod - (ratio_hint or 0)) > 0.5:
                status = "CONFLICT"
                notes.append(
                    f"Ledger factor product={prod:.4f} vs price ratio_hint={ratio_hint} "
                    "(hint is investigative only)."
                )
        return Discontinuity(
            symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
            d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
            pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
            pre_adj=pre_adj, post_adj=post_adj,
            classification="UNRESOLVED", ca_status=status,
            ca_events_near=ca_near, ratio_hint=ratio_hint,
            notes=" ".join(notes), impacts_phase_a5=a5,
        )

    # No CA near; consecutive session; still huge after "adjustment" (none)
    if adj_ok:
        return Discontinuity(
            symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
            d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
            pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
            pre_adj=pre_adj, post_adj=post_adj,
            classification="GENUINE_MARKET_MOVE", ca_status="MISSING_SOURCE",
            ca_events_near=ca_near, ratio_hint=ratio_hint,
            notes="Large raw move but adjusted continuity OK / below threshold after adj.",
            impacts_phase_a5=a5,
        )

    return Discontinuity(
        symbol=symbol.upper(), d0=str(pd.Timestamp(d0).date()),
        d1=str(pd.Timestamp(d1).date()), cal_days=cal_days,
        pct_raw=pct_raw, pct_adj=pct_adj, pre_raw=pre_raw, post_raw=post_raw,
        pre_adj=pre_adj, post_adj=post_adj,
        classification="UNRESOLVED", ca_status="MISSING_SOURCE",
        ca_events_near=ca_near, ratio_hint=ratio_hint,
        notes=(
            "Consecutive-session discontinuity without official parseable CA factor. "
            "ratio_hint is investigative only and must not become an authoritative factor."
        ),
        impacts_phase_a5=a5,
    )


def audit_symbol(symbol: str, *, events: dict | None = None, threshold: float = _GAP_PCT) -> list[Discontinuity]:
    from data.corporate_actions import load_events
    from data.security_identity import load_identity_ledger

    events = events if events is not None else load_events()
    sym = symbol.upper()
    rdf = _raw_frame(sym)
    if rdf is None or rdf.empty:
        return []
    ev = events.get(sym, [])
    adf = _adj_prices(rdf, ev)
    raw_gaps = phantom_gaps(rdf["close"].to_numpy(dtype=float), threshold)
    # identity transitions near dates
    id_ledger = load_identity_ledger()
    changes = []
    for ch in (id_ledger or {}).get("symbol_changes") or []:
        if ch.get("old_symbol") == sym or ch.get("new_symbol") == sym:
            changes.append(ch)

    out: list[Discontinuity] = []
    for g in raw_gaps:
        i = g["index"]
        d0, d1 = rdf.index[i - 1], rdf.index[i]
        cal = int((pd.Timestamp(d1) - pd.Timestamp(d0)).days)
        pre = float(rdf["close"].iloc[i - 1])
        post = float(rdf["close"].iloc[i])
        try:
            apre = float(adf.loc[d0, "close"])
            apost = float(adf.loc[d1, "close"])
            pct_adj = round((apost - apre) / apre * 100.0, 1) if apre else None
        except Exception:
            apre = apost = pct_adj = None
        ca_near = _ca_near(ev, d0, d1)
        identity_hit = False
        for ch in changes:
            try:
                when = pd.Timestamp(ch.get("effective_date"))
                if min(abs((when - pd.Timestamp(d0)).days), abs((when - pd.Timestamp(d1)).days)) <= 5:
                    identity_hit = True
            except Exception:
                pass
        out.append(classify_discontinuity(
            symbol=sym, d0=d0, d1=d1, cal_days=cal,
            pct_raw=float(g["pct"]), pct_adj=pct_adj,
            pre_raw=pre, post_raw=post, pre_adj=apre, post_adj=apost,
            ca_near=ca_near, identity_hit=identity_hit, threshold=threshold,
        ))
    return out


def audit_universe(symbols: Iterable[str] | None = None, *, sample: int | None = 400) -> dict[str, Any]:
    from data.bhavcopy_runtime import ensure_loaded
    from data import bhavcopy_store as BS
    from data.corporate_actions import load_events
    from data.bhavcopy_store import reload_corporate_actions

    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    events = load_events()
    if symbols is None:
        syms = BS.store_symbols()
        if sample is not None:
            syms = syms[: int(sample)]
    else:
        syms = [str(s).upper() for s in symbols]

    rows: list[Discontinuity] = []
    for s in syms:
        rows.extend(audit_symbol(s, events=events))

    by_class: dict[str, int] = {c: 0 for c in CLASSES}
    by_ca: dict[str, int] = {c: 0 for c in CA_STATUSES}
    for r in rows:
        by_class[r.classification] = by_class.get(r.classification, 0) + 1
        by_ca[r.ca_status] = by_ca.get(r.ca_status, 0) + 1

    consec = [r for r in rows if r.cal_days <= _CONSEC_CAL_DAYS]
    unresolved_consec = [
        r for r in consec
        if r.classification == "UNRESOLVED" or r.ca_status in {"UNRESOLVED", "CONFLICT", "PARTIAL", "MISSING_SOURCE"}
        and r.classification not in {"SUPPORTED_CA", "SUSPENSION_OR_RELISTING", "GENUINE_MARKET_MOVE"}
    ]
    # Tighten: unresolved consecutive = classification UNRESOLVED only
    unresolved_consec = [r for r in consec if r.classification == "UNRESOLVED"]
    verified = [r for r in rows if r.ca_status == "VERIFIED"]
    a5_unresolved = [r for r in unresolved_consec if r.impacts_phase_a5]

    # Symbol-level unresolved rate among sampled symbols (research quality measure)
    syms_with_unresolved = {r.symbol for r in unresolved_consec}
    checked = len(syms)
    unresolved_symbol_rate = (len(syms_with_unresolved) / checked) if checked else 1.0

    # Impact ranking
    def impact_key(r: Discontinuity):
        return (
            0 if r.impacts_phase_a5 else 1,
            0 if r.cal_days <= _CONSEC_CAL_DAYS else 1,
            -abs(r.pct_raw),
            r.symbol,
        )

    ranked = sorted(unresolved_consec, key=impact_key)

    return {
        "threshold_pct": _GAP_PCT,
        "consec_calendar_days": _CONSEC_CAL_DAYS,
        "symbols_checked": checked,
        "discontinuities": len(rows),
        "by_class": by_class,
        "by_ca_status": by_ca,
        "consecutive_events": len(consec),
        "sparse_or_suspension_events": by_class.get("SUSPENSION_OR_RELISTING", 0),
        "verified_ca_events": len(verified),
        "unresolved_consecutive_events": len(unresolved_consec),
        "unresolved_symbol_rate": round(unresolved_symbol_rate, 4),
        "legacy_any_large_move_symbol_rate_note": (
            "Previous gap_rate counted symbols with any ≥35% adjacent-bar move, "
            "including suspension/sparse spans. That inflated the failure rate."
        ),
        "phase_a5_unresolved": [r.as_dict() for r in a5_unresolved],
        "top_unresolved": [r.as_dict() for r in ranked[:40]],
        "plain": {
            "unresolved": render_layers(_plain_for_class("UNRESOLVED")),
            "supported": render_layers(_plain_for_class("SUPPORTED_CA", "VERIFIED")),
            "suspension": render_layers(_plain_for_class("SUSPENSION_OR_RELISTING")),
        },
    }


def verification_trace(disc: Discontinuity) -> dict:
    """R2-style trace for one discontinuity (raw preserved; hint non-authoritative)."""
    card = _plain_for_class(disc.classification, disc.ca_status)
    return {
        "security_symbol_as_of": disc.symbol,
        "event_type": ",".join(e.get("type", "") for e in disc.ca_events_near) or None,
        "ex_dates": [e.get("ex_date") for e in disc.ca_events_near],
        "ratios": [e.get("factor") for e in disc.ca_events_near],
        "raw_pre_event_price": disc.pre_raw,
        "raw_post_event_price": disc.post_raw,
        "adjusted_pre": disc.pre_adj,
        "adjusted_post": disc.post_adj,
        "expected_adjustment_factors": disc.ca_events_near,
        "actual_adjusted_continuity_pct": disc.pct_adj,
        "verification_status": disc.ca_status,
        "classification": disc.classification,
        "ratio_hint_investigative_only": disc.ratio_hint,
        "notes": disc.notes,
        "user_facing": render_layers(card),
    }

"""D1 — classify securities for research-eligible (scoped) certification.

Global trust remains OPERATIONAL_ONLY. Classification never invents CA ratios
or symbol lineage. Unknown remains unknown.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.bhavcopy_runtime import ensure_loaded
from data.bhavcopy_store import get_ohlcv, reload_corporate_actions, symbol_date_spans
from data.corporate_actions import load_events
from data.nse_ca_ingest import ADJUSTMENT_POLICY_VERSION
from data.security_identity import load_identity_ledger, resolve_as_of
from data.universe_history import history_path, ledger_status as universe_ledger_status
from research.intelligence.data.discontinuity_audit import _CONSEC_CAL_DAYS, audit_symbol
from research.phase_a5.scoped_certification import FROZEN_PANEL

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "logs" / "research_expansion"

# Research window for expanded certification (trustworthy official bhav span).
WINDOW_START = "2020-01-01"
WINDOW_END = "2026-08-11"
MID_ASOF = "2023-06-15"

# Liquid / history gates for CERTIFIABLE (defensible liquid NSE panel).
MIN_SESSIONS_CERT = 1000
MIN_SESSIONS_PARTIAL = 400
MIN_MEDIAN_VOLUME = 50_000
# Soft liquidity for PARTIAL (still identity/CA clean but thinner).
MIN_MEDIAN_VOLUME_PARTIAL = 10_000

CLASSES = (
    "CERTIFIABLE",
    "PARTIAL",
    "BLOCKED_IDENTITY",
    "BLOCKED_CA",
    "BLOCKED_UNIVERSE",
    "INSUFFICIENT_HISTORY",
    "OTHER",
)


@dataclass
class SecurityClass:
    symbol: str
    classification: str
    security_id: str | None = None
    isin: str | None = None
    series: str | None = None
    listing_date: str | None = None
    delisting_date: str | None = None
    sessions: int = 0
    first_session: str | None = None
    last_session: str | None = None
    median_volume: float | None = None
    unresolved_consecutive: int = 0
    ca_events: int = 0
    reasons: list[str] = field(default_factory=list)
    in_prior_29: bool = False
    quality_status: str = "UNKNOWN"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassificationResult:
    generated_at: str
    window_start: str
    window_end: str
    global_trust_class: str
    n_store_symbols: int
    n_sessions_store: int
    counts: dict[str, int]
    securities: list[dict]
    certifiable_symbols: list[str]
    partial_symbols: list[str]
    hashes: dict[str, str | None]
    git_sha: str
    notes: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_index(ledger: dict) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in ledger.get("securities") or []:
        sym = str(r.get("symbol") or "").upper()
        if sym:
            by.setdefault(sym, []).append(r)
    return by


def _universe_index() -> dict[str, dict]:
    p = history_path()
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {str(r.get("symbol") or "").upper(): r for r in (raw.get("rows") or [])}


def _median_volume(sym: str) -> float | None:
    df = get_ohlcv(sym)
    if df is None or df.empty or "volume" not in df.columns:
        return None
    try:
        return float(df["volume"].median())
    except Exception:
        return None


def _unresolved_consecutive(sym: str, events: dict, *, start: str, end: str) -> int:
    n = 0
    for h in audit_symbol(sym, events=events):
        if h.cal_days > _CONSEC_CAL_DAYS:
            continue
        if not (start <= h.d0[:10] <= end or start <= h.d1[:10] <= end):
            continue
        if h.classification == "UNRESOLVED":
            n += 1
    return n


def classify_universe(
    *,
    run_ca_audit: bool = True,
    progress_every: int = 100,
) -> ClassificationResult:
    """Classify every store symbol into research eligibility buckets."""
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()

    spans = symbol_date_spans()
    id_ledger = load_identity_ledger()
    by_id = _identity_index(id_ledger)
    by_uni = _universe_index()
    events = load_events()
    prior29 = {s.upper() for s in FROZEN_PANEL}

    # Unique session count from longest-history names
    n_sessions_store = max((s["sessions"] for s in spans.values()), default=0)

    rows: list[SecurityClass] = []
    # Pass 1 — structural gates (no CA audit yet)
    ca_candidates: list[SecurityClass] = []

    for sym, span in sorted(spans.items()):
        sessions = int(span.get("sessions") or 0)
        first = str(span["first"]) if span.get("first") else None
        last = str(span["last"]) if span.get("last") else None
        id_rows = by_id.get(sym) or []
        uni = by_uni.get(sym)
        reasons: list[str] = []
        classification = None

        # Identity
        openish = [
            m for m in id_rows
            if not m.get("valid_to") and not m.get("delisting_date")
        ]
        sec = id_rows[0] if id_rows else None
        if len(openish) > 1:
            classification = "BLOCKED_IDENTITY"
            reasons.append(f"multiple_open_identity_rows={len(openish)}")
        elif not sec:
            classification = "BLOCKED_IDENTITY"
            reasons.append("missing_identity_ledger_row")
        elif not sec.get("isin") or not sec.get("security_id"):
            classification = "BLOCKED_IDENTITY"
            reasons.append("missing_isin_or_security_id")
        else:
            res = resolve_as_of(sym, MID_ASOF, id_ledger)
            if res.get("status") != "OK":
                # IPO after mid-window can still be PARTIAL later; treat as identity soft fail
                if res.get("status") == "NOT_YET_LISTED":
                    reasons.append(f"resolve_mid={res.get('status')}")
                else:
                    classification = "BLOCKED_IDENTITY"
                    reasons.append(f"resolve_mid={res.get('status')}")

        # Universe
        if classification is None:
            if not uni:
                classification = "BLOCKED_UNIVERSE"
                reasons.append("missing_universe_membership_row")
            else:
                listed = uni.get("listed")
                delisted = uni.get("delisted")
                if listed is None:
                    classification = "BLOCKED_UNIVERSE"
                    reasons.append("universe_listed_date_unknown")
                elif delisted and str(delisted)[:10] <= WINDOW_START:
                    classification = "BLOCKED_UNIVERSE"
                    reasons.append("delisted_before_window")

        # History
        if classification is None:
            if sessions < MIN_SESSIONS_PARTIAL:
                classification = "INSUFFICIENT_HISTORY"
                reasons.append(f"sessions={sessions}<{MIN_SESSIONS_PARTIAL}")

        sc = SecurityClass(
            symbol=sym,
            classification=classification or "PENDING_CA",
            security_id=(sec or {}).get("security_id"),
            isin=(sec or {}).get("isin"),
            series=(sec or {}).get("series"),
            listing_date=(
                (sec or {}).get("listing_date")
                or (uni or {}).get("listed")
            ),
            delisting_date=(
                (sec or {}).get("delisting_date")
                or (uni or {}).get("delisted")
            ),
            sessions=sessions,
            first_session=first,
            last_session=last,
            reasons=reasons,
            in_prior_29=sym in prior29,
            quality_status="PENDING",
        )
        if classification is not None:
            sc.quality_status = classification
            rows.append(sc)
        else:
            ca_candidates.append(sc)

    # Pass 2 — CA + liquidity for candidates
    if run_ca_audit:
        for i, sc in enumerate(ca_candidates, 1):
            med = _median_volume(sc.symbol)
            sc.median_volume = med
            sc.ca_events = len(events.get(sc.symbol) or [])
            unr = _unresolved_consecutive(
                sc.symbol, events, start=WINDOW_START, end=WINDOW_END
            )
            sc.unresolved_consecutive = unr
            if unr > 0:
                sc.classification = "BLOCKED_CA"
                sc.reasons.append(f"unresolved_consecutive={unr}")
                sc.quality_status = "BLOCKED_CA"
            else:
                # Liquidity + depth → CERTIFIABLE vs PARTIAL
                if (
                    sc.sessions >= MIN_SESSIONS_CERT
                    and med is not None
                    and med >= MIN_MEDIAN_VOLUME
                    and (sc.last_session or "") >= "2026-01-01"
                ):
                    sc.classification = "CERTIFIABLE"
                    sc.quality_status = "SCOPED_RESEARCH_READY"
                    sc.reasons.append("identity+universe+ca_clean+liquid+deep_history")
                elif (
                    sc.sessions >= MIN_SESSIONS_PARTIAL
                    and med is not None
                    and med >= MIN_MEDIAN_VOLUME_PARTIAL
                ):
                    sc.classification = "PARTIAL"
                    sc.quality_status = "PARTIAL_SCOPED"
                    sc.reasons.append("ca_clean_but_thinner_liquidity_or_history")
                elif sc.sessions >= MIN_SESSIONS_PARTIAL and (med is None or med < MIN_MEDIAN_VOLUME_PARTIAL):
                    sc.classification = "OTHER"
                    sc.quality_status = "ILLIQUID"
                    sc.reasons.append("ca_clean_but_illiquid")
                else:
                    sc.classification = "OTHER"
                    sc.quality_status = "OTHER"
                    sc.reasons.append("passed_structural_gates_but_not_liquid_deep")
            rows.append(sc)
            if progress_every and i % progress_every == 0:
                print(f"ca_audit {i}/{len(ca_candidates)}", flush=True)
    else:
        for sc in ca_candidates:
            sc.classification = "PARTIAL"
            sc.quality_status = "CA_AUDIT_SKIPPED"
            sc.reasons.append("ca_audit_skipped")
            rows.append(sc)

    counts = {c: 0 for c in CLASSES}
    for r in rows:
        counts[r.classification] = counts.get(r.classification, 0) + 1

    certifiable = sorted(r.symbol for r in rows if r.classification == "CERTIFIABLE")
    partial = sorted(r.symbol for r in rows if r.classification == "PARTIAL")

    # Ensure prior 29 that are clean stay visible
    missing_29 = [s for s in FROZEN_PANEL if s not in certifiable and s not in partial]
    notes = [
        "Global trust remains OPERATIONAL_ONLY; CERTIFIABLE is protocol-scoped eligibility only.",
        "CA ratios are never inferred from price jumps.",
        "Symbol lineage is never guessed.",
        f"Window {WINDOW_START}→{WINDOW_END}; mid resolve {MID_ASOF}.",
        f"CERTIFIABLE gates: sessions>={MIN_SESSIONS_CERT}, median_volume>={MIN_MEDIAN_VOLUME}, "
        f"unresolved_consecutive=0, ISIN+security_id, universe listed date known.",
        f"Prior-29 absent from CERTIFIABLE/PARTIAL: {missing_29 or 'none'}.",
        f"Universe ledger: {universe_ledger_status().get('source')}",
        f"Adjustment policy: {ADJUSTMENT_POLICY_VERSION}",
    ]

    result = ClassificationResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        global_trust_class="OPERATIONAL_ONLY",
        n_store_symbols=len(spans),
        n_sessions_store=n_sessions_store,
        counts=counts,
        securities=[r.as_dict() for r in sorted(rows, key=lambda x: x.symbol)],
        certifiable_symbols=certifiable,
        partial_symbols=partial,
        hashes={
            "ca_events": _file_sha(REPO_ROOT / "logs" / "ca_events.json"),
            "security_identity": _file_sha(REPO_ROOT / "logs" / "security_identity.json"),
            "universe_history": _file_sha(REPO_ROOT / "logs" / "universe_history.json"),
        },
        git_sha=_git_sha(),
        notes=notes,
    )
    return result


def write_classification(result: ClassificationResult, path: Path | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (OUT_DIR / "classification.json")
    # Full file can be large — write summary + paths
    summary = {
        k: v for k, v in result.as_dict().items() if k != "securities"
    }
    summary["n_securities_rows"] = len(result.securities)
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    full = OUT_DIR / "classification_full.json"
    full.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    # Compact CSV-like list for certifiable
    (OUT_DIR / "certifiable_symbols.json").write_text(
        json.dumps({
            "window": [result.window_start, result.window_end],
            "n": len(result.certifiable_symbols),
            "symbols": result.certifiable_symbols,
            "partial_n": len(result.partial_symbols),
            "partial_symbols": result.partial_symbols,
            "counts": result.counts,
        }, indent=2),
        encoding="utf-8",
    )
    return path

"""D7 — commit immutable expanded research snapshot via existing SnapshotStore."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.bhavcopy_runtime import ensure_loaded
from data.bhavcopy_store import get_ohlcv, reload_corporate_actions
from data.nse_ca_ingest import ADJUSTMENT_POLICY_VERSION
from research.data_expansion.classify import (
    WINDOW_END,
    WINDOW_START,
    ClassificationResult,
)
from research.intelligence.data.snapshot_store import SnapshotStore
from research.phase_a5.scoped_certification import FROZEN_PANEL

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAP_ROOT = REPO_ROOT / "logs" / "research_expansion" / "snapshots"
VALIDATOR_VERSION = "research_data_expansion.scoped.v1"
PROTOCOL_VERSION = "QUANTTERM_RESEARCH_DATA_EXPANSION@2026-08-11"


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


def build_panel_closes(
    symbols: list[str],
    *,
    start: str = WINDOW_START,
    end: str = WINDOW_END,
    how: str = "outer",
) -> pd.DataFrame:
    """Adjusted closes for research-eligible symbols.

    Uses outer join (not dropna-any) so breadth is preserved; callers must
    handle missingness PIT-honestly. Raw observations stay immutable in bhav.
    """
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    cols: dict[str, pd.Series] = {}
    for sym in symbols:
        df = get_ohlcv(sym)
        if df is None or df.empty:
            continue
        sub = df.loc[
            (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end)),
            "close",
        ]
        if len(sub) == 0:
            continue
        cols[sym] = sub
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols).sort_index()
    if how == "inner_prior29_overlap":
        # Not used for expanded; keep for tests
        return panel.dropna(how="any")
    return panel


def commit_expanded_snapshot(
    symbols: list[str],
    *,
    classification: ClassificationResult | dict | None = None,
    security_ids: dict[str, str] | None = None,
    parent_id: str = "a7a9828ec37e09e4",
    scope_name: str = "EXPANDED_CERTIFIABLE_LIQUID",
    start: str = WINDOW_START,
    end: str = WINDOW_END,
) -> dict[str, Any]:
    """Commit scoped immutable snapshot for the expanded certifiable set."""
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    closes = build_panel_closes(symbols, start=start, end=end)
    if closes.empty:
        raise ValueError("empty panel — cannot commit snapshot")

    store = SnapshotStore(SNAP_ROOT)
    rows = []
    for sym in closes.columns:
        raw = get_ohlcv(sym)
        series = closes[sym].dropna()
        for ts, px in series.items():
            c = float(px)
            if raw is not None and ts in raw.index:
                bar = raw.loc[ts]
                o = float(bar["open"])
                h = float(bar["high"])
                l = float(bar["low"])
                c = float(bar["close"])
                vol = int(float(bar.get("volume") or 0))
            else:
                o = h = l = c
                vol = 0
            d = pd.Timestamp(ts).strftime("%Y-%m-%d")
            rows.append((sym, d, o, h, l, c, vol, "EQ"))

    # Equal-weight panel index over available names each day (not a traded index)
    ew = closes.mean(axis=1, skipna=True)
    index_rows = []
    prev = float(ew.iloc[0])
    for ts, px in ew.items():
        d = pd.Timestamp(ts).strftime("%Y-%m-%d")
        c = float(px)
        index_rows.append(("EXPANDED_PANEL_EW", d, prev, max(prev, c), min(prev, c), c))
        prev = c

    # Also attach official Nifty 50 / India VIX when available (research context)
    try:
        from data.index_store import get_index_ohlcv

        for ticker, name in (("^NSEI", "Nifty 50"), ("^INDIAVIX", "India VIX")):
            df = get_index_ohlcv(ticker)
            if df is None or df.empty:
                continue
            sub = df.loc[
                (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
            ]
            for ts, bar in sub.iterrows():
                d = pd.Timestamp(ts).strftime("%Y-%m-%d")
                o = float(bar.get("Open", bar.get("open", bar.get("Close", 0))))
                h = float(bar.get("High", bar.get("high", o)))
                l = float(bar.get("Low", bar.get("low", o)))
                c = float(bar.get("Close", bar.get("close", o)))
                index_rows.append((name, d, o, h, l, c))
    except Exception:
        pass

    cls = classification.as_dict() if hasattr(classification, "as_dict") else (classification or {})
    sec_ids = security_ids or {}
    if not sec_ids and cls.get("securities"):
        sec_ids = {
            r["symbol"]: r.get("security_id")
            for r in cls["securities"]
            if r.get("symbol") in symbols and r.get("security_id")
        }

    hashes = {
        "ca_events": _file_sha(REPO_ROOT / "logs" / "ca_events.json"),
        "security_identity": _file_sha(REPO_ROOT / "logs" / "security_identity.json"),
        "universe_history": _file_sha(REPO_ROOT / "logs" / "universe_history.json"),
        "prior_29_snapshot": "a7a9828ec37e09e4",
    }
    if cls.get("hashes"):
        hashes.update({f"classification_{k}": v for k, v in cls["hashes"].items()})

    # Coverage stats
    avail = closes.notna().sum(axis=1)
    extra = {
        "scope": scope_name,
        "global_trust_class": "OPERATIONAL_ONLY",
        "trust_class": "OPERATIONAL_ONLY",
        "research_grade": False,
        "scoped_certification": "SCOPED_RESEARCH_READY",
        "scoped_eligible_for_scientific_rerun": True,
        "panel": list(symbols),
        "security_ids": sec_ids,
        "n_symbols": int(closes.shape[1]),
        "n_sessions": int(len(closes)),
        "date_start": start,
        "date_end": end,
        "parent_snapshot_id": parent_id,
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "nse_bhav_adjusted_on_read",
        "hashes": hashes,
        "sector_history": "NOT_RESEARCH_READY",
        "pit_sector_history": False,
        "prior_29_panel": list(FROZEN_PANEL),
        "prior_29_in_panel": sorted(set(symbols) & set(FROZEN_PANEL)),
        "coverage": {
            "median_names_per_session": float(avail.median()) if len(avail) else 0.0,
            "min_names_per_session": int(avail.min()) if len(avail) else 0,
            "max_names_per_session": int(avail.max()) if len(avail) else 0,
            "total_security_sessions": int(closes.notna().sum().sum()),
        },
        "do_not_rerun_automatically": True,
        "plain_language": {
            "label": "Expanded research stock group",
            "explanation": (
                "This larger group of stocks passed the historical data checks "
                "required for serious research on price history."
            ),
            "implication": (
                "QuantTerm can test ideas that need more stocks and more years "
                "than the original 29-name panel, without claiming the whole "
                "market database is fully certified."
            ),
            "technical": "SCOPED_RESEARCH_READY; global_trust=OPERATIONAL_ONLY",
        },
    }
    sid = store.commit_snapshot(rows, index_rows=index_rows, parent_id=parent_id, extra_manifest=extra)
    ok, fails = store.verify_snapshot(sid)
    manifest_path = SNAP_ROOT / sid / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "snapshot_id": sid,
        "root": str(SNAP_ROOT),
        "verify_ok": ok,
        "verify_fails": fails,
        "n_symbols": int(closes.shape[1]),
        "n_sessions": int(len(closes)),
        "date_range": [str(closes.index.min().date()), str(closes.index.max().date())],
        "coverage": extra["coverage"],
        "manifest": manifest,
    }


def build_expanded_snapshot(classification: ClassificationResult) -> dict[str, Any]:
    symbols = list(classification.certifiable_symbols)
    if not symbols:
        raise ValueError("no CERTIFIABLE symbols")
    # Always include any prior-29 that remain certifiable; already in list if clean
    return commit_expanded_snapshot(symbols, classification=classification)

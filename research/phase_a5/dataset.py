"""Phase A.5 dataset loaders — exploratory panel + certified scoped snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence.data.pit_contract import PitContract

ROOT = Path(__file__).resolve().parents[2] / "logs" / "phase_a5"
CLOSES_CSV = ROOT / "exploratory_closes.csv"
SECTOR_JSON = ROOT / "sector_map.json"
MANIFEST_JSON = ROOT / "dataset_manifest.json"
SNAPSHOT_ROOT = ROOT / "snapshots"
SCOPED_SNAPSHOT_ROOT = (
    Path(__file__).resolve().parents[2] / "logs" / "phase_a5_scoped" / "snapshots"
)
CERTIFIED_SNAPSHOT_ID = "a7a9828ec37e09e4"


def load_manifest() -> dict:
    if not MANIFEST_JSON.exists():
        return {
            "trust_class": "MISSING",
            "research_tier": "OPERATIONAL_ONLY",
            "research_grade": False,
            "note": "Phase A.5 exploratory panel not materialised",
        }
    m = json.loads(MANIFEST_JSON.read_text())
    m["research_grade"] = False  # hard invariant for yfinance/display sources
    return m


def load_closes() -> pd.DataFrame:
    if not CLOSES_CSV.exists():
        raise FileNotFoundError(
            f"missing {CLOSES_CSV} — materialise exploratory panel first"
        )
    df = pd.read_csv(CLOSES_CSV, index_col=0, parse_dates=True)
    df.columns = [str(c).upper() for c in df.columns]
    return df.sort_index()


def load_sectors() -> dict[str, str]:
    if not SECTOR_JSON.exists():
        return {}
    raw = json.loads(SECTOR_JSON.read_text())
    return {str(k).upper(): str(v) for k, v in raw.items()}


def load_certified_snapshot(
    snapshot_id: str = CERTIFIED_SNAPSHOT_ID,
    *,
    root: Path | None = None,
) -> tuple[str, PitContract, dict, pd.DataFrame]:
    """Load the immutable Phase A.5 scoped certification snapshot only.

    Does not read the global OPERATIONAL_ONLY store outside this snapshot.
    """
    store_root = root or SCOPED_SNAPSHOT_ROOT
    store = SnapshotStore(store_root)
    ok, fails = store.verify_snapshot(snapshot_id)
    if not ok:
        raise ValueError(f"certified snapshot {snapshot_id} failed verify: {fails}")
    snap = store.open_snapshot(snapshot_id)
    manifest = dict(snap.manifest)
    if manifest.get("snapshot_id") != snapshot_id:
        raise ValueError("snapshot_id mismatch in manifest")
    if manifest.get("scoped_certification") != "READY_FOR_SCIENTIFIC_RERUN":
        raise ValueError(
            f"snapshot {snapshot_id} is not scoped-certified for scientific rerun"
        )

    # Close panel from snapshot equity rows (content-addressed; no global mix-in)
    by_sym: dict[str, dict[str, float]] = {}
    for r in snap._equity:
        sym = str(r["symbol"]).upper()
        by_sym.setdefault(sym, {})[r["date"]] = float(r["close"])
    closes = pd.DataFrame({sym: pd.Series(vals) for sym, vals in by_sym.items()})
    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="any")
    closes.columns = [str(c).upper() for c in closes.columns]

    pit = PitContract.from_store(store, snapshot_id)
    # PIT smoke: as-of mid must not leak future bars
    if len(closes.index) >= 2:
        mid = closes.index[len(closes) // 2].strftime("%Y-%m-%d")
        sample = list(closes.columns)[0]
        read = pit.as_of("bars", when=mid, symbol=sample)
        if not read.usable:
            raise ValueError(f"PitContract unusable on certified snapshot: {read}")
        if any(b.date > mid for b in read.data):
            raise ValueError("PIT violation: future bar leaked from certified snapshot")

    manifest.setdefault("scoped_eligible_for_scientific_rerun", True)
    manifest.setdefault("scope", "PHASE_A5_FROZEN_PROTOCOL")
    manifest["snapshot_id"] = snapshot_id
    return snapshot_id, pit, manifest, closes


def commit_exploratory_snapshot(closes: pd.DataFrame | None = None) -> tuple[str, PitContract, dict]:
    """Commit closes into SnapshotStore so PitContract can serve bars.

    Manifest is stamped DISPLAY_ONLY / LIMITED_RESEARCH — not research-grade.
    """
    closes = load_closes() if closes is None else closes
    manifest = load_manifest()
    store = SnapshotStore(SNAPSHOT_ROOT)
    rows = []
    for sym in closes.columns:
        prev = float(closes[sym].iloc[0])
        for dt, px in closes[sym].items():
            d = pd.Timestamp(dt).strftime("%Y-%m-%d")
            c = float(px)
            rows.append((sym, d, prev, max(prev, c) * 1.001, min(prev, c) * 0.999, c, 1000, "EQ"))
            prev = c
    # Synthetic flat benchmark from equal-weight panel
    ew = closes.mean(axis=1)
    index_rows = []
    prev = float(ew.iloc[0])
    for dt, px in ew.items():
        d = pd.Timestamp(dt).strftime("%Y-%m-%d")
        c = float(px)
        index_rows.append(("PANEL_EW", d, prev, max(prev, c), min(prev, c), c))
        prev = c
    extra = {
        "trust_class": manifest.get("trust_class", "DISPLAY_ONLY"),
        "research_tier": "LIMITED_RESEARCH",
        "research_grade": False,
        "has_universe_history": False,
        "adjustment_consistent": False,
        "corporate_action_coverage": 0.0,
        "missing_session_rate": 0.0,
        "validation_errors": 0,
        "freshness_days": 0,
        "source": manifest.get("source", "yfinance"),
        "phase_a5_note": manifest.get("note", ""),
    }
    sid = store.commit_snapshot(rows, index_rows=index_rows, extra_manifest=extra)
    store.activate_snapshot(sid, actor="phase_a5", reason="exploratory evidence activation")
    pit = PitContract.from_store(store, sid)
    return sid, pit, {**manifest, **extra, "snapshot_id": sid}

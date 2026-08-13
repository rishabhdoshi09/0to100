"""Phase-next dataset + partition helpers (certified snapshot only)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.phase_a5.dataset import load_certified_snapshot, load_sectors
from research.phase_a5.scoped_certification import FROZEN_PANEL
from research.phase_next import protocol as P


@dataclass(frozen=True)
class PanelBundle:
    snapshot_id: str
    closes: pd.DataFrame
    sectors: dict
    manifest: dict
    pit: object


def load_research_panel() -> PanelBundle:
    sid, pit, manifest, closes = load_certified_snapshot(P.SNAPSHOT_ID)
    if sid != P.SNAPSHOT_ID:
        raise ValueError(f"refusing snapshot {sid}")
    if manifest.get("scoped_certification") != "READY_FOR_SCIENTIFIC_RERUN":
        raise ValueError("snapshot not scoped-certified")
    ordered = [s for s in FROZEN_PANEL if s in closes.columns]
    if len(ordered) != 29:
        raise ValueError(f"panel size {len(ordered)} != 29")
    closes = closes[ordered].sort_index()
    return PanelBundle(
        snapshot_id=sid,
        closes=closes,
        sectors=load_sectors(),
        manifest=manifest,
        pit=pit,
    )


def slice_period(closes: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    idx = closes.index
    mask = pd.Series(True, index=idx)
    if start:
        mask &= idx >= pd.Timestamp(start)
    if end:
        mask &= idx <= pd.Timestamp(end)
    return closes.loc[mask]


def period_masks(closes: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
    idx = closes.index
    warmup = idx[idx <= pd.Timestamp(P.WARMUP_END)]
    discovery = idx[
        (idx >= pd.Timestamp(P.DISCOVERY_START)) & (idx <= pd.Timestamp(P.DISCOVERY_END))
    ]
    confirm = idx[idx >= pd.Timestamp(P.CONFIRM_START)]
    return {"warmup": warmup, "discovery": discovery, "confirm": confirm}

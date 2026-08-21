"""Incremental causal VCP state machine.

Every state is computed from bars seen so far. Appending a future bar can
only change the snapshot *after* that bar — never a prior date's state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from research.sepa.config import SepaConfig
from research.sepa.vcp import detect_vcp


STATES = (
    "NO_SETUP",
    "BASE_FORMING",
    "CONTRACTION_1",
    "CONTRACTION_2",
    "VCP_FORMING",
    "PIVOT_DEFINED",
    "ENTRY_READY",
    "BROKEN_OUT",
    "EXTENDED",
    "FAILED",
)


def _iso(ts) -> str:
    try:
        return str(pd.Timestamp(ts).date())
    except Exception:
        return str(ts)


@dataclass
class VcpStateMachine:
    """Feed bars in time order; `snapshot` uses only consumed data."""

    config: SepaConfig
    highs: list[float] = field(default_factory=list)
    lows: list[float] = field(default_factory=list)
    closes: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.volumes.clear()
        self.dates.clear()
        self.history.clear()

    def update(
        self,
        high: float,
        low: float,
        close: float,
        volume: float,
        date,
    ) -> dict[str, Any]:
        self.highs.append(float(high))
        self.lows.append(float(low))
        self.closes.append(float(close))
        self.volumes.append(float(volume))
        self.dates.append(_iso(date))
        snap = self.snapshot()
        self.history.append({
            "date": self.dates[-1],
            "state": snap.get("state"),
            "detected": bool(snap.get("detected")),
            "pivot": snap.get("pivot"),
            "pivot_knowable_date": snap.get("pivot_knowable_date"),
            "vcp_knowable_date": snap.get("vcp_knowable_date"),
        })
        return snap

    def feed_frame(self, frame) -> dict[str, Any]:
        self.reset()
        if frame is None or len(frame) == 0:
            return detect_vcp(None, self.config)
        high = pd.to_numeric(frame["high"] if "high" in frame.columns else frame["close"], errors="coerce")
        low = pd.to_numeric(frame["low"] if "low" in frame.columns else frame["close"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        vol = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else pd.Series(np.ones(len(frame)))
        last = {}
        for i, ts in enumerate(frame.index):
            last = self.update(
                float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i]),
                float(vol.iloc[i]) if vol.iloc[i] == vol.iloc[i] else 0.0, ts,
            )
        return last

    def snapshot(self) -> dict[str, Any]:
        n = len(self.closes)
        if n < 40:
            return detect_vcp(None, self.config)
        lookback = min(int(self.config.vcp_lookback), n)
        idx = pd.DatetimeIndex(self.dates[-lookback:])
        frame = pd.DataFrame(
            {
                "high": self.highs[-lookback:],
                "low": self.lows[-lookback:],
                "close": self.closes[-lookback:],
                "volume": self.volumes[-lookback:],
            },
            index=idx,
        )
        return detect_vcp(frame, self.config)

    def first_detected_date(self) -> str | None:
        for row in self.history:
            if row.get("detected"):
                return row.get("date")
        return None

    def first_state_date(self, state: str) -> str | None:
        for row in self.history:
            if row.get("state") == state:
                return row.get("date")
        return None

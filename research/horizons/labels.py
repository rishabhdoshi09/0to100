"""Leakage-safe forward-return label construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from research.horizons.spec import TargetSpec


@dataclass(frozen=True)
class LabelFrame:
    """Aligned feature/target timestamps with returns and optional class labels."""

    frame: pd.DataFrame
    target: TargetSpec

    @property
    def empty(self) -> bool:
        return self.frame is None or self.frame.empty


def _as_close_series(close) -> pd.Series:
    if isinstance(close, pd.DataFrame):
        if "close" in close.columns:
            s = close["close"]
        else:
            raise ValueError("DataFrame close input must contain a 'close' column")
    else:
        s = pd.Series(close) if not isinstance(close, pd.Series) else close
    s = pd.to_numeric(s, errors="coerce")
    if s.index is None or len(s.index) != len(s):
        s = s.reset_index(drop=True)
    return s.astype(float)


def build_forward_return_labels(
    close,
    target: TargetSpec,
    *,
    benchmark_close=None,
    dates=None,
) -> LabelFrame:
    """Build point-in-time labels for ``target``.

    For each feature bar at index ``i`` (timestamp ``feature_ts``):
      entry = close[i]          (known at feature time)
      exit  = close[i + H]      (realised H bars later)
      raw_return = exit/entry - 1
      net_return = raw_return - cost_pct_roundtrip

    Invariants
    ----------
    * ``target_ts`` is strictly after ``feature_ts`` in the bar sequence (H >= 1).
    * Rows without a realised exit are dropped (no peeking / no fill).
    * When ``benchmark_relative``, excess = asset_return - benchmark_return over
      the *same* window (benchmark series must be aligned by position/index).
    """
    px = _as_close_series(close)
    h = int(target.bars)
    if h < 1:
        raise ValueError("horizon bars must be >= 1")

    if dates is not None:
        date_index = pd.Index(list(dates)[: len(px)])
        if len(date_index) != len(px):
            raise ValueError("dates length must match close length")
        px = pd.Series(px.to_numpy(), index=date_index)
    else:
        date_index = px.index

    # Forward return: pct_change(h).shift(-h) matches ml.multi_horizon / xgboost path.
    # At index i, value = close[i+h]/close[i] - 1.
    raw = px.pct_change(h).shift(-h)

    if target.kind == "benchmark_relative" or target.benchmark_relative:
        if benchmark_close is None:
            raise ValueError("benchmark_close required for benchmark_relative targets")
        bpx = _as_close_series(benchmark_close)
        if len(bpx) != len(px):
            # Align on shared index when possible; else require equal length.
            bpx = bpx.reindex(px.index)
        if bpx.isna().all():
            raise ValueError("benchmark_close could not be aligned to asset close")
        braw = bpx.pct_change(h).shift(-h)
        raw = raw - braw

    net = raw - float(target.cost_pct_roundtrip)

    # target bar index / timestamp
    n = len(px)
    feature_pos = np.arange(n)
    target_pos = feature_pos + h
    valid = target_pos < n

    feature_ts = pd.Series(date_index, index=px.index)
    # Map target timestamps by position
    date_vals = list(date_index)
    target_ts = [
        date_vals[i + h] if (i + h) < n else pd.NaT for i in range(n)
    ]

    frame = pd.DataFrame({
        "feature_ts": feature_ts.to_numpy(),
        "target_ts": target_ts,
        "feature_pos": feature_pos,
        "target_pos": target_pos,
        "entry_px": px.to_numpy(),
        "raw_return": raw.to_numpy(),
        "net_return": net.to_numpy(),
    }, index=px.index)

    frame = frame.loc[valid].copy()
    frame = frame.dropna(subset=["raw_return", "net_return"])

    # Defence: every surviving row must have target_pos > feature_pos
    if not frame.empty and not (frame["target_pos"] > frame["feature_pos"]).all():
        raise AssertionError("label construction produced target_pos <= feature_pos")

    if target.kind == "classification" or (
        target.buy_thresh is not None and target.sell_thresh is not None
    ):
        frame["label"] = classification_from_returns(
            frame["raw_return"],
            buy_thresh=target.buy_thresh,
            sell_thresh=target.sell_thresh,
        )

    return LabelFrame(frame=frame, target=target)


def classification_from_returns(
    returns,
    *,
    buy_thresh: float | None,
    sell_thresh: float | None,
) -> pd.Series:
    """Map continuous returns to {-1, 0, +1} using thresholds.

    Matches ``ml.multi_horizon`` conventions:
      return > buy_thresh  → +1 (BUY)
      return < sell_thresh → -1 (SELL)
      otherwise            →  0 (HOLD)
    """
    if buy_thresh is None or sell_thresh is None:
        raise ValueError("buy_thresh and sell_thresh required for classification")
    r = pd.to_numeric(pd.Series(returns), errors="coerce")
    labels = pd.Series(np.nan, index=r.index, dtype=float)
    labels[r > buy_thresh] = 1.0
    labels[r < sell_thresh] = -1.0
    labels[(r >= sell_thresh) & (r <= buy_thresh)] = 0.0
    return labels


def horizon_agreement(actions: dict[str, str]) -> dict[str, Any]:
    """Simple multi-horizon agreement summary (research reporting helper)."""
    vals = [str(a).upper() for a in actions.values()]
    if not vals:
        return {"action": "HOLD", "agreement": 0, "dispersion": 0.0, "n": 0}
    from collections import Counter
    counts = Counter(vals)
    action, agree = counts.most_common(1)[0]
    dispersion = 1.0 - (agree / len(vals))
    return {
        "action": action,
        "agreement": int(agree),
        "dispersion": round(float(dispersion), 4),
        "n": len(vals),
        "counts": dict(counts),
    }

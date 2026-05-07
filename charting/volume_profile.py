"""
Volume Profile (VPVR) computation from OHLCV data.

Each candle's volume is distributed across price bins in proportion
to how much of the candle's high-low range overlaps each bin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from charting.explanations import explain as _explain
from logger import get_logger

log = get_logger(__name__)


@dataclass
class VPResult:
    prices: List[float]   # bin centre prices
    volumes: List[float]  # volume at each bin
    poc: float            # Point of Control — highest volume bin
    vah: float            # Value Area High (top of 70% volume zone)
    val: float            # Value Area Low  (bottom of 70% volume zone)
    hvn: List[float]      # High Volume Nodes
    lvn: List[float]      # Low Volume Nodes
    bin_height: float     # height of one price bin (for rendering)


class VolumeProfile:
    """
    Compute the Volume Profile (VPVR) from an OHLCV DataFrame.

    Usage
    -----
    result = VolumeProfile().compute(df, bins=50)
    """

    def compute(self, df: pd.DataFrame, bins: int = 50) -> Optional[VPResult]:
        """
        Distribute each candle's volume across price bins and return
        POC, Value Area, HVN, and LVN.
        """
        if len(df) < 2:
            return None

        lo = float(df["low"].min())
        hi = float(df["high"].max())
        if hi <= lo:
            return None

        bin_edges = np.linspace(lo, hi, bins + 1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_vol = np.zeros(bins, dtype=float)

        for _, row in df.iterrows():
            c_lo = float(row["low"])
            c_hi = float(row["high"])
            c_vol = float(row["volume"])
            candle_range = c_hi - c_lo if c_hi > c_lo else 1e-9
            overlap_lo = np.maximum(bin_edges[:-1], c_lo)
            overlap_hi = np.minimum(bin_edges[1:], c_hi)
            overlap = np.maximum(0.0, overlap_hi - overlap_lo)
            bin_vol += (overlap / candle_range) * c_vol

        # Point of Control
        poc_idx = int(np.argmax(bin_vol))
        poc = float(bin_centres[poc_idx])

        # Value Area: expand from POC until 70% of total volume is enclosed
        total_vol = bin_vol.sum()
        target = total_vol * 0.70
        vah_idx = poc_idx
        val_idx = poc_idx
        va_vol = bin_vol[poc_idx]
        while va_vol < target:
            can_up = vah_idx + 1 < bins
            can_dn = val_idx - 1 >= 0
            if can_up and can_dn:
                if bin_vol[vah_idx + 1] >= bin_vol[val_idx - 1]:
                    vah_idx += 1
                    va_vol += bin_vol[vah_idx]
                else:
                    val_idx -= 1
                    va_vol += bin_vol[val_idx]
            elif can_up:
                vah_idx += 1
                va_vol += bin_vol[vah_idx]
            elif can_dn:
                val_idx -= 1
                va_vol += bin_vol[val_idx]
            else:
                break

        vah = float(bin_centres[vah_idx])
        val = float(bin_centres[val_idx])

        # High / Low Volume Nodes
        mean_v = bin_vol.mean()
        std_v = bin_vol.std()
        hvn = [float(bin_centres[i]) for i in range(bins) if bin_vol[i] > mean_v + 0.5 * std_v]
        lvn = [float(bin_centres[i]) for i in range(bins) if bin_vol[i] < mean_v - 0.5 * std_v]

        bin_height = float(bin_edges[1] - bin_edges[0])

        log.debug("volume_profile_computed", bins=bins, poc=round(poc, 2),
                  vah=round(vah, 2), val=round(val, 2),
                  hvn_count=len(hvn), lvn_count=len(lvn))

        return VPResult(
            prices=bin_centres.tolist(),
            volumes=bin_vol.tolist(),
            poc=poc,
            vah=vah,
            val=val,
            hvn=hvn,
            lvn=lvn,
            bin_height=bin_height,
        )

    @staticmethod
    def explain() -> str:
        return _explain("volume_profile")

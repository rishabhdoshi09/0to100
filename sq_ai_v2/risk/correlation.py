"""
Correlation-aware position sizing.

When two positions are highly correlated (e.g., RELIANCE and ONGC — both
energy sector), the second position adds little diversification.
We penalise the size of the new position proportional to its correlation
with the existing book.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class CorrelationManager:
    def __init__(
        self,
        window: int = 60,
        high_corr_threshold: float = 0.70,
        max_penalty: float = 0.5,
    ) -> None:
        self.window = window
        self.high_corr_threshold = high_corr_threshold
        self.max_penalty = max_penalty
        self._returns_cache: Dict[str, pd.Series] = {}
        self._corr_matrix: Optional[pd.DataFrame] = None

    # ── Update correlation matrix ─────────────────────────────────────────

    def update(self, symbol: str, close: pd.Series) -> None:
        """Ingest latest close prices for a symbol."""
        log_ret = np.log(close / close.shift(1)).dropna()
        self._returns_cache[symbol] = log_ret.tail(self.window)
        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        if len(self._returns_cache) < 2:
            self._corr_matrix = None
            return
        df = pd.DataFrame(self._returns_cache).dropna()
        if len(df) < 10:
            self._corr_matrix = None
            return
        self._corr_matrix = df.corr()

    # ── Penalty computation ───────────────────────────────────────────────

    def correlation_penalty(
        self,
        new_symbol: str,
        open_symbols: List[str],
    ) -> float:
        """
        Returns a multiplier in [max_penalty, 1.0].
        1.0 → no penalty (new symbol uncorrelated with book).
        0.5 → maximum penalty (highly correlated with existing positions).
        """
        if not open_symbols or self._corr_matrix is None:
            return 1.0

        if new_symbol not in self._corr_matrix.columns:
            return 1.0

        corrs = []
        for sym in open_symbols:
            if sym in self._corr_matrix.columns:
                c = abs(float(self._corr_matrix.loc[new_symbol, sym]))
                corrs.append(c)

        if not corrs:
            return 1.0

        max_corr = max(corrs)

        # Linear penalty between threshold and 1.0
        if max_corr <= self.high_corr_threshold:
            return 1.0

        excess = (max_corr - self.high_corr_threshold) / (1 - self.high_corr_threshold)
        penalty = 1.0 - excess * (1 - self.max_penalty)
        penalty = float(np.clip(penalty, self.max_penalty, 1.0))
        logger.debug(f"Correlation penalty: {new_symbol} max_corr={max_corr:.2f} → {penalty:.2f}")
        return penalty

    # ── Correlation matrix accessor ───────────────────────────────────────

    def get_matrix(self) -> Optional[pd.DataFrame]:
        return self._corr_matrix

    def get_pairwise(self, sym_a: str, sym_b: str) -> float:
        if self._corr_matrix is None:
            return 0.0
        try:
            return float(self._corr_matrix.loc[sym_a, sym_b])
        except KeyError:
            return 0.0

    # ── Cluster detection ─────────────────────────────────────────────────

    def find_clusters(self, threshold: float = 0.70) -> List[List[str]]:
        """
        Return groups of highly correlated symbols.
        Simple greedy clustering.
        """
        if self._corr_matrix is None:
            return []

        symbols = list(self._corr_matrix.columns)
        visited = set()
        clusters = []

        for s in symbols:
            if s in visited:
                continue
            cluster = [s]
            visited.add(s)
            for t in symbols:
                if t not in visited and abs(self._corr_matrix.loc[s, t]) >= threshold:
                    cluster.append(t)
                    visited.add(t)
            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

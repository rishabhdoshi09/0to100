"""
Graph Neural Network for modelling inter-stock correlations.

Graph construction:
  • One node per symbol, each node's features = last-bar normalised returns + features.
  • Edges = Pearson correlation > threshold over rolling window.
  • Message passing via GCNConv (torch_geometric).
  • Output: per-node P(up) ∈ [0, 1].

Falls back gracefully if torch_geometric is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from config.settings import settings

# torch_geometric is optional
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    _GEOMETRIC_AVAILABLE = True
except ImportError:
    _GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not installed — GNN will use MLP fallback")


class GCNNet(nn.Module):
    def __init__(self, node_features: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = torch.relu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


class MLPFallback(nn.Module):
    """Used when torch_geometric is unavailable."""

    def __init__(self, node_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, _edge_index=None) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GNNWrapper:
    """
    Trains and runs GCN (or MLP fallback) over a multi-symbol graph.
    Each call to predict() re-builds the graph from current correlations.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        corr_window: int = 60,
        corr_threshold: float = 0.4,
        hidden: int = 64,
        lr: float = 1e-3,
        epochs: int = 20,
        model_path: Optional[Path] = None,
    ) -> None:
        self.symbols = symbols or settings.symbol_list
        self.corr_window = corr_window
        self.corr_threshold = corr_threshold
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self._model_path = model_path or (settings.model_dir / "gnn.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: Optional[nn.Module] = None
        self.n_features = 0
        if self._model_path.exists():
            self.load()

    # ── Graph construction ────────────────────────────────────────────────

    def _build_graph(
        self, feature_matrix: np.ndarray, returns_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feature_matrix: (n_symbols, n_features)  — node features
        returns_matrix: (n_symbols, corr_window) — for edge construction
        """
        corr = np.corrcoef(returns_matrix)
        n = len(self.symbols)

        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j and corr[i, j] > self.corr_threshold:
                    src.append(i)
                    dst.append(j)

        if not src:
            # Fully disconnected — add self-loops only
            src = list(range(n))
            dst = list(range(n))

        x = torch.tensor(feature_matrix, dtype=torch.float32)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return x, edge_index

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        feature_dict: Dict[str, pd.DataFrame],
        label_dict: Dict[str, pd.Series],
    ) -> None:
        """
        feature_dict: symbol → feature DataFrame (aligned dates)
        label_dict:   symbol → binary label Series
        """
        dates = None
        for sym in self.symbols:
            if sym in feature_dict:
                dates = feature_dict[sym].index
                break
        if dates is None:
            logger.warning("GNN: no data")
            return

        self.n_features = next(iter(feature_dict.values())).shape[1]
        net_cls = GCNNet if _GEOMETRIC_AVAILABLE else MLPFallback
        self._net = net_cls(self.n_features, self.hidden).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        self._net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            count = 0
            for t in range(self.corr_window, len(dates)):
                ts = dates[t]
                feat_rows, return_rows, labels = [], [], []
                for sym in self.symbols:
                    if sym not in feature_dict or sym not in label_dict:
                        continue
                    fdf = feature_dict[sym]
                    if ts not in fdf.index:
                        continue
                    row = fdf.loc[ts].fillna(0).values
                    feat_rows.append(row)
                    past_idx = fdf.index[max(0, t - self.corr_window): t]
                    ret = fdf.loc[past_idx, "return_1d"].fillna(0).values if "return_1d" in fdf.columns else np.zeros(self.corr_window)
                    return_rows.append(ret[-self.corr_window:] if len(ret) >= self.corr_window else np.pad(ret, (self.corr_window - len(ret), 0)))
                    lbl = label_dict[sym].get(ts, np.nan)
                    labels.append(lbl)

                if len(feat_rows) < 2:
                    continue
                feat_matrix = np.array(feat_rows, dtype=np.float32)
                ret_matrix = np.array(return_rows, dtype=np.float32)
                label_arr = np.array(labels, dtype=np.float32)
                valid = ~np.isnan(label_arr)
                if valid.sum() == 0:
                    continue

                x, edge_index = self._build_graph(feat_matrix, ret_matrix)
                x = x.to(self.device)
                edge_index = edge_index.to(self.device)
                y = torch.tensor(label_arr[valid], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                preds = self._net(x, edge_index)[valid]
                loss = criterion(preds, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                count += 1

            if count > 0 and (epoch + 1) % 5 == 0:
                logger.debug(f"GNN epoch {epoch+1}/{self.epochs} avg_loss={total_loss/count:.4f}")

        logger.info("GNN training complete")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(
        self,
        feature_dict: Dict[str, pd.Series],
        return_history: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        feature_dict: symbol → latest feature Series
        return_history: symbol → array of last corr_window returns
        Returns: symbol → P(up)
        """
        if self._net is None:
            return {sym: 0.5 for sym in self.symbols}

        feat_rows, ret_rows, syms_present = [], [], []
        for sym in self.symbols:
            if sym in feature_dict:
                feat_rows.append(feature_dict[sym].fillna(0).values)
                ret = return_history.get(sym, np.zeros(self.corr_window))
                ret_rows.append(ret[-self.corr_window:] if len(ret) >= self.corr_window else np.pad(ret, (self.corr_window - len(ret), 0)))
                syms_present.append(sym)

        if not feat_rows:
            return {sym: 0.5 for sym in self.symbols}

        feat_matrix = np.array(feat_rows, dtype=np.float32)
        ret_matrix = np.array(ret_rows, dtype=np.float32)
        x, edge_index = self._build_graph(feat_matrix, ret_matrix)

        self._net.eval()
        with torch.no_grad():
            probs = self._net(x.to(self.device), edge_index.to(self.device)).cpu().numpy()

        return {sym: float(probs[i]) for i, sym in enumerate(syms_present)}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net:
            torch.save({"state_dict": self._net.state_dict(), "n_features": self.n_features}, path)
        logger.info(f"GNN saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.n_features = ckpt["n_features"]
            net_cls = GCNNet if _GEOMETRIC_AVAILABLE else MLPFallback
            self._net = net_cls(self.n_features, self.hidden).to(self.device)
            self._net.load_state_dict(ckpt["state_dict"])
            self._net.eval()
            logger.info(f"GNN loaded ← {path}")
        except Exception as exc:
            logger.warning(f"GNN load failed: {exc}")
            self._net = None

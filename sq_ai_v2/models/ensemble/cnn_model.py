"""
1-D CNN for short-term pattern recognition.
Input: (batch, channels, window) — last `window` bars of normalised features.
Output: scalar in [0, 1] = P(up next bar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from config.settings import settings


class CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNN1DModel(nn.Module):
    """
    Stacked 1-D CNN with global average pooling → binary output.
    """

    def __init__(self, n_features: int = 36, window: int = 20, dropout: float = 0.3) -> None:
        super().__init__()
        self.window = window
        self.n_features = n_features

        self.encoder = nn.Sequential(
            CNNBlock(n_features, 64, 3),
            CNNBlock(64, 128, 3),
            CNNBlock(128, 64, 3),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, window)
        h = self.encoder(x)          # (batch, 64, window)
        h = h.mean(dim=-1)           # global average pool → (batch, 64)
        return self.head(h).squeeze(-1)


class CNNWrapper:
    """
    Scikit-learn–style wrapper around CNN1DModel.
    """

    def __init__(
        self,
        n_features: int = 36,
        window: int = 20,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 64,
        model_path: Optional[Path] = None,
    ) -> None:
        self.window = window
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model_path = model_path or (settings.model_dir / "cnn.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: Optional[CNN1DModel] = None
        if self._model_path.exists():
            self.load()

    def _build(self) -> None:
        self._net = CNN1DModel(self.n_features, self.window).to(self.device)

    # ── Data preparation ──────────────────────────────────────────────────

    def _to_windows(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert (T, F) array to (T-window, F, window) windows."""
        n, f = X.shape
        X_win = np.stack(
            [X[i : i + self.window].T for i in range(n - self.window)],
            axis=0,
        ).astype(np.float32)
        y_win = y[self.window:].astype(np.float32) if y is not None else None
        return X_win, y_win

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.feature_names_ = list(X.columns)
        X_np = X.values
        y_np = y.values

        X_win, y_win = self._to_windows(X_np, y_np)
        if len(X_win) < 32:
            logger.warning("CNN: not enough training samples")
            return

        self._build()
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_win), torch.from_numpy(y_win)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        self._net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self._net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"CNN epoch {epoch+1}/{self.epochs} loss={total_loss/len(loader):.4f}")

        logger.info("CNN training complete")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            return np.full(len(X), 0.5)

        X_np = X.values.astype(np.float32)
        if len(X_np) < self.window:
            return np.full(len(X_np), 0.5)

        X_win, _ = self._to_windows(X_np)
        self._net.eval()
        with torch.no_grad():
            probs = self._net(torch.from_numpy(X_win).to(self.device)).cpu().numpy()

        # Pad beginning with 0.5 (no prediction for warm-up period)
        padding = np.full(self.window, 0.5)
        return np.concatenate([padding, probs])

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net is not None:
            torch.save(
                {
                    "state_dict": self._net.state_dict(),
                    "n_features": self.n_features,
                    "window": self.window,
                },
                path,
            )
        logger.info(f"CNN saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.n_features = ckpt["n_features"]
            self.window = ckpt["window"]
            self._build()
            self._net.load_state_dict(ckpt["state_dict"])
            self._net.eval()
            logger.info(f"CNN loaded ← {path}")
        except Exception as exc:
            logger.warning(f"CNN load failed: {exc}")
            self._net = None

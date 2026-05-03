"""
Bidirectional LSTM with attention for longer-term pattern recognition.
Input: (batch, seq_len, features)
Output: P(up next bar) ∈ [0, 1]
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


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden*2)
        return context


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = AttentionLayer(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)           # (batch, seq, hidden*2)
        context = self.attn(out)        # (batch, hidden*2)
        context = self.dropout(context)
        return self.head(context).squeeze(-1)


class LSTMWrapper:
    """Scikit-learn–style wrapper for LSTMNet."""

    def __init__(
        self,
        seq_len: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        lr: float = 5e-4,
        epochs: int = 30,
        batch_size: int = 32,
        model_path: Optional[Path] = None,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model_path = model_path or (settings.model_dir / "lstm.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net: Optional[LSTMNet] = None
        self.n_features = 0
        if self._model_path.exists():
            self.load()

    def _build(self) -> None:
        self._net = LSTMNet(self.n_features, self.hidden_size, self.num_layers).to(
            self.device
        )

    # ── Data prep ─────────────────────────────────────────────────────────

    def _to_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n = len(X)
        X_seq = np.stack(
            [X[i : i + self.seq_len] for i in range(n - self.seq_len)],
            axis=0,
        ).astype(np.float32)
        y_seq = y[self.seq_len:].astype(np.float32) if y is not None else None
        return X_seq, y_seq

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.n_features = X.shape[1]
        X_np, y_np = self._to_sequences(X.values, y.values)

        if len(X_np) < 32:
            logger.warning("LSTM: not enough samples")
            return

        self._build()
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=max(1, len(X_np) // self.batch_size)
        )
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_np), torch.from_numpy(y_np)
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
                scheduler.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.debug(
                    f"LSTM epoch {epoch+1}/{self.epochs} loss={total_loss/len(loader):.4f}"
                )

        logger.info("LSTM training complete")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._net is None:
            return np.full(len(X), 0.5)

        X_np = X.values.astype(np.float32)
        if len(X_np) < self.seq_len:
            return np.full(len(X_np), 0.5)

        X_seq, _ = self._to_sequences(X_np)
        self._net.eval()
        with torch.no_grad():
            probs = self._net(torch.from_numpy(X_seq).to(self.device)).cpu().numpy()

        padding = np.full(self.seq_len, 0.5)
        return np.concatenate([padding, probs])

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._net:
            torch.save(
                {
                    "state_dict": self._net.state_dict(),
                    "n_features": self.n_features,
                    "seq_len": self.seq_len,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                },
                path,
            )
        logger.info(f"LSTM saved → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        path = path or self._model_path
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.n_features = ckpt["n_features"]
            self.seq_len = ckpt["seq_len"]
            self.hidden_size = ckpt["hidden_size"]
            self.num_layers = ckpt["num_layers"]
            self._build()
            self._net.load_state_dict(ckpt["state_dict"])
            self._net.eval()
            logger.info(f"LSTM loaded ← {path}")
        except Exception as exc:
            logger.warning(f"LSTM load failed: {exc}")
            self._net = None

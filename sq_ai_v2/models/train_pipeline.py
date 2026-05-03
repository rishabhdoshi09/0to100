"""
End-to-end training pipeline.

Steps:
  1. Load features from FeatureStore for all symbols.
  2. Build labels (next-bar direction).
  3. Time-series split into train / validation folds.
  4. Train each base model (LightGBM, CNN, LSTM).
  5. Collect OOS predictions → train MetaLearner.
  6. Calibrate MetaLearner output on a held-out calibration set.
  7. Train HMM regime model.
  8. Save all artefacts.
  9. Register model versions in PostgreSQL.
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import settings
from data.feature_store.store import FeatureStore
from data.storage.postgres_client import PostgresClient
from models.calibration import CalibrationWrapper
from models.ensemble.cnn_model import CNNWrapper
from models.ensemble.hmm_regime import HMMRegime
from models.ensemble.lightgbm_model import LightGBMModel
from models.ensemble.lstm_model import LSTMWrapper
from models.meta_learner import MetaLearner

warnings.filterwarnings("ignore")


class TrainingPipeline:
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._dir = model_dir or settings.model_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._feature_store = FeatureStore()
        self._pg = PostgresClient()

    # ── Public entry point ─────────────────────────────────────────────────

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        data: {symbol: OHLCV DataFrame} — full history.
        Returns dict of validation metrics.
        """
        logger.info(f"Training pipeline started: {len(data)} symbols")

        # 1. Compute features
        feat_dict: Dict[str, pd.DataFrame] = {}
        label_dict: Dict[str, pd.Series] = {}
        for sym, df in data.items():
            fdf = self._feature_store.compute(sym, df)
            fdf = fdf.replace([np.inf, -np.inf], np.nan).dropna()
            labels = LightGBMModel.make_labels(df.loc[fdf.index, "close"], forward_bars=1)
            labels = labels.reindex(fdf.index).dropna()
            fdf = fdf.loc[labels.index]
            feat_dict[sym] = fdf
            label_dict[sym] = labels

        # 2. Concatenate across symbols for cross-sectional training
        X_all = pd.concat(list(feat_dict.values()), axis=0).sort_index()
        y_all = pd.concat(list(label_dict.values()), axis=0).sort_index()
        X_all, y_all = X_all.align(y_all, join="inner", axis=0)

        logger.info(f"Training matrix: {X_all.shape}")

        # 3. Normalise features
        scaler = StandardScaler()
        X_norm = pd.DataFrame(
            scaler.fit_transform(X_all.fillna(0)),
            index=X_all.index,
            columns=X_all.columns,
        )
        # Save scaler
        import pickle
        with open(self._dir / "feature_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # 4. Train / val split (80 / 20, time-ordered)
        split = int(len(X_norm) * 0.8)
        cal_split = int(len(X_norm) * 0.9)

        X_train, y_train = X_norm.iloc[:split], y_all.iloc[:split]
        X_val, y_val = X_norm.iloc[split:cal_split], y_all.iloc[split:cal_split]
        X_cal, y_cal = X_norm.iloc[cal_split:], y_all.iloc[cal_split:]

        metrics: Dict = {}

        # 5. Train base models
        lgbm = LightGBMModel(self._dir / "lgbm.pkl")
        lgbm_metrics = lgbm.fit(
            X_train, y_train, eval_set=(X_val, y_val), num_rounds=500
        )
        lgbm.save()
        metrics["lgbm"] = lgbm_metrics

        cnn = CNNWrapper(n_features=X_norm.shape[1], model_path=self._dir / "cnn.pt")
        cnn.fit(X_train, y_train)
        cnn.save()

        lstm = LSTMWrapper(model_path=self._dir / "lstm.pt")
        lstm.fit(X_train, y_train)
        lstm.save()

        # 6. Collect OOS predictions for meta-learner
        lgbm_val_preds = lgbm.predict_proba(X_val)
        cnn_val_preds = cnn.predict_proba(X_val)[-len(X_val):]
        lstm_val_preds = lstm.predict_proba(X_val)[-len(X_val):]

        oos_preds = {
            "lgbm": lgbm_val_preds,
            "cnn": cnn_val_preds,
            "lstm": lstm_val_preds,
        }

        meta = MetaLearner(self._dir / "meta_learner.pkl")
        meta_metrics = meta.fit(oos_preds, y_val.values)
        meta.save()
        metrics["meta"] = meta_metrics

        # 7. Calibrate on calibration set
        meta_cal_raw = []
        for i in range(len(X_cal)):
            row = X_cal.iloc[i]
            base = {
                "lgbm": float(lgbm.predict_proba(row.to_frame().T)[0]),
                "cnn": float(cnn.predict_proba(row.to_frame().T)[-1]),
                "lstm": float(lstm.predict_proba(row.to_frame().T)[-1]),
            }
            meta_cal_raw.append(meta.predict_proba(base))

        cal = CalibrationWrapper(method="isotonic", model_path=self._dir / "calibrator.pkl")
        cal.fit(np.array(meta_cal_raw), y_cal.values)
        cal.save()
        metrics["calibration"] = {"samples": len(y_cal)}

        # 8. Train HMM
        hmm_model = HMMRegime(self._dir / "hmm.pkl")
        hmm_model.fit(data)
        hmm_model.save()

        # 9. Register in Postgres
        version = date.today().strftime("%Y%m%d")
        val_acc = float(meta_metrics.get("accuracy", 0))
        val_sharpe = self._estimate_sharpe(lgbm_val_preds, y_val.values)
        try:
            self._pg.register_model(
                model_name="ensemble_v2",
                version=version,
                train_start=X_train.index.min(),
                train_end=X_train.index.max(),
                val_sharpe=val_sharpe,
                val_accuracy=val_acc,
                artifact_path=str(self._dir),
            )
        except Exception as exc:
            logger.warning(f"PostgreSQL model registration skipped: {exc}")

        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    # ── Walk-forward delegated to walk_forward.py ─────────────────────────

    @staticmethod
    def _estimate_sharpe(probs: np.ndarray, y_true: np.ndarray) -> float:
        """Rough Sharpe from signal-weighted returns (simulation)."""
        signals = np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0))
        pnl = signals * (2 * y_true - 1)
        if pnl.std() == 0:
            return 0.0
        return float(pnl.mean() / pnl.std() * np.sqrt(252))

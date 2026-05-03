"""Local training script – same logic as colab_train.ipynb, no Colab needed.

Usage (from ~/0to100):
    python -m sq_ai.train.train_local

Outputs saved to ./models/:
    lgb_trading_model.pkl
    feature_names.txt
"""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import roc_auc_score

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
]

FEATURES = ["sma_20", "sma_50", "volatility_20", "momentum_5d",
            "volume_trend", "rsi", "atr", "regime"]

TRAIN_END = "2022-01-01"    # walk-forward split: train < 2022, test ≥ 2022
DATA_START = "2018-01-01"
DATA_END = "2024-01-01"

OUT_DIR = Path(__file__).resolve().parents[2] / "models"


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                   for c in out.columns]
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    ret = out["close"].pct_change()
    out["volatility_20"] = ret.rolling(20).std() * np.sqrt(252)
    out["momentum_5d"] = out["close"].pct_change(5)
    out["volume_trend"] = out["volume"] / out["volume"].rolling(20).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    tr = pd.concat([
        (out["high"] - out["low"]),
        (out["high"] - out["close"].shift()).abs(),
        (out["low"] - out["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["regime"] = np.where(
        out["sma_20"] > out["sma_50"] * 1.005, 2,
        np.where(out["sma_20"] < out["sma_50"] * 0.995, 0, 1),
    )
    # label: 1 if close is up >1 % in 5 trading days
    out["target"] = (out["close"].shift(-5) / out["close"] - 1 > 0.01).astype(int)
    return out.dropna()


def build_dataset(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        print(f"  downloading {sym} …", end=" ", flush=True)
        raw = yf.download(sym, start=DATA_START, end=DATA_END,
                          progress=False, auto_adjust=False)
        if len(raw) < 200:
            print(f"skip (only {len(raw)} rows)")
            continue
        feat = _engineer(raw)
        feat["symbol"] = sym
        frames.append(feat)
        print(f"{len(feat)} rows")
    if not frames:
        raise RuntimeError("No data downloaded — check network / symbol list.")
    return pd.concat(frames).sort_index()


def train(all_df: pd.DataFrame) -> lgb.LGBMClassifier:
    train_df = all_df[all_df.index < TRAIN_END]
    test_df = all_df[all_df.index >= TRAIN_END]
    X_tr, y_tr = train_df[FEATURES], train_df["target"]
    X_te, y_te = test_df[FEATURES], test_df["target"]

    print(f"\nTrain rows: {len(X_tr):,}  |  Test rows: {len(X_te):,}")
    print(f"Train label balance: {y_tr.mean():.1%} positive")

    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=63,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])

    p_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    print(f"OOS AUC: {auc:.4f}  (target > 0.55)")
    if auc < 0.55:
        print("WARNING: AUC below 0.55 — model adds little information over base rate.")
    return clf


def save(clf: lgb.LGBMClassifier) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "lgb_trading_model.pkl"
    feat_path = OUT_DIR / "feature_names.txt"
    joblib.dump(clf, model_path)
    feat_path.write_text("\n".join(FEATURES))
    print(f"\nSaved:\n  {model_path}\n  {feat_path}")
    print("Restart the cockpit — it will pick up the new model on the next cycle.")


def main() -> None:
    print("=== sq_ai LightGBM training ===")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Data range: {DATA_START} → {DATA_END}")
    print(f"Walk-forward split: train < {TRAIN_END}, test ≥ {TRAIN_END}\n")

    print("Downloading data …")
    all_df = build_dataset(SYMBOLS)
    print(f"\nTotal rows: {len(all_df):,}")

    clf = train(all_df)
    save(clf)


if __name__ == "__main__":
    main()
